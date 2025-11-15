"""Tests for AgentManager component."""

import uuid
from typing import List

import numpy as np
import pytest

from rl_environments.ticket_routing.core.data_models import (
    AgentConfig,
    Ticket,
    TicketType,
    Priority,
    SupportAgent,
)
from rl_environments.ticket_routing.core.agent_manager import AgentManager


@pytest.fixture
def default_config() -> AgentConfig:
    """Provide default agent configuration."""
    return AgentConfig(max_capacity=10, expertise_range=(0.3, 0.9))


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def agent_manager(default_config: AgentConfig, rng: np.random.Generator) -> AgentManager:
    """Provide initialized agent manager."""
    return AgentManager(num_agents=5, config=default_config, rng=rng)


@pytest.fixture
def sample_ticket() -> Ticket:
    """Provide sample ticket for testing."""
    return Ticket(
        id=uuid.uuid4(),
        type=TicketType.TECHNICAL,
        priority=Priority.HIGH,
        arrival_step=0,
    )


class TestAgentManagerInitialization:
    """Tests for AgentManager initialization."""
    
    def test_initialization_valid(
        self, default_config: AgentConfig, rng: np.random.Generator
    ) -> None:
        """Test valid initialization."""
        manager = AgentManager(num_agents=5, config=default_config, rng=rng)
        
        assert manager.num_agents == 5
        assert len(manager.agents) == 5
        assert len(manager.active_assignments) == 5
        assert manager.config == default_config
    
    def test_initialization_invalid_num_agents(
        self, default_config: AgentConfig, rng: np.random.Generator
    ) -> None:
        """Test initialization with invalid number of agents."""
        with pytest.raises(ValueError, match="num_agents must be at least 1"):
            AgentManager(num_agents=0, config=default_config, rng=rng)
    
    def test_agents_have_valid_expertise(self, agent_manager: AgentManager) -> None:
        """Test that all agents have valid expertise profiles."""
        for agent in agent_manager.agents:
            assert agent.expertise.shape == (len(TicketType),)
            assert np.all(agent.expertise >= 0.3)
            assert np.all(agent.expertise <= 0.9)
    
    def test_agents_start_with_zero_workload(self, agent_manager: AgentManager) -> None:
        """Test that agents start with zero workload."""
        for agent in agent_manager.agents:
            assert agent.current_workload == 0
            assert agent.is_available()
    
    def test_deterministic_initialization(
        self, default_config: AgentConfig
    ) -> None:
        """Test that seeding produces deterministic agent initialization."""
        rng1 = np.random.default_rng(123)
        manager1 = AgentManager(num_agents=3, config=default_config, rng=rng1)
        
        rng2 = np.random.default_rng(123)
        manager2 = AgentManager(num_agents=3, config=default_config, rng=rng2)
        
        for agent1, agent2 in zip(manager1.agents, manager2.agents):
            np.testing.assert_array_equal(agent1.expertise, agent2.expertise)


class TestTicketAssignment:
    """Tests for ticket assignment functionality."""
    
    def test_assign_ticket_valid(
        self, agent_manager: AgentManager, sample_ticket: Ticket
    ) -> None:
        """Test valid ticket assignment."""
        assignment = agent_manager.assign_ticket(
            ticket=sample_ticket,
            agent_id=0,
            current_step=100,
        )
        
        assert assignment.ticket_id == sample_ticket.id
        assert assignment.agent_id == 0
        assert assignment.assignment_step == 100
        assert assignment.estimated_resolution_time > 0
        
        # Check agent workload increased
        agent = agent_manager.agents[0]
        assert agent.current_workload == 1
        
        # Check assignment tracked
        assert len(agent_manager.active_assignments[0]) == 1
        assert agent_manager.active_assignments[0][0].id == assignment.id
    
    def test_assign_ticket_invalid_agent_id(
        self, agent_manager: AgentManager, sample_ticket: Ticket
    ) -> None:
        """Test assignment with invalid agent ID."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            agent_manager.assign_ticket(
                ticket=sample_ticket,
                agent_id=-1,
                current_step=0,
            )
        
        with pytest.raises(ValueError, match="Invalid agent_id"):
            agent_manager.assign_ticket(
                ticket=sample_ticket,
                agent_id=999,
                current_step=0,
            )
    
    def test_assign_ticket_to_full_agent(
        self, agent_manager: AgentManager
    ) -> None:
        """Test assignment to agent at capacity."""
        agent_id = 0
        
        # Fill agent to capacity
        for i in range(10):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            agent_manager.assign_ticket(ticket, agent_id, current_step=0)
        
        # Verify agent is at capacity
        assert not agent_manager.agents[agent_id].is_available()
        
        # Try to assign one more ticket
        extra_ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.LOW,
            arrival_step=0,
        )
        
        with pytest.raises(ValueError, match="is at capacity"):
            agent_manager.assign_ticket(extra_ticket, agent_id, current_step=0)
    
    def test_multiple_assignments(self, agent_manager: AgentManager) -> None:
        """Test multiple ticket assignments across agents."""
        tickets = [
            Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.HIGH,
                arrival_step=i,
            )
            for i in range(15)
        ]
        
        # Assign tickets round-robin
        for i, ticket in enumerate(tickets):
            agent_id = i % 5
            agent_manager.assign_ticket(ticket, agent_id, current_step=i)
        
        # Each agent should have 3 assignments
        for agent in agent_manager.agents:
            assert agent.current_workload == 3
        
        assert agent_manager.get_total_assignments() == 15


class TestWorkloadManagement:
    """Tests for workload tracking and updates."""
    
    def test_update_workloads_no_completions(
        self, agent_manager: AgentManager
    ) -> None:
        """Test workload update when nothing completes."""
        completed = agent_manager.update_workloads(current_step=100)
        assert completed == 0
    
    def test_update_workloads_with_completions(
        self, agent_manager: AgentManager
    ) -> None:
        """Test workload update when assignments complete."""
        # Assign ticket to agent 0
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        assignment = agent_manager.assign_ticket(ticket, agent_id=0, current_step=0)
        
        # Check initial state
        assert agent_manager.agents[0].current_workload == 1
        assert len(agent_manager.active_assignments[0]) == 1
        
        # Calculate completion step
        # resolution_time is in hours, convert to steps (1000 steps = 24 hours)
        steps_to_complete = int(assignment.estimated_resolution_time * (1000.0 / 24.0))
        completion_step = steps_to_complete
        
        # Update workload at various steps before completion
        for step in range(completion_step):
            completed = agent_manager.update_workloads(step)
            assert completed == 0
            assert agent_manager.agents[0].current_workload == 1
        
        # Update at completion step
        completed = agent_manager.update_workloads(completion_step)
        assert completed == 1
        assert agent_manager.agents[0].current_workload == 0
        assert len(agent_manager.active_assignments[0]) == 0
    
    def test_get_agent_workloads(self, agent_manager: AgentManager) -> None:
        """Test getting agent workload array."""
        # Assign different numbers of tickets to agents
        workloads = [1, 3, 2, 0, 4]
        
        for agent_id, num_tickets in enumerate(workloads):
            for _ in range(num_tickets):
                ticket = Ticket(
                    id=uuid.uuid4(),
                    type=TicketType.TECHNICAL,
                    priority=Priority.MEDIUM,
                    arrival_step=0,
                )
                agent_manager.assign_ticket(ticket, agent_id, current_step=0)
        
        workload_array = agent_manager.get_agent_workloads()
        assert workload_array.shape == (5,)
        np.testing.assert_array_equal(workload_array, np.array(workloads))
    
    def test_get_workload_statistics(self, agent_manager: AgentManager) -> None:
        """Test workload statistics calculation."""
        # Assign tickets to create varied workload
        workloads = [2, 4, 3, 1, 5]
        
        for agent_id, num_tickets in enumerate(workloads):
            for _ in range(num_tickets):
                ticket = Ticket(
                    id=uuid.uuid4(),
                    type=TicketType.TECHNICAL,
                    priority=Priority.MEDIUM,
                    arrival_step=0,
                )
                agent_manager.assign_ticket(ticket, agent_id, current_step=0)
        
        stats = agent_manager.get_workload_statistics()
        
        assert stats['mean'] == 3.0
        assert stats['std'] == pytest.approx(1.4142, rel=1e-3)
        assert stats['min'] == 1
        assert stats['max'] == 5


class TestAvailability:
    """Tests for agent availability checking."""
    
    def test_get_available_agents_all_available(
        self, agent_manager: AgentManager
    ) -> None:
        """Test getting available agents when all are free."""
        available = agent_manager.get_available_agents()
        assert len(available) == 5
        assert available == [0, 1, 2, 3, 4]
    
    def test_get_available_agents_some_busy(
        self, agent_manager: AgentManager
    ) -> None:
        """Test getting available agents when some are at capacity."""
        # Fill agent 0 and 2 to capacity
        for agent_id in [0, 2]:
            for _ in range(10):
                ticket = Ticket(
                    id=uuid.uuid4(),
                    type=TicketType.TECHNICAL,
                    priority=Priority.MEDIUM,
                    arrival_step=0,
                )
                agent_manager.assign_ticket(ticket, agent_id, current_step=0)
        
        available = agent_manager.get_available_agents()
        assert len(available) == 3
        assert set(available) == {1, 3, 4}
    
    def test_get_available_agents_all_busy(
        self, agent_manager: AgentManager
    ) -> None:
        """Test getting available agents when all are at capacity."""
        # Fill all agents to capacity
        for agent_id in range(5):
            for _ in range(10):
                ticket = Ticket(
                    id=uuid.uuid4(),
                    type=TicketType.TECHNICAL,
                    priority=Priority.MEDIUM,
                    arrival_step=0,
                )
                agent_manager.assign_ticket(ticket, agent_id, current_step=0)
        
        available = agent_manager.get_available_agents()
        assert len(available) == 0


class TestAgentRetrieval:
    """Tests for agent retrieval methods."""
    
    def test_get_agent_valid(self, agent_manager: AgentManager) -> None:
        """Test getting agent by valid ID."""
        agent = agent_manager.get_agent(0)
        assert isinstance(agent, SupportAgent)
        assert agent.id == 0
    
    def test_get_agent_invalid_id(self, agent_manager: AgentManager) -> None:
        """Test getting agent with invalid ID."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            agent_manager.get_agent(-1)
        
        with pytest.raises(ValueError, match="Invalid agent_id"):
            agent_manager.get_agent(999)
    
    def test_get_total_assignments(self, agent_manager: AgentManager) -> None:
        """Test getting total assignment count."""
        assert agent_manager.get_total_assignments() == 0
        
        # Assign some tickets
        for i in range(7):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            agent_manager.assign_ticket(ticket, agent_id=i % 5, current_step=0)
        
        assert agent_manager.get_total_assignments() == 7


class TestReset:
    """Tests for reset functionality."""
    
    def test_reset_clears_workloads(self, agent_manager: AgentManager) -> None:
        """Test that reset clears agent workloads."""
        # Assign tickets
        for i in range(10):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            agent_manager.assign_ticket(ticket, agent_id=i % 5, current_step=0)
        
        # Verify assignments exist
        assert agent_manager.get_total_assignments() == 10
        
        # Reset
        agent_manager.reset()
        
        # Verify all cleared
        for agent in agent_manager.agents:
            assert agent.current_workload == 0
        
        assert agent_manager.get_total_assignments() == 0
        assert len(agent_manager.completion_schedule) == 0
    
    def test_reset_preserves_expertise(self, agent_manager: AgentManager) -> None:
        """Test that reset preserves agent expertise profiles."""
        # Store original expertise
        original_expertise = [agent.expertise.copy() for agent in agent_manager.agents]
        
        # Assign and reset
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        agent_manager.assign_ticket(ticket, agent_id=0, current_step=0)
        agent_manager.reset()
        
        # Verify expertise unchanged
        for agent, original in zip(agent_manager.agents, original_expertise):
            np.testing.assert_array_equal(agent.expertise, original)


class TestStatistics:
    """Tests for statistics retrieval."""
    
    def test_get_statistics(self, agent_manager: AgentManager) -> None:
        """Test comprehensive statistics retrieval."""
        # Assign some tickets
        for i in range(12):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            agent_manager.assign_ticket(ticket, agent_id=i % 5, current_step=0)
        
        stats = agent_manager.get_statistics()
        
        assert stats['num_agents'] == 5
        assert stats['available_agents'] == 5  # Not at capacity yet
        assert stats['total_assignments'] == 12
        assert stats['workload_mean'] == pytest.approx(2.4)
        assert stats['workload_std'] >= 0
        assert stats['workload_min'] == 2
        assert stats['workload_max'] == 3


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_agent(
        self, default_config: AgentConfig, rng: np.random.Generator
    ) -> None:
        """Test manager with single agent."""
        manager = AgentManager(num_agents=1, config=default_config, rng=rng)
        
        assert manager.num_agents == 1
        assert len(manager.agents) == 1
        assert manager.get_available_agents() == [0]
    
    def test_many_agents(
        self, default_config: AgentConfig, rng: np.random.Generator
    ) -> None:
        """Test manager with many agents."""
        manager = AgentManager(num_agents=100, config=default_config, rng=rng)
        
        assert manager.num_agents == 100
        assert len(manager.agents) == 100
        assert len(manager.get_available_agents()) == 100
    
    def test_custom_capacity(self, rng: np.random.Generator) -> None:
        """Test agents with custom capacity."""
        config = AgentConfig(max_capacity=3, expertise_range=(0.5, 0.7))
        manager = AgentManager(num_agents=2, config=config, rng=rng)
        
        # Fill agent to custom capacity
        for _ in range(3):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            manager.assign_ticket(ticket, agent_id=0, current_step=0)
        
        assert not manager.agents[0].is_available()
        
        # Should fail to assign 4th ticket
        with pytest.raises(ValueError, match="is at capacity"):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.BILLING,
                priority=Priority.LOW,
                arrival_step=0,
            )
            manager.assign_ticket(ticket, agent_id=0, current_step=0)
