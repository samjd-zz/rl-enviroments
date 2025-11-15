"""Tests for StateManager component."""

import uuid
from collections import deque

import numpy as np
import pytest

from rl_environments.ticket_routing.core.data_models import (
    Assignment,
    Ticket,
    TicketType,
    Priority,
)
from rl_environments.ticket_routing.core.state_manager import StateManager


@pytest.fixture
def state_manager() -> StateManager:
    """Provide initialized state manager."""
    return StateManager(max_queue_size=50, num_agents=5)


@pytest.fixture
def sample_ticket() -> Ticket:
    """Provide sample ticket for testing."""
    return Ticket(
        id=uuid.uuid4(),
        type=TicketType.TECHNICAL,
        priority=Priority.HIGH,
        arrival_step=0,
    )


@pytest.fixture
def sample_assignment(sample_ticket: Ticket) -> Assignment:
    """Provide sample assignment for testing."""
    return Assignment(
        id=uuid.uuid4(),
        ticket_id=sample_ticket.id,
        agent_id=0,
        assignment_step=100,
        estimated_resolution_time=24.0,
    )


class TestStateManagerInitialization:
    """Tests for StateManager initialization."""
    
    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        manager = StateManager(max_queue_size=50, num_agents=5)
        
        assert manager.max_queue_size == 50
        assert manager.num_agents == 5
        assert manager.step == 0
        assert manager.tickets_processed == 0
        assert manager.total_resolution_time == 0.0
        assert manager.current_ticket is None
        assert len(manager.queue) == 0
        assert len(manager.assignment_history) == 0
    
    def test_initialization_invalid_queue_size(self) -> None:
        """Test initialization with invalid queue size."""
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StateManager(max_queue_size=0, num_agents=5)
        
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StateManager(max_queue_size=-1, num_agents=5)
    
    def test_initialization_invalid_num_agents(self) -> None:
        """Test initialization with invalid number of agents."""
        with pytest.raises(ValueError, match="num_agents must be at least 1"):
            StateManager(max_queue_size=50, num_agents=0)


class TestStateUpdates:
    """Tests for state update operations."""
    
    def test_update_state(
        self,
        state_manager: StateManager,
        sample_ticket: Ticket,
        sample_assignment: Assignment,
    ) -> None:
        """Test updating state after assignment."""
        # Set current ticket first
        state_manager.set_current_ticket(sample_ticket)
        assert state_manager.current_ticket is not None
        
        # Update state
        state_manager.update_state(
            ticket=sample_ticket,
            agent_id=0,
            assignment=sample_assignment,
        )
        
        # Verify updates
        assert state_manager.tickets_processed == 1
        assert state_manager.total_resolution_time == 24.0
        assert len(state_manager.assignment_history) == 1
        assert state_manager.assignment_history[0] == sample_assignment
        assert state_manager.current_ticket is None  # Cleared after assignment
    
    def test_update_state_tracks_ticket_types(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test that state update tracks ticket type distribution."""
        tickets = [
            (TicketType.TECHNICAL, Priority.HIGH),
            (TicketType.TECHNICAL, Priority.LOW),
            (TicketType.BILLING, Priority.MEDIUM),
            (TicketType.FEATURE, Priority.HIGH),
        ]
        
        for ticket_type, priority in tickets:
            ticket = Ticket(
                id=uuid.uuid4(),
                type=ticket_type,
                priority=priority,
                arrival_step=0,
            )
            assignment = Assignment(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                agent_id=0,
                assignment_step=0,
                estimated_resolution_time=24.0,
            )
            state_manager.update_state(ticket, 0, assignment)
        
        # Verify type tracking
        assert state_manager.episode_tickets_by_type[TicketType.TECHNICAL.value] == 2
        assert state_manager.episode_tickets_by_type[TicketType.BILLING.value] == 1
        assert state_manager.episode_tickets_by_type[TicketType.FEATURE.value] == 1
    
    def test_update_state_tracks_priorities(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test that state update tracks priority distribution."""
        tickets = [
            (TicketType.TECHNICAL, Priority.HIGH),
            (TicketType.BILLING, Priority.HIGH),
            (TicketType.FEATURE, Priority.MEDIUM),
            (TicketType.BUG, Priority.LOW),
        ]
        
        for ticket_type, priority in tickets:
            ticket = Ticket(
                id=uuid.uuid4(),
                type=ticket_type,
                priority=priority,
                arrival_step=0,
            )
            assignment = Assignment(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                agent_id=0,
                assignment_step=0,
                estimated_resolution_time=24.0,
            )
            state_manager.update_state(ticket, 0, assignment)
        
        # Verify priority tracking
        assert state_manager.episode_tickets_by_priority[Priority.HIGH.value] == 2
        assert state_manager.episode_tickets_by_priority[Priority.MEDIUM.value] == 1
        assert state_manager.episode_tickets_by_priority[Priority.LOW.value] == 1
    
    def test_set_current_ticket(
        self,
        state_manager: StateManager,
        sample_ticket: Ticket,
    ) -> None:
        """Test setting current ticket."""
        assert state_manager.current_ticket is None
        
        state_manager.set_current_ticket(sample_ticket)
        assert state_manager.current_ticket == sample_ticket
        
        state_manager.set_current_ticket(None)
        assert state_manager.current_ticket is None
    
    def test_increment_step(self, state_manager: StateManager) -> None:
        """Test step counter increment."""
        assert state_manager.step == 0
        
        state_manager.increment_step()
        assert state_manager.step == 1
        
        state_manager.increment_step()
        assert state_manager.step == 2


class TestQueueOperations:
    """Tests for queue management operations."""
    
    def test_add_to_queue(
        self,
        state_manager: StateManager,
        sample_ticket: Ticket,
    ) -> None:
        """Test adding ticket to queue."""
        assert state_manager.get_queue_size() == 0
        
        success = state_manager.add_to_queue(sample_ticket)
        assert success
        assert state_manager.get_queue_size() == 1
    
    def test_add_to_full_queue(self, state_manager: StateManager) -> None:
        """Test adding to queue at capacity."""
        # Fill queue to capacity
        for i in range(50):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            success = state_manager.add_to_queue(ticket)
            assert success
        
        # Try to add one more
        extra_ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.LOW,
            arrival_step=50,
        )
        success = state_manager.add_to_queue(extra_ticket)
        assert not success
        assert state_manager.get_queue_size() == 50
    
    def test_pop_from_queue(self, state_manager: StateManager) -> None:
        """Test removing ticket from queue."""
        tickets = [
            Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.HIGH,
                arrival_step=i,
            )
            for i in range(3)
        ]
        
        for ticket in tickets:
            state_manager.add_to_queue(ticket)
        
        # Pop tickets (FIFO order)
        popped1 = state_manager.pop_from_queue()
        assert popped1 == tickets[0]
        assert state_manager.get_queue_size() == 2
        
        popped2 = state_manager.pop_from_queue()
        assert popped2 == tickets[1]
        assert state_manager.get_queue_size() == 1
    
    def test_pop_from_empty_queue(self, state_manager: StateManager) -> None:
        """Test popping from empty queue."""
        assert state_manager.get_queue_size() == 0
        
        popped = state_manager.pop_from_queue()
        assert popped is None
    
    def test_peek_queue(self, state_manager: StateManager) -> None:
        """Test viewing queue without removing."""
        ticket1 = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.HIGH,
            arrival_step=0,
        )
        ticket2 = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.MEDIUM,
            arrival_step=1,
        )
        
        state_manager.add_to_queue(ticket1)
        state_manager.add_to_queue(ticket2)
        
        # Peek should return first ticket without removing
        peeked = state_manager.peek_queue()
        assert peeked == ticket1
        assert state_manager.get_queue_size() == 2  # Size unchanged
        
        # Peek again should return same ticket
        peeked_again = state_manager.peek_queue()
        assert peeked_again == ticket1
    
    def test_peek_empty_queue(self, state_manager: StateManager) -> None:
        """Test peeking at empty queue."""
        peeked = state_manager.peek_queue()
        assert peeked is None
    
    def test_is_queue_full(self, state_manager: StateManager) -> None:
        """Test checking if queue is full."""
        assert not state_manager.is_queue_full()
        
        # Fill to capacity
        for i in range(50):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            state_manager.add_to_queue(ticket)
        
        assert state_manager.is_queue_full()


class TestStateRetrieval:
    """Tests for state retrieval operations."""
    
    def test_get_state(
        self,
        state_manager: StateManager,
        sample_ticket: Ticket,
    ) -> None:
        """Test getting complete environment state."""
        state_manager.set_current_ticket(sample_ticket)
        state_manager.increment_step()
        
        agent_workloads = np.array([2, 3, 1, 4, 0], dtype=np.int32)
        
        env_state = state_manager.get_state(agent_workloads)
        
        assert env_state.step == 1
        assert env_state.tickets_processed == 0
        assert env_state.current_ticket == sample_ticket
        assert len(env_state.queue) == 0
        np.testing.assert_array_equal(env_state.agent_workloads, agent_workloads)
    
    def test_get_state_creates_copy(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test that get_state returns copies not references."""
        agent_workloads = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        
        env_state = state_manager.get_state(agent_workloads)
        
        # Modify returned arrays
        env_state.agent_workloads[0] = 999
        
        # Original should be unchanged
        assert agent_workloads[0] == 1


class TestMetrics:
    """Tests for metrics calculation."""
    
    def test_get_metrics_initial(self, state_manager: StateManager) -> None:
        """Test getting metrics before any processing."""
        metrics = state_manager.get_metrics()
        
        assert metrics['step'] == 0.0
        assert metrics['tickets_processed'] == 0.0
        assert metrics['total_resolution_time'] == 0.0
        assert metrics['queue_size'] == 0.0
        assert metrics['avg_resolution_time'] == 0.0
    
    def test_get_metrics_after_processing(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test metrics after processing tickets."""
        # Process some tickets
        for i in range(5):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            assignment = Assignment(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                agent_id=0,
                assignment_step=i,
                estimated_resolution_time=20.0,
            )
            state_manager.update_state(ticket, 0, assignment)
        
        state_manager.increment_step()
        state_manager.increment_step()
        
        metrics = state_manager.get_metrics()
        
        assert metrics['step'] == 2.0
        assert metrics['tickets_processed'] == 5.0
        assert metrics['total_resolution_time'] == 100.0
        assert metrics['avg_resolution_time'] == 20.0
    
    def test_get_metrics_with_type_distribution(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test that metrics include ticket type distribution."""
        tickets = [
            TicketType.TECHNICAL,
            TicketType.TECHNICAL,
            TicketType.BILLING,
        ]
        
        for ticket_type in tickets:
            ticket = Ticket(
                id=uuid.uuid4(),
                type=ticket_type,
                priority=Priority.MEDIUM,
                arrival_step=0,
            )
            assignment = Assignment(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                agent_id=0,
                assignment_step=0,
                estimated_resolution_time=24.0,
            )
            state_manager.update_state(ticket, 0, assignment)
        
        metrics = state_manager.get_metrics()
        
        assert metrics[f'tickets_type_{TicketType.TECHNICAL.value}'] == 2.0
        assert metrics[f'tickets_type_{TicketType.BILLING.value}'] == 1.0
    
    def test_get_episode_statistics(
        self,
        state_manager: StateManager,
    ) -> None:
        """Test comprehensive episode statistics."""
        # Add some tickets to queue
        for i in range(3):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            state_manager.add_to_queue(ticket)
        
        # Process some tickets
        for i in range(5):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.BILLING,
                priority=Priority.HIGH,
                arrival_step=i,
            )
            assignment = Assignment(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                agent_id=0,
                assignment_step=i,
                estimated_resolution_time=24.0,
            )
            state_manager.update_state(ticket, 0, assignment)
        
        stats = state_manager.get_episode_statistics()
        
        assert stats['step'] == 0
        assert stats['tickets_processed'] == 5
        assert stats['total_assignments'] == 5
        assert stats['queue_size'] == 3
        assert stats['avg_resolution_time'] == 24.0
        assert stats['tickets_by_type'][TicketType.BILLING.value] == 5
        assert stats['tickets_by_priority'][Priority.HIGH.value] == 5


class TestReset:
    """Tests for reset functionality."""
    
    def test_reset_clears_all_state(
        self,
        state_manager: StateManager,
        sample_ticket: Ticket,
        sample_assignment: Assignment,
    ) -> None:
        """Test that reset clears all state and metrics."""
        # Build up some state
        state_manager.set_current_ticket(sample_ticket)
        state_manager.increment_step()
        state_manager.increment_step()
        
        for i in range(3):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            state_manager.add_to_queue(ticket)
        
        state_manager.update_state(sample_ticket, 0, sample_assignment)
        
        # Verify state is not empty
        assert state_manager.step == 2
        assert state_manager.tickets_processed == 1
        assert state_manager.get_queue_size() == 3
        assert len(state_manager.assignment_history) == 1
        
        # Reset
        state_manager.reset()
        
        # Verify all state cleared
        assert state_manager.step == 0
        assert state_manager.tickets_processed == 0
        assert state_manager.total_resolution_time == 0.0
        assert state_manager.current_ticket is None
        assert state_manager.get_queue_size() == 0
        assert len(state_manager.assignment_history) == 0
        assert len(state_manager.episode_tickets_by_type) == 0
        assert len(state_manager.episode_tickets_by_priority) == 0
    
    def test_reset_allows_reuse(self, state_manager: StateManager) -> None:
        """Test that manager can be reused after reset."""
        # Use once
        ticket1 = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.HIGH,
            arrival_step=0,
        )
        state_manager.add_to_queue(ticket1)
        state_manager.increment_step()
        
        # Reset
        state_manager.reset()
        
        # Use again
        ticket2 = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        state_manager.add_to_queue(ticket2)
        state_manager.increment_step()
        
        # Verify it works
        assert state_manager.step == 1
        assert state_manager.get_queue_size() == 1
        peeked = state_manager.peek_queue()
        assert peeked == ticket2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_small_queue_size(self) -> None:
        """Test manager with small queue size."""
        manager = StateManager(max_queue_size=2, num_agents=3)
        
        assert manager.max_queue_size == 2
        
        # Fill queue
        for i in range(2):
            ticket = Ticket(
                id=uuid.uuid4(),
                type=TicketType.TECHNICAL,
                priority=Priority.MEDIUM,
                arrival_step=i,
            )
            manager.add_to_queue(ticket)
        
        assert manager.is_queue_full()
    
    def test_large_queue_size(self) -> None:
        """Test manager with large queue size."""
        manager = StateManager(max_queue_size=1000, num_agents=10)
        
        assert manager.max_queue_size == 1000
        assert not manager.is_queue_full()
    
    def test_single_agent(self) -> None:
        """Test manager with single agent."""
        manager = StateManager(max_queue_size=50, num_agents=1)
        
        assert manager.num_agents == 1
        
        agent_workloads = np.array([5], dtype=np.int32)
        state = manager.get_state(agent_workloads)
        
        assert len(state.agent_workloads) == 1
        assert len(state.active_assignments) == 1
