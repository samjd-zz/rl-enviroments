"""Tests for data models."""

import uuid
import pytest
import numpy as np
from collections import deque

from rl_environments.ticket_routing.core.data_models import (
    TicketType,
    Priority,
    Ticket,
    SupportAgent,
    Assignment,
    EnvironmentState,
    TicketConfig,
    AgentConfig,
    RewardConfig,
    EnvironmentConfig,
)


class TestTicketType:
    """Tests for TicketType enum."""

    def test_ticket_type_values(self):
        """Test that ticket types have expected values."""
        assert TicketType.TECHNICAL == 0
        assert TicketType.BILLING == 1
        assert TicketType.FEATURE == 2
        assert TicketType.BUG == 3
        assert TicketType.INTEGRATION == 4

    def test_ticket_type_count(self):
        """Test number of ticket types."""
        assert len(TicketType) == 5


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_values(self):
        """Test that priorities have expected values."""
        assert Priority.CRITICAL == 0
        assert Priority.HIGH == 1
        assert Priority.MEDIUM == 2
        assert Priority.LOW == 3

    def test_priority_count(self):
        """Test number of priority levels."""
        assert len(Priority) == 4


class TestTicket:
    """Tests for Ticket dataclass."""

    def test_ticket_creation(self):
        """Test creating a valid ticket."""
        ticket_id = uuid.uuid4()
        ticket = Ticket(
            id=ticket_id,
            type=TicketType.TECHNICAL,
            priority=Priority.HIGH,
            arrival_step=10
        )
        
        assert ticket.id == ticket_id
        assert ticket.type == TicketType.TECHNICAL
        assert ticket.priority == Priority.HIGH
        assert ticket.arrival_step == 10

    def test_ticket_immutable(self):
        """Test that tickets are immutable."""
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.MEDIUM,
            arrival_step=5
        )
        
        with pytest.raises(AttributeError):
            ticket.type = TicketType.TECHNICAL  # type: ignore

    def test_ticket_negative_arrival_step(self):
        """Test that negative arrival_step raises error."""
        with pytest.raises(ValueError, match="arrival_step must be non-negative"):
            Ticket(
                id=uuid.uuid4(),
                type=TicketType.BUG,
                priority=Priority.LOW,
                arrival_step=-1
            )


class TestSupportAgent:
    """Tests for SupportAgent dataclass."""

    def test_agent_creation(self):
        """Test creating a valid agent."""
        expertise = np.array([0.8, 0.6, 0.4, 0.7, 0.5])
        agent = SupportAgent(
            id=0,
            expertise=expertise,
            current_workload=3,
            max_capacity=10
        )
        
        assert agent.id == 0
        assert np.array_equal(agent.expertise, expertise)
        assert agent.current_workload == 3
        assert agent.max_capacity == 10

    def test_agent_default_workload(self):
        """Test agent defaults to zero workload."""
        agent = SupportAgent(
            id=1,
            expertise=np.array([0.5] * 5)
        )
        
        assert agent.current_workload == 0
        assert agent.max_capacity == 10

    def test_agent_is_available(self):
        """Test availability checking."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.5] * 5),
            current_workload=5,
            max_capacity=10
        )
        
        assert agent.is_available() is True
        
        agent.current_workload = 10
        assert agent.is_available() is False
        
        agent.current_workload = 11
        assert agent.is_available() is False

    def test_agent_resolution_time(self):
        """Test resolution time calculation."""
        # Agent with high expertise (0.9) for technical tickets
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.9, 0.5, 0.3, 0.6, 0.4])
        )
        
        # High expertise -> faster (closer to 24 hours)
        time_tech = agent.get_resolution_time(TicketType.TECHNICAL)
        assert 24.0 <= time_tech <= 28.0  # 24 * (2.0 - 0.9) = 26.4
        
        # Low expertise -> slower (closer to 48 hours)
        time_feature = agent.get_resolution_time(TicketType.FEATURE)
        assert 40.0 <= time_feature <= 48.0  # 24 * (2.0 - 0.3) = 40.8

    def test_agent_invalid_id(self):
        """Test that negative ID raises error."""
        with pytest.raises(ValueError, match="Agent id must be non-negative"):
            SupportAgent(
                id=-1,
                expertise=np.array([0.5] * 5)
            )

    def test_agent_invalid_expertise_type(self):
        """Test that non-array expertise raises error."""
        with pytest.raises(TypeError, match="expertise must be numpy array"):
            SupportAgent(
                id=0,
                expertise=[0.5] * 5  # type: ignore
            )

    def test_agent_invalid_expertise_shape(self):
        """Test that wrong shape expertise raises error."""
        with pytest.raises(ValueError, match="expertise must have shape"):
            SupportAgent(
                id=0,
                expertise=np.array([0.5, 0.6, 0.7])  # Wrong length
            )

    def test_agent_expertise_out_of_range(self):
        """Test that expertise values outside [0,1] raise error."""
        with pytest.raises(ValueError, match="expertise values must be in range"):
            SupportAgent(
                id=0,
                expertise=np.array([0.5, 1.5, 0.3, 0.6, 0.4])  # 1.5 > 1.0
            )


class TestTicketConfig:
    """Tests for TicketConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TicketConfig()
        
        assert config.arrival_rate == 50.0
        assert len(config.type_distribution) == 5
        assert len(config.priority_distribution) == 4
        assert np.isclose(sum(config.type_distribution.values()), 1.0)
        assert np.isclose(sum(config.priority_distribution.values()), 1.0)

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = TicketConfig(
            arrival_rate=100.0,
            type_distribution={
                TicketType.TECHNICAL: 0.5,
                TicketType.BILLING: 0.5,
                TicketType.FEATURE: 0.0,
                TicketType.BUG: 0.0,
                TicketType.INTEGRATION: 0.0,
            },
            priority_distribution={
                Priority.CRITICAL: 0.1,
                Priority.HIGH: 0.2,
                Priority.MEDIUM: 0.3,
                Priority.LOW: 0.4,
            }
        )
        
        assert config.arrival_rate == 100.0
        assert config.type_distribution[TicketType.TECHNICAL] == 0.5

    def test_invalid_arrival_rate(self):
        """Test that invalid arrival rate raises error."""
        with pytest.raises(ValueError, match="arrival_rate must be positive"):
            TicketConfig(arrival_rate=0.0)

    def test_invalid_type_distribution(self):
        """Test that invalid type distribution raises error."""
        with pytest.raises(ValueError, match="type_distribution must sum to 1.0"):
            TicketConfig(
                type_distribution={
                    TicketType.TECHNICAL: 0.5,
                    TicketType.BILLING: 0.3,  # Sum = 0.8 != 1.0
                    TicketType.FEATURE: 0.0,
                    TicketType.BUG: 0.0,
                    TicketType.INTEGRATION: 0.0,
                }
            )


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        
        assert config.max_capacity == 10
        assert config.expertise_range == (0.3, 0.9)

    def test_invalid_max_capacity(self):
        """Test that invalid max_capacity raises error."""
        with pytest.raises(ValueError, match="max_capacity must be positive"):
            AgentConfig(max_capacity=0)

    def test_invalid_expertise_range(self):
        """Test that invalid expertise range raises error."""
        with pytest.raises(ValueError, match="expertise_range must satisfy"):
            AgentConfig(expertise_range=(0.9, 0.3))  # min > max


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvironmentConfig()
        
        assert config.num_agents == 5
        assert config.episode_length == 1000
        assert config.max_queue_size == 50
        assert isinstance(config.ticket_config, TicketConfig)
        assert isinstance(config.agent_config, AgentConfig)
        assert isinstance(config.reward_config, RewardConfig)

    def test_invalid_num_agents(self):
        """Test that invalid num_agents raises error."""
        with pytest.raises(ValueError, match="num_agents must be at least 2"):
            EnvironmentConfig(num_agents=1)

    def test_invalid_episode_length(self):
        """Test that invalid episode_length raises error."""
        with pytest.raises(ValueError, match="episode_length must be positive"):
            EnvironmentConfig(episode_length=0)
