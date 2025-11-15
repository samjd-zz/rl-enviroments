"""Data models for support ticket routing environment."""

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Deque
from collections import deque

import numpy as np


class TicketType(IntEnum):
    """Types of support tickets."""

    TECHNICAL = 0
    BILLING = 1
    FEATURE = 2
    BUG = 3
    INTEGRATION = 4


class Priority(IntEnum):
    """Priority levels for tickets."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(frozen=True)
class Ticket:
    """Represents a support ticket.
    
    Args:
        id: Unique identifier for the ticket
        type: Type of support issue
        priority: Priority level
        arrival_step: Simulation step when ticket arrived
    """

    id: uuid.UUID
    type: TicketType
    priority: Priority
    arrival_step: int

    def __post_init__(self) -> None:
        """Validate ticket data."""
        if self.arrival_step < 0:
            raise ValueError(f"arrival_step must be non-negative, got {self.arrival_step}")


@dataclass
class SupportAgent:
    """Represents a support agent.
    
    Args:
        id: Unique identifier for the agent (0-indexed)
        expertise: Array of expertise scores for each ticket type (0.0-1.0)
        current_workload: Number of currently assigned tickets
        max_capacity: Maximum number of concurrent tickets
    """

    id: int
    expertise: np.ndarray
    current_workload: int = 0
    max_capacity: int = 10

    def __post_init__(self) -> None:
        """Validate agent data."""
        if self.id < 0:
            raise ValueError(f"Agent id must be non-negative, got {self.id}")
        
        if not isinstance(self.expertise, np.ndarray):
            raise TypeError(f"expertise must be numpy array, got {type(self.expertise)}")
        
        if self.expertise.shape != (len(TicketType),):
            raise ValueError(
                f"expertise must have shape ({len(TicketType)},), "
                f"got {self.expertise.shape}"
            )
        
        if not np.all((self.expertise >= 0.0) & (self.expertise <= 1.0)):
            raise ValueError("All expertise values must be in range [0.0, 1.0]")
        
        if self.current_workload < 0:
            raise ValueError(f"current_workload must be non-negative, got {self.current_workload}")
        
        if self.max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {self.max_capacity}")

    def is_available(self) -> bool:
        """Check if agent can accept more tickets.
        
        Returns:
            True if agent is below capacity, False otherwise
        """
        return self.current_workload < self.max_capacity

    def get_resolution_time(self, ticket_type: TicketType) -> float:
        """Calculate expected resolution time for a ticket type.
        
        Args:
            ticket_type: Type of ticket to resolve
            
        Returns:
            Expected resolution time in hours
        """
        base_time = 24.0  # hours
        expertise_factor = self.expertise[ticket_type.value]
        # Higher expertise (closer to 1.0) -> faster resolution (closer to 1x base time)
        # Lower expertise (closer to 0.0) -> slower resolution (up to 2x base time)
        time_multiplier = 2.0 - expertise_factor
        return base_time * time_multiplier


@dataclass(frozen=True)
class Assignment:
    """Represents a ticket assignment to an agent.
    
    Args:
        id: Unique identifier for the assignment
        ticket_id: ID of the assigned ticket
        agent_id: ID of the assigned agent
        assignment_step: Simulation step when assignment occurred
        estimated_resolution_time: Expected time to resolve (hours)
    """

    id: uuid.UUID
    ticket_id: uuid.UUID
    agent_id: int
    assignment_step: int
    estimated_resolution_time: float


@dataclass
class EnvironmentState:
    """Complete state of the environment.
    
    Args:
        step: Current simulation step
        tickets_processed: Total number of tickets processed
        total_resolution_time: Cumulative resolution time
        agent_workloads: Array of current workloads per agent
        current_ticket: Ticket currently being assigned (if any)
        queue: Deque of pending tickets
        active_assignments: Map of agent_id to their active assignments
    """

    step: int
    tickets_processed: int
    total_resolution_time: float
    agent_workloads: np.ndarray
    current_ticket: Optional[Ticket]
    queue: Deque[Ticket]
    active_assignments: Dict[int, List[Assignment]]


# Configuration Classes

@dataclass
class TicketConfig:
    """Configuration for ticket generation.
    
    Args:
        arrival_rate: Average tickets per day
        type_distribution: Probability distribution over ticket types
        priority_distribution: Probability distribution over priorities
    """

    arrival_rate: float = 50.0
    type_distribution: Dict[TicketType, float] = field(default_factory=lambda: {
        TicketType.TECHNICAL: 0.30,
        TicketType.BILLING: 0.25,
        TicketType.FEATURE: 0.20,
        TicketType.BUG: 0.15,
        TicketType.INTEGRATION: 0.10,
    })
    priority_distribution: Dict[Priority, float] = field(default_factory=lambda: {
        Priority.CRITICAL: 0.05,
        Priority.HIGH: 0.15,
        Priority.MEDIUM: 0.50,
        Priority.LOW: 0.30,
    })

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be positive, got {self.arrival_rate}")
        
        # Validate type distribution
        if not np.isclose(sum(self.type_distribution.values()), 1.0):
            raise ValueError(
                f"type_distribution must sum to 1.0, got {sum(self.type_distribution.values())}"
            )
        
        # Validate priority distribution
        if not np.isclose(sum(self.priority_distribution.values()), 1.0):
            raise ValueError(
                f"priority_distribution must sum to 1.0, "
                f"got {sum(self.priority_distribution.values())}"
            )


@dataclass
class AgentConfig:
    """Configuration for support agents.
    
    Args:
        max_capacity: Maximum concurrent tickets per agent
        expertise_range: (min, max) expertise values for random initialization
    """

    max_capacity: int = 10
    expertise_range: tuple[float, float] = (0.3, 0.9)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_capacity <= 0:
            raise ValueError(f"max_capacity must be positive, got {self.max_capacity}")
        
        min_exp, max_exp = self.expertise_range
        if not (0.0 <= min_exp <= max_exp <= 1.0):
            raise ValueError(
                f"expertise_range must satisfy 0.0 <= min <= max <= 1.0, "
                f"got {self.expertise_range}"
            )


@dataclass
class RewardConfig:
    """Configuration for reward calculation.
    
    Args:
        base_reward: Base reward for valid assignment
        expertise_multiplier: Multiplier for expertise bonus
        resolution_penalty_rate: Rate for resolution time penalty
        workload_threshold: Standard deviation threshold for balance penalty
        workload_penalty: Penalty for imbalanced workload
    """

    base_reward: float = 10.0
    expertise_multiplier: float = 5.0
    resolution_penalty_rate: float = 0.1
    workload_threshold: float = 2.0
    workload_penalty: float = 5.0


@dataclass
class EnvironmentConfig:
    """Complete environment configuration.
    
    Args:
        num_agents: Number of support agents
        episode_length: Maximum steps per episode
        max_queue_size: Maximum size of ticket queue
        ticket_config: Ticket generation configuration
        agent_config: Agent configuration
        reward_config: Reward calculation configuration
    """

    num_agents: int = 5
    episode_length: int = 1000
    max_queue_size: int = 50
    ticket_config: TicketConfig = field(default_factory=TicketConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_agents < 2:
            raise ValueError(f"num_agents must be at least 2, got {self.num_agents}")
        
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {self.episode_length}")
        
        if self.max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {self.max_queue_size}")

    @classmethod
    def from_yaml(cls, path: str) -> "EnvironmentConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Loaded configuration
        """
        import yaml
        from pathlib import Path
        
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse ticket config
        ticket_data = data.get('ticket', {})
        ticket_config = TicketConfig(
            arrival_rate=ticket_data.get('arrival_rate', 50.0),
            type_distribution={
                TicketType[k.upper()]: v
                for k, v in ticket_data.get('type_distribution', {}).items()
            } if 'type_distribution' in ticket_data else TicketConfig().type_distribution,
            priority_distribution={
                Priority[k.upper()]: v
                for k, v in ticket_data.get('priority_distribution', {}).items()
            } if 'priority_distribution' in ticket_data else TicketConfig().priority_distribution,
        )
        
        # Parse agent config
        agent_data = data.get('agent', {})
        agent_config = AgentConfig(
            max_capacity=agent_data.get('max_capacity', 10),
            expertise_range=tuple(agent_data.get('expertise_range', [0.3, 0.9])),
        )
        
        # Parse reward config
        reward_data = data.get('reward', {})
        reward_config = RewardConfig(
            base_reward=reward_data.get('base_reward', 10.0),
            expertise_multiplier=reward_data.get('expertise_multiplier', 5.0),
            resolution_penalty_rate=reward_data.get('resolution_penalty_rate', 0.1),
            workload_threshold=reward_data.get('workload_threshold', 2.0),
            workload_penalty=reward_data.get('workload_penalty', 5.0),
        )
        
        # Parse environment config
        env_data = data.get('environment', {})
        return cls(
            num_agents=env_data.get('num_agents', 5),
            episode_length=env_data.get('episode_length', 1000),
            max_queue_size=env_data.get('max_queue_size', 50),
            ticket_config=ticket_config,
            agent_config=agent_config,
            reward_config=reward_config,
        )
