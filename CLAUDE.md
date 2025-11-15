# CLAUDE.md - Development Guide for RL Environments

## Project Information

**Project Name**: RL Environments - Support Ticket Routing  
**Version**: 0.1.0-alpha  
**Python Version**: 3.8+  
**Primary Framework**: Gymnasium (OpenAI Gym successor)  
**Development Methodology**: AI-TDD (Test-Driven Development with AI assistance)

## Technology Stack

### Core Dependencies
- **Gymnasium** (>=0.29.0): RL environment API standard
- **NumPy** (>=1.24.0): Numerical operations and vectorization
- **Pandas** (>=2.0.0): Data logging and analysis
- **PyYAML** (>=6.0): Configuration management

### Visualization & Logging
- **Matplotlib** (>=3.7.0): Plotting and visualization
- **Seaborn** (>=0.12.0): Statistical visualizations
- **TensorBoard**: Training metrics and monitoring (via RL libraries)

### Development Tools
- **pytest** (>=7.3.0): Testing framework
- **hypothesis** (>=6.80.0): Property-based testing
- **mypy** (>=1.4.0): Static type checking
- **black** (>=23.0.0): Code formatting
- **ruff**: Fast Python linter

### Optional RL Libraries
- **stable-baselines3** (>=2.0.0): RL algorithms (PPO, A2C, SAC, etc.)
- **torch** (>=2.0.0): Deep learning backend for SB3
- **rllib**: Alternative RL framework

## Architecture & Design Patterns

### Core Architecture
The project follows a **modular component-based architecture** with clear separation of concerns:

```
TicketRoutingEnv (Gymnasium.Env)
├── StateManager: Episode state and metrics tracking
├── TicketGenerator: Poisson process ticket generation
├── AgentManager: Agent pool and workload management
├── RewardCalculator: Multi-component reward computation
├── ConfigLoader: YAML configuration management
├── MetricsCollector: Observer pattern for metrics
└── Renderer: Visualization and debugging
```

### Design Patterns

#### 1. Strategy Pattern
**Used for**: Reward functions, ticket generation strategies  
**Benefit**: Pluggable components for experimentation

```python
class RewardStrategy(Protocol):
    def calculate(self, ticket: Ticket, agent_id: int, 
                  agent_manager: AgentManager) -> float: ...

class TicketRoutingEnv:
    def __init__(self, reward_strategy: RewardStrategy): ...
```

#### 2. Observer Pattern
**Used for**: Metrics collection  
**Benefit**: Decouple metrics from core logic

```python
class MetricsObserver(Protocol):
    def on_step(self, step_data: dict) -> None: ...
    def on_episode_end(self, episode_data: dict) -> None: ...
```

#### 3. Dependency Injection
**Used for**: Configuration and component wiring  
**Benefit**: Testability and flexibility

```python
class TicketRoutingEnv:
    def __init__(self, 
                 config: EnvironmentConfig,
                 reward_calculator: Optional[RewardCalculator] = None):
        self.reward_calculator = reward_calculator or RewardCalculator(config.reward_config)
```

#### 4. Factory Pattern
**Used for**: Environment creation with presets  
**Benefit**: Easy instantiation with common configurations

```python
class EnvironmentFactory:
    @staticmethod
    def create_small() -> TicketRoutingEnv:
        return TicketRoutingEnv(EnvironmentConfig(num_agents=3, episode_length=500))
```

## Coding Standards

### Type Hints
**Requirement**: 100% type hint coverage on public APIs  
**Tool**: mypy --strict mode

```python
# Good
def generate_tickets(self, current_step: int) -> List[Ticket]:
    """Generate tickets based on current simulation time."""
    ...

# Bad - missing type hints
def generate_tickets(self, current_step):
    ...
```

### Docstrings
**Format**: Google style docstrings  
**Requirement**: All public methods, classes, and modules

```python
def assign_ticket(self, ticket: Ticket, agent_id: int) -> None:
    """Assign a ticket to a support agent.
    
    Args:
        ticket: The ticket to assign
        agent_id: ID of the agent to assign to (0-indexed)
        
    Raises:
        ValueError: If agent_id is out of range or agent at capacity
        
    Example:
        >>> manager = AgentManager(config)
        >>> ticket = Ticket(id=uuid4(), type=TicketType.TECHNICAL, ...)
        >>> manager.assign_ticket(ticket, agent_id=0)
    """
    ...
```

### Code Formatting
**Tool**: black with 100 character line length  
**Configuration**: `black --line-length 100 rl_environments/`

```python
# Good - respects line length
def calculate_reward(
    self, ticket: Ticket, agent_id: int, agent_manager: AgentManager
) -> float:
    ...

# Bad - exceeds line length
def calculate_reward(self, ticket: Ticket, agent_id: int, agent_manager: AgentManager) -> float:
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `TicketRoutingEnv`, `StateManager` |
| Functions/Methods | snake_case | `generate_tickets()`, `get_agent_state()` |
| Private Methods | _snake_case | `_validate_action()`, `_build_observation()` |
| Constants | UPPER_SNAKE_CASE | `MAX_QUEUE_SIZE`, `DEFAULT_CAPACITY` |
| Type Aliases | PascalCase | `ObservationType`, `ActionType` |
| Module Names | snake_case | `ticket_generator.py`, `data_models.py` |

### Error Handling

```python
# Good - specific exceptions with clear messages
def _validate_action(self, action: int) -> bool:
    if not (0 <= action < self.config.num_agents):
        raise ValueError(
            f"Action {action} out of range. "
            f"Expected 0 <= action < {self.config.num_agents}"
        )
    
    agent = self.agent_manager.get_agent(action)
    if not agent.is_available():
        raise ValueError(
            f"Agent {action} at capacity "
            f"({agent.current_workload}/{agent.max_capacity})"
        )
    
    return True

# Bad - generic exceptions, unclear messages
def _validate_action(self, action: int) -> bool:
    if action >= self.config.num_agents:
        raise Exception("Invalid action")
    return True
```

## Testing Standards

### Test Structure
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_data_models.py
│   ├── test_ticket_generator.py
│   ├── test_agent_manager.py
│   ├── test_state_manager.py
│   └── test_reward_calculator.py
├── integration/             # Integration tests
│   ├── test_environment.py
│   └── test_rl_integration.py
├── property/                # Property-based tests
│   └── test_properties.py
└── conftest.py              # Shared fixtures
```

### Test Coverage Requirements
- **Minimum**: 90% overall coverage
- **Critical components**: 95%+ coverage (environment.py, reward_calculator.py)
- **Exemptions**: Only for visualization code

### Unit Testing Example
```python
import pytest
from rl_environments.ticket_routing.core.ticket_generator import TicketGenerator
from rl_environments.ticket_routing.core.data_models import TicketConfig, TicketType

def test_ticket_generator_determinism():
    """Test that seeded generator produces deterministic results."""
    config = TicketConfig(arrival_rate=50)
    
    # Create two generators with same seed
    gen1 = TicketGenerator(config, seed=42)
    gen2 = TicketGenerator(config, seed=42)
    
    tickets1 = gen1.generate_tickets(current_step=10)
    tickets2 = gen2.generate_tickets(current_step=10)
    
    assert len(tickets1) == len(tickets2)
    for t1, t2 in zip(tickets1, tickets2):
        assert t1.type == t2.type
        assert t1.priority == t2.priority

def test_ticket_generator_distribution():
    """Test that ticket types follow configured distribution."""
    config = TicketConfig(
        arrival_rate=1000,
        type_distribution={
            TicketType.TECHNICAL: 0.5,
            TicketType.BILLING: 0.5,
        }
    )
    
    gen = TicketGenerator(config, seed=42)
    tickets = gen.generate_tickets(current_step=1000)
    
    type_counts = {t: 0 for t in TicketType}
    for ticket in tickets:
        type_counts[ticket.type] += 1
    
    # Check distribution is approximately correct (within 10%)
    total = len(tickets)
    assert abs(type_counts[TicketType.TECHNICAL] / total - 0.5) < 0.1
    assert abs(type_counts[TicketType.BILLING] / total - 0.5) < 0.1
```

### Integration Testing Example
```python
def test_full_episode_execution():
    """Test complete episode from reset to termination."""
    config = EnvironmentConfig(num_agents=5, episode_length=100)
    env = TicketRoutingEnv(config)
    
    # Reset
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert "queue_size" in info
    
    # Run episode
    total_reward = 0
    step_count = 0
    
    while step_count < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    assert step_count > 0
    assert "avg_resolution_time" in info
    env.close()
```

### Property-Based Testing Example
```python
from hypothesis import given, strategies as st

@given(
    num_agents=st.integers(min_value=2, max_value=20),
    action=st.integers(min_value=0, max_value=19)
)
def test_action_validation_properties(num_agents, action):
    """Test action validation for any valid configuration."""
    config = EnvironmentConfig(num_agents=num_agents)
    env = TicketRoutingEnv(config)
    env.reset(seed=42)
    
    if 0 <= action < num_agents:
        # Valid action - should not raise
        try:
            env._validate_action(action)
        except ValueError:
            # May raise if agent at capacity - that's valid
            pass
    else:
        # Invalid action - should raise ValueError
        with pytest.raises(ValueError):
            env._validate_action(action)
```

## Performance Optimization

### Profiling
Use `cProfile` to identify bottlenecks:

```python
import cProfile
import pstats

def benchmark_environment():
    env = TicketRoutingEnv()
    obs, info = env.reset(seed=42)
    
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)

# Profile
profiler = cProfile.Profile()
profiler.enable()
benchmark_environment()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Optimization Guidelines

1. **Use NumPy vectorization** for batch operations
```python
# Good - vectorized
workloads = np.array([agent.current_workload for agent in self.agents])
mean_workload = np.mean(workloads)
std_workload = np.std(workloads)

# Bad - loops
total = sum(agent.current_workload for agent in self.agents)
mean_workload = total / len(self.agents)
```

2. **Use `collections.deque`** for O(1) queue operations
```python
from collections import deque

# Good - O(1) append/pop
self.queue = deque()
self.queue.append(ticket)
next_ticket = self.queue.popleft()

# Bad - O(n) operations
self.queue = []
self.queue.append(ticket)
next_ticket = self.queue.pop(0)  # O(n)
```

3. **Lazy evaluation** for expensive metrics
```python
class StateManager:
    @property
    def avg_resolution_time(self) -> float:
        """Calculate only when accessed."""
        if self._cached_avg_resolution_time is None:
            self._cached_avg_resolution_time = self._compute_avg_resolution_time()
        return self._cached_avg_resolution_time
```

## Gymnasium API Compliance

### Required Methods

```python
class TicketRoutingEnv(gymnasium.Env):
    """Gymnasium-compliant RL environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.observation_space = self._build_observation_space()
        self.action_space = gym.spaces.Discrete(config.num_agents)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        # Implementation...
        return observation, info
    
    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one step.
        
        Returns:
            observation: Current observation
            reward: Reward for the action
            terminated: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Implementation...
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render environment state."""
        # Implementation...
    
    def close(self) -> None:
        """Cleanup resources."""
        # Implementation...
```

### Validation
Always validate with Gymnasium's checker:

```python
from gymnasium.utils.env_checker import check_env

env = TicketRoutingEnv()
check_env(env, warn=True)  # Raises errors for non-compliance
```

## Configuration Management

### YAML Configuration
```yaml
# config/default_config.yaml
environment:
  num_agents: 5
  episode_length: 1000
  max_queue_size: 50

ticket:
  arrival_rate: 50  # tickets per day
  type_distribution:
    technical: 0.30
    billing: 0.25
    feature: 0.20
    bug: 0.15
    integration: 0.10
  priority_distribution:
    critical: 0.05
    high: 0.15
    medium: 0.50
    low: 0.30

reward:
  base_reward: 10.0
  expertise_multiplier: 5.0
  resolution_penalty_rate: 0.1
  workload_threshold: 2.0
  workload_penalty: 5.0

agent:
  max_capacity: 10
  expertise_range: [0.3, 0.9]  # min, max expertise values
```

### Configuration Loading
```python
from dataclasses import dataclass
import yaml

@dataclass
class EnvironmentConfig:
    num_agents: int = 5
    episode_length: int = 1000
    # ... other fields
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EnvironmentConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate and construct
        return cls(
            num_agents=data['environment']['num_agents'],
            episode_length=data['environment']['episode_length'],
            # ...
        )
```

## Development Workflow

### 1. Start New Feature
```bash
git checkout -b feature/my-feature
```

### 2. Write Tests First (TDD)
```bash
# Create test file
touch rl_environments/ticket_routing/tests/test_my_feature.py

# Write failing tests
pytest rl_environments/ticket_routing/tests/test_my_feature.py
```

### 3. Implement Feature
```bash
# Implement until tests pass
pytest rl_environments/ticket_routing/tests/test_my_feature.py
```

### 4. Quality Checks
```bash
# Type checking
mypy rl_environments --strict

# Formatting
black rl_environments

# Linting
ruff check rl_environments

# Full test suite
pytest --cov=rl_environments --cov-report=html
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

## Common Commands

### Development
```bash
# Install in editable mode
pip install -e .

# Run tests
pytest

# Run with coverage
pytest --cov=rl_environments --cov-report=html

# Type check
mypy rl_environments --strict

# Format code
black rl_environments

# Lint
ruff check rl_environments --fix
```

### Environment Testing
```bash
# Validate Gymnasium compliance
python -c "from gymnasium.utils.env_checker import check_env; from rl_environments.ticket_routing import TicketRoutingEnv; check_env(TicketRoutingEnv())"

# Run random policy
python examples/random_policy.py

# Train with SB3
python examples/train_ppo.py
```

## Troubleshooting

### Common Issues

**Issue**: Observation space mismatch  
**Solution**: Validate observation against space: `assert env.observation_space.contains(obs)`

**Issue**: Non-deterministic behavior  
**Solution**: Ensure seeding: `env.reset(seed=42)` and use `np.random.Generator`

**Issue**: Slow step time  
**Solution**: Profile with `cProfile`, optimize hot paths with NumPy vectorization

**Issue**: Type errors with mypy  
**Solution**: Add explicit type hints, avoid `Any` types

## Resources

- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **NumPy Best Practices**: https://numpy.org/doc/stable/user/basics.performance.html
- **Python Type Hints**: https://mypy.readthedocs.io/

---

**Last Updated**: 2025-11-14  
**Maintained By**: Development Team
