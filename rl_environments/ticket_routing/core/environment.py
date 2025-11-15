"""Main Gymnasium environment for support ticket routing."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agent_manager import AgentManager
from .data_models import EnvironmentConfig, Ticket
from .reward_calculator import RewardCalculator
from .state_manager import StateManager
from .ticket_generator import TicketGenerator


class TicketRoutingEnv(gym.Env):
    """Gymnasium environment for intelligent support ticket routing.
    
    This environment simulates a B2B SaaS support system where an RL agent
    learns to optimally assign incoming support tickets to available agents
    based on their expertise, workload, and ticket priority.
    
    Observation Space:
        Dict with:
        - current_ticket: Box(4,) [type, priority, arrival_step_norm, queue_pos_norm]
        - agents: Box(num_agents, 6) [id, workload_norm, capacity_norm, ...expertise(5)]
        - queue_size: Box(1,) [normalized queue size]
        - time_step: Box(1,) [normalized time step]
    
    Action Space:
        Discrete(num_agents) - select which agent to assign the current ticket to
    
    Rewards:
        Multi-component reward based on:
        - Expertise matching between agent and ticket type
        - Expected resolution time
        - Workload balance across agents
        - Ticket priority
    
    Episode Termination:
        - Episode ends after episode_length steps
        - Truncation occurs if queue fills up
    
    Args:
        config: Environment configuration
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the ticket routing environment.
        
        Args:
            config: Environment configuration (uses defaults if None)
            render_mode: Rendering mode ("human" for console output)
        """
        super().__init__()
        
        # Load configuration
        if config is None:
            config = EnvironmentConfig()
        self.config = config
        
        # Set render mode
        self.render_mode = render_mode
        
        # Initialize RNG (will be set properly in reset)
        self._rng = np.random.default_rng()
        
        # Initialize components (will be properly initialized in reset)
        self.ticket_generator: Optional[TicketGenerator] = None
        self.agent_manager: Optional[AgentManager] = None
        self.state_manager: Optional[StateManager] = None
        self.reward_calculator: Optional[RewardCalculator] = None
        
        # Define action space: select which agent to assign ticket to
        self.action_space = spaces.Discrete(config.num_agents)
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Episode tracking
        self.current_step = 0

    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space definition.
        
        Returns:
            Gymnasium Dict space for observations
        """
        num_agents = self.config.num_agents
        
        return spaces.Dict({
            # Current ticket to assign: [type, priority, arrival_step_norm, queue_pos_norm]
            "current_ticket": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([4.0, 3.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            # Agent states: [id_norm, workload_norm, capacity_norm, expertise(5)]
            "agents": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_agents, 8),
                dtype=np.float32,
            ),
            # Queue size (normalized)
            "queue_size": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
            # Current time step (normalized)
            "time_step": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        # Set seed
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Initialize components
        self.ticket_generator = TicketGenerator(
            config=self.config.ticket_config,
            rng=self._rng,
        )
        
        self.agent_manager = AgentManager(
            num_agents=self.config.num_agents,
            config=self.config.agent_config,
            rng=self._rng,
        )
        
        self.state_manager = StateManager(
            max_queue_size=self.config.max_queue_size,
            num_agents=self.config.num_agents,
        )
        
        self.reward_calculator = RewardCalculator(
            config=self.config.reward_config,
        )
        
        # Reset episode tracking
        self.current_step = 0
        
        # Generate initial tickets
        self.ticket_generator.generate_tickets(self.current_step)
        
        # Get first ticket (or None if queue empty)
        current_ticket = self.ticket_generator.get_next_ticket()
        self.state_manager.set_current_ticket(current_ticket)
        
        # Build initial observation
        observation = self._build_observation()
        
        # Build info dict
        info = self._build_info()
        
        return observation, info

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Agent ID to assign current ticket to
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.ticket_generator is not None, "Call reset() before step()"
        assert self.agent_manager is not None, "Call reset() before step()"
        assert self.state_manager is not None, "Call reset() before step()"
        assert self.reward_calculator is not None, "Call reset() before step()"
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get current ticket
        current_ticket = self.state_manager.current_ticket
        
        # Handle case where there's no ticket to assign
        if current_ticket is None:
            # No ticket available, give small negative reward
            reward = -1.0
            self.current_step += 1
            
            # Generate new tickets
            self.ticket_generator.generate_tickets(self.current_step)
            
            # Try to get next ticket
            next_ticket = self.ticket_generator.get_next_ticket()
            self.state_manager.set_current_ticket(next_ticket)
            
            # Update agent workloads
            self.agent_manager.update_workloads(self.current_step)
            
            observation = self._build_observation()
            terminated = self.current_step >= self.config.episode_length
            truncated = False
            info = self._build_info()
            
            return observation, reward, terminated, truncated, info
        
        # Check if selected agent is available
        agent = self.agent_manager.get_agent(action)
        if not agent.is_available():
            # Penalize invalid action (trying to assign to full agent)
            reward = -10.0
            
            # Don't process the ticket, keep it as current
            observation = self._build_observation()
            terminated = False
            truncated = False
            info = self._build_info()
            info['invalid_action'] = True
            
            return observation, reward, terminated, truncated, info
        
        # Assign ticket to agent
        assignment = self.agent_manager.assign_ticket(
            ticket=current_ticket,
            agent_id=action,
            current_step=self.current_step,
        )
        
        # Calculate reward
        agent_workloads = self.agent_manager.get_agent_workloads()
        reward = self.reward_calculator.calculate_reward(
            ticket=current_ticket,
            agent=agent,
            assignment=assignment,
            agent_workloads=agent_workloads,
        )
        
        # Update state
        self.state_manager.update_state(
            ticket=current_ticket,
            agent_id=action,
            assignment=assignment,
        )
        
        # Increment step
        self.current_step += 1
        self.state_manager.increment_step()
        
        # Generate new tickets for this step
        self.ticket_generator.generate_tickets(self.current_step)
        
        # Get next ticket
        next_ticket = self.ticket_generator.get_next_ticket()
        self.state_manager.set_current_ticket(next_ticket)
        
        # Update agent workloads (tickets may have completed)
        self.agent_manager.update_workloads(self.current_step)
        
        # Build next observation
        observation = self._build_observation()
        
        # Check termination conditions
        terminated = self.current_step >= self.config.episode_length
        truncated = self.state_manager.is_queue_full()
        
        # Build info
        info = self._build_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation from current state.
        
        Returns:
            Observation dictionary
        """
        assert self.state_manager is not None
        assert self.agent_manager is not None
        assert self.ticket_generator is not None
        
        current_ticket = self.state_manager.current_ticket
        
        # Build current ticket observation
        if current_ticket is not None:
            ticket_obs = np.array([
                float(current_ticket.type.value),
                float(current_ticket.priority.value),
                float(self.current_step) / self.config.episode_length,
                float(self.state_manager.get_queue_size()) / self.config.max_queue_size,
            ], dtype=np.float32)
        else:
            # No current ticket (queue empty)
            ticket_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Build agents observation
        agents_obs = []
        for agent in self.agent_manager.agents:
            agent_row = np.concatenate([
                np.array([float(agent.id) / self.config.num_agents], dtype=np.float32),
                np.array([float(agent.current_workload) / agent.max_capacity], dtype=np.float32),
                np.array([float(agent.max_capacity) / agent.max_capacity], dtype=np.float32),  # Always 1.0
                agent.expertise.astype(np.float32),
            ])
            agents_obs.append(agent_row)
        
        agents_obs = np.array(agents_obs, dtype=np.float32)
        
        # Build queue size observation
        queue_size_obs = np.array([
            float(self.state_manager.get_queue_size()) / self.config.max_queue_size
        ], dtype=np.float32)
        
        # Build time step observation
        time_step_obs = np.array([
            float(self.current_step) / self.config.episode_length
        ], dtype=np.float32)
        
        observation = {
            "current_ticket": ticket_obs,
            "agents": agents_obs,
            "queue_size": queue_size_obs,
            "time_step": time_step_obs,
        }
        
        return observation

    def _build_info(self) -> Dict[str, Any]:
        """Build info dictionary with episode statistics.
        
        Returns:
            Info dictionary
        """
        assert self.state_manager is not None
        assert self.agent_manager is not None
        assert self.ticket_generator is not None
        
        metrics = self.state_manager.get_metrics()
        agent_stats = self.agent_manager.get_workload_statistics()
        
        info = {
            'step': self.current_step,
            'tickets_processed': metrics.get('tickets_processed', 0),
            'avg_resolution_time': metrics.get('avg_resolution_time', 0.0),
            'queue_size': metrics.get('queue_size', 0),
            'agent_workload_mean': agent_stats['mean'],
            'agent_workload_std': agent_stats['std'],
            'available_agents': len(self.agent_manager.get_available_agents()),
        }
        
        return info

    def render(self) -> None:
        """Render the environment state (console mode)."""
        if self.render_mode != "human":
            return
        
        assert self.state_manager is not None
        assert self.agent_manager is not None
        
        print(f"\n=== Step {self.current_step}/{self.config.episode_length} ===")
        print(f"Queue Size: {self.state_manager.get_queue_size()}/{self.config.max_queue_size}")
        print(f"Tickets Processed: {self.state_manager.tickets_processed}")
        
        current_ticket = self.state_manager.current_ticket
        if current_ticket:
            print(f"Current Ticket: {current_ticket.type.name} (Priority: {current_ticket.priority.name})")
        else:
            print("Current Ticket: None")
        
        print("\nAgent Status:")
        for agent in self.agent_manager.agents:
            status = "AVAILABLE" if agent.is_available() else "AT CAPACITY"
            print(f"  Agent {agent.id}: {agent.current_workload}/{agent.max_capacity} tickets [{status}]")
        
        workload_stats = self.agent_manager.get_workload_statistics()
        print(f"\nWorkload Balance: mean={workload_stats['mean']:.2f}, std={workload_stats['std']:.2f}")

    def close(self) -> None:
        """Clean up resources."""
        pass
