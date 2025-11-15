"""Agent management for support ticket routing environment."""

import uuid
from typing import Dict, List, Optional

import numpy as np

from .data_models import (
    Assignment,
    SupportAgent,
    Ticket,
    TicketType,
    AgentConfig,
)


class AgentManager:
    """Manages support agents and their assignments.
    
    Tracks agent workloads, handles ticket assignments, and calculates
    resolution times based on agent expertise.
    
    Args:
        num_agents: Number of support agents in the pool
        config: Agent configuration parameters
        rng: NumPy random number generator for reproducibility
    """

    def __init__(
        self,
        num_agents: int,
        config: AgentConfig,
        rng: np.random.Generator,
    ) -> None:
        """Initialize agent manager.
        
        Args:
            num_agents: Number of support agents
            config: Agent configuration
            rng: Random number generator
            
        Raises:
            ValueError: If num_agents < 1
        """
        if num_agents < 1:
            raise ValueError(f"num_agents must be at least 1, got {num_agents}")
        
        self.num_agents = num_agents
        self.config = config
        self.rng = rng
        
        # Initialize agents with random expertise profiles
        self.agents: List[SupportAgent] = self._initialize_agents()
        
        # Track active assignments per agent
        self.active_assignments: Dict[int, List[Assignment]] = {
            i: [] for i in range(num_agents)
        }
        
        # Track when assignments will complete (step -> list of (agent_id, assignment_id))
        self.completion_schedule: Dict[int, List[tuple[int, uuid.UUID]]] = {}

    def _initialize_agents(self) -> List[SupportAgent]:
        """Initialize agent pool with random expertise profiles.
        
        Returns:
            List of initialized support agents
        """
        agents = []
        min_exp, max_exp = self.config.expertise_range
        
        for agent_id in range(self.num_agents):
            # Generate random expertise for each ticket type
            expertise = self.rng.uniform(
                low=min_exp,
                high=max_exp,
                size=len(TicketType),
            )
            
            agent = SupportAgent(
                id=agent_id,
                expertise=expertise,
                current_workload=0,
                max_capacity=self.config.max_capacity,
            )
            agents.append(agent)
        
        return agents

    def assign_ticket(
        self,
        ticket: Ticket,
        agent_id: int,
        current_step: int,
    ) -> Assignment:
        """Assign a ticket to an agent.
        
        Args:
            ticket: Ticket to assign
            agent_id: ID of agent to assign to
            current_step: Current simulation step
            
        Returns:
            Created assignment object
            
        Raises:
            ValueError: If agent_id is invalid or agent is at capacity
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(
                f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents - 1}]"
            )
        
        agent = self.agents[agent_id]
        
        if not agent.is_available():
            raise ValueError(
                f"Agent {agent_id} is at capacity "
                f"({agent.current_workload}/{agent.max_capacity})"
            )
        
        # Calculate resolution time
        resolution_time = agent.get_resolution_time(ticket.type)
        
        # Create assignment
        assignment = Assignment(
            id=uuid.uuid4(),
            ticket_id=ticket.id,
            agent_id=agent_id,
            assignment_step=current_step,
            estimated_resolution_time=resolution_time,
        )
        
        # Update agent workload
        agent.current_workload += 1
        
        # Track assignment
        self.active_assignments[agent_id].append(assignment)
        
        # Schedule completion (convert hours to steps: 1000 steps = 1 day = 24 hours)
        steps_to_complete = int(resolution_time * (1000.0 / 24.0))
        completion_step = current_step + steps_to_complete
        
        if completion_step not in self.completion_schedule:
            self.completion_schedule[completion_step] = []
        self.completion_schedule[completion_step].append((agent_id, assignment.id))
        
        return assignment

    def update_workloads(self, current_step: int) -> int:
        """Update agent workloads based on completed assignments.
        
        Args:
            current_step: Current simulation step
            
        Returns:
            Number of assignments completed at this step
        """
        completed_count = 0
        
        # Check if any assignments complete at this step
        if current_step in self.completion_schedule:
            completions = self.completion_schedule.pop(current_step)
            
            for agent_id, assignment_id in completions:
                # Find and remove the assignment
                agent_assignments = self.active_assignments[agent_id]
                assignment_found = False
                
                for i, assignment in enumerate(agent_assignments):
                    if assignment.id == assignment_id:
                        agent_assignments.pop(i)
                        assignment_found = True
                        break
                
                if assignment_found:
                    # Decrease agent workload
                    agent = self.agents[agent_id]
                    agent.current_workload = max(0, agent.current_workload - 1)
                    completed_count += 1
        
        return completed_count

    def get_available_agents(self) -> List[int]:
        """Get list of agents that can accept more tickets.
        
        Returns:
            List of agent IDs with capacity available
        """
        return [
            agent.id
            for agent in self.agents
            if agent.is_available()
        ]

    def get_agent(self, agent_id: int) -> SupportAgent:
        """Get agent by ID.
        
        Args:
            agent_id: ID of agent to retrieve
            
        Returns:
            Support agent object
            
        Raises:
            ValueError: If agent_id is invalid
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(
                f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents - 1}]"
            )
        return self.agents[agent_id]

    def get_agent_workloads(self) -> np.ndarray:
        """Get current workloads for all agents.
        
        Returns:
            Array of workloads (shape: (num_agents,))
        """
        return np.array([agent.current_workload for agent in self.agents], dtype=np.int32)

    def get_workload_statistics(self) -> Dict[str, float]:
        """Calculate workload balance statistics.
        
        Returns:
            Dictionary with mean, std, min, max workload values
        """
        workloads = self.get_agent_workloads()
        return {
            'mean': float(np.mean(workloads)),
            'std': float(np.std(workloads)),
            'min': int(np.min(workloads)),
            'max': int(np.max(workloads)),
        }

    def get_total_assignments(self) -> int:
        """Get total number of active assignments across all agents.
        
        Returns:
            Total number of active assignments
        """
        return sum(len(assignments) for assignments in self.active_assignments.values())

    def reset(self) -> None:
        """Reset agent manager to initial state.
        
        Clears all assignments and resets workloads to zero.
        Agents retain their expertise profiles.
        """
        # Reset workloads
        for agent in self.agents:
            agent.current_workload = 0
        
        # Clear assignments
        self.active_assignments = {i: [] for i in range(self.num_agents)}
        
        # Clear completion schedule
        self.completion_schedule = {}

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about agent manager state.
        
        Returns:
            Dictionary with various statistics
        """
        workload_stats = self.get_workload_statistics()
        
        return {
            'num_agents': self.num_agents,
            'available_agents': len(self.get_available_agents()),
            'total_assignments': self.get_total_assignments(),
            'workload_mean': workload_stats['mean'],
            'workload_std': workload_stats['std'],
            'workload_min': workload_stats['min'],
            'workload_max': workload_stats['max'],
        }
