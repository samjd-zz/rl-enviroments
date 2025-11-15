"""Reward calculation for support ticket routing environment."""

from typing import Dict

import numpy as np

from .data_models import (
    Assignment,
    Priority,
    RewardConfig,
    SupportAgent,
    Ticket,
)


class RewardCalculator:
    """Calculates rewards for ticket assignments.
    
    Implements a multi-component reward function that balances:
    - Base reward for valid assignment
    - Expertise bonus for good agent-ticket matching
    - Resolution time penalty
    - Workload balance penalty
    - Priority-based multipliers
    
    Args:
        config: Reward calculation configuration
    """

    def __init__(self, config: RewardConfig) -> None:
        """Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
        """
        self.config = config

    def calculate_reward(
        self,
        ticket: Ticket,
        agent: SupportAgent,
        assignment: Assignment,
        agent_workloads: np.ndarray,
    ) -> float:
        """Calculate reward for a ticket assignment.
        
        Args:
            ticket: Assigned ticket
            agent: Agent assigned to
            assignment: Created assignment
            agent_workloads: Current workloads of all agents
            
        Returns:
            Calculated reward value
        """
        # Start with base reward
        reward = self.config.base_reward
        
        # Add expertise bonus
        expertise_bonus = self._calculate_expertise_bonus(ticket, agent)
        reward += expertise_bonus
        
        # Subtract resolution time penalty
        resolution_penalty = self._calculate_resolution_penalty(assignment)
        reward -= resolution_penalty
        
        # Subtract workload imbalance penalty
        workload_penalty = self._calculate_workload_penalty(agent_workloads)
        reward -= workload_penalty
        
        # Apply priority multiplier
        priority_multiplier = self._get_priority_multiplier(ticket.priority)
        reward *= priority_multiplier
        
        return reward

    def _calculate_expertise_bonus(
        self,
        ticket: Ticket,
        agent: SupportAgent,
    ) -> float:
        """Calculate bonus for agent-ticket expertise matching.
        
        Higher expertise for the ticket type yields higher bonus.
        
        Args:
            ticket: Assigned ticket
            agent: Agent assigned to
            
        Returns:
            Expertise bonus value
        """
        expertise_level = agent.expertise[ticket.type.value]
        # Scale expertise (0.0-1.0) by multiplier
        bonus = expertise_level * self.config.expertise_multiplier
        return bonus

    def _calculate_resolution_penalty(self, assignment: Assignment) -> float:
        """Calculate penalty based on expected resolution time.
        
        Longer resolution times incur higher penalties to encourage
        efficient assignments.
        
        Args:
            assignment: Created assignment
            
        Returns:
            Resolution time penalty value
        """
        # Penalty increases with resolution time
        penalty = (
            assignment.estimated_resolution_time
            * self.config.resolution_penalty_rate
        )
        return penalty

    def _calculate_workload_penalty(self, agent_workloads: np.ndarray) -> float:
        """Calculate penalty for workload imbalance.
        
        Penalizes assignments that create high workload variance,
        encouraging balanced distribution across agents.
        
        Args:
            agent_workloads: Current workloads of all agents
            
        Returns:
            Workload imbalance penalty value
        """
        if len(agent_workloads) == 0:
            return 0.0
        
        # Calculate workload standard deviation
        workload_std = float(np.std(agent_workloads))
        
        # Apply penalty if std exceeds threshold
        if workload_std > self.config.workload_threshold:
            penalty = self.config.workload_penalty
        else:
            penalty = 0.0
        
        return penalty

    def _get_priority_multiplier(self, priority: Priority) -> float:
        """Get reward multiplier based on ticket priority.
        
        Higher priority tickets receive higher reward multipliers
        to encourage their timely assignment.
        
        Args:
            priority: Ticket priority level
            
        Returns:
            Priority-based reward multiplier
        """
        # Priority multipliers (higher priority -> higher multiplier)
        multipliers = {
            Priority.CRITICAL: 2.0,
            Priority.HIGH: 1.5,
            Priority.MEDIUM: 1.0,
            Priority.LOW: 0.8,
        }
        
        return multipliers.get(priority, 1.0)

    def get_reward_breakdown(
        self,
        ticket: Ticket,
        agent: SupportAgent,
        assignment: Assignment,
        agent_workloads: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate reward with component breakdown for analysis.
        
        Args:
            ticket: Assigned ticket
            agent: Agent assigned to
            assignment: Created assignment
            agent_workloads: Current workloads of all agents
            
        Returns:
            Dictionary with reward components and total
        """
        base_reward = self.config.base_reward
        expertise_bonus = self._calculate_expertise_bonus(ticket, agent)
        resolution_penalty = self._calculate_resolution_penalty(assignment)
        workload_penalty = self._calculate_workload_penalty(agent_workloads)
        priority_multiplier = self._get_priority_multiplier(ticket.priority)
        
        # Calculate total (before multiplier)
        subtotal = (
            base_reward
            + expertise_bonus
            - resolution_penalty
            - workload_penalty
        )
        
        # Apply multiplier
        total = subtotal * priority_multiplier
        
        return {
            'base_reward': base_reward,
            'expertise_bonus': expertise_bonus,
            'resolution_penalty': resolution_penalty,
            'workload_penalty': workload_penalty,
            'subtotal': subtotal,
            'priority_multiplier': priority_multiplier,
            'total_reward': total,
        }
