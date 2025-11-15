"""State management for support ticket routing environment."""

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from .data_models import (
    Assignment,
    EnvironmentState,
    Ticket,
)


class StateManager:
    """Manages environment state and episode metrics.
    
    Tracks current state, assignment history, and calculates
    episode-level performance metrics.
    
    Args:
        max_queue_size: Maximum size of ticket queue
        num_agents: Number of support agents
    """

    def __init__(self, max_queue_size: int, num_agents: int) -> None:
        """Initialize state manager.
        
        Args:
            max_queue_size: Maximum size of ticket queue
            num_agents: Number of support agents
            
        Raises:
            ValueError: If max_queue_size or num_agents is invalid
        """
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {max_queue_size}")
        
        if num_agents < 1:
            raise ValueError(f"num_agents must be at least 1, got {num_agents}")
        
        self.max_queue_size = max_queue_size
        self.num_agents = num_agents
        
        # Initialize state
        self.step: int = 0
        self.tickets_processed: int = 0
        self.total_resolution_time: float = 0.0
        self.current_ticket: Optional[Ticket] = None
        self.queue: Deque[Ticket] = deque(maxlen=max_queue_size)
        
        # Assignment history
        self.assignment_history: List[Assignment] = []
        
        # Episode statistics
        self.episode_tickets_by_type: Dict[int, int] = {}
        self.episode_tickets_by_priority: Dict[int, int] = {}

    def update_state(
        self,
        ticket: Ticket,
        agent_id: int,
        assignment: Assignment,
    ) -> None:
        """Update state after ticket assignment.
        
        Args:
            ticket: Assigned ticket
            agent_id: ID of agent assigned to
            assignment: Created assignment object
        """
        # Update metrics
        self.tickets_processed += 1
        self.total_resolution_time += assignment.estimated_resolution_time
        
        # Track assignment history
        self.assignment_history.append(assignment)
        
        # Update episode statistics
        ticket_type_value = ticket.type.value
        if ticket_type_value not in self.episode_tickets_by_type:
            self.episode_tickets_by_type[ticket_type_value] = 0
        self.episode_tickets_by_type[ticket_type_value] += 1
        
        ticket_priority_value = ticket.priority.value
        if ticket_priority_value not in self.episode_tickets_by_priority:
            self.episode_tickets_by_priority[ticket_priority_value] = 0
        self.episode_tickets_by_priority[ticket_priority_value] += 1
        
        # Clear current ticket since it's been assigned
        self.current_ticket = None

    def set_current_ticket(self, ticket: Optional[Ticket]) -> None:
        """Set the current ticket being processed.
        
        Args:
            ticket: Ticket to set as current, or None
        """
        self.current_ticket = ticket

    def add_to_queue(self, ticket: Ticket) -> bool:
        """Add ticket to queue.
        
        Args:
            ticket: Ticket to add
            
        Returns:
            True if added successfully, False if queue is full
        """
        if len(self.queue) >= self.max_queue_size:
            return False
        
        self.queue.append(ticket)
        return True

    def pop_from_queue(self) -> Optional[Ticket]:
        """Remove and return ticket from front of queue.
        
        Returns:
            Next ticket from queue, or None if empty
        """
        if len(self.queue) == 0:
            return None
        
        return self.queue.popleft()

    def peek_queue(self) -> Optional[Ticket]:
        """View ticket at front of queue without removing.
        
        Returns:
            Next ticket from queue, or None if empty
        """
        if len(self.queue) == 0:
            return None
        
        return self.queue[0]

    def get_queue_size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of tickets in queue
        """
        return len(self.queue)

    def is_queue_full(self) -> bool:
        """Check if queue is at capacity.
        
        Returns:
            True if queue is full, False otherwise
        """
        return len(self.queue) >= self.max_queue_size

    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.step += 1

    def get_state(self, agent_workloads: np.ndarray) -> EnvironmentState:
        """Get current environment state.
        
        Args:
            agent_workloads: Current workloads for all agents
            
        Returns:
            Complete environment state
        """
        # Create active assignments map (agent_id -> assignments)
        active_assignments: Dict[int, List[Assignment]] = {
            i: [] for i in range(self.num_agents)
        }
        
        for assignment in self.assignment_history:
            # Only include recent assignments (simple heuristic)
            if assignment.agent_id in active_assignments:
                active_assignments[assignment.agent_id].append(assignment)
        
        return EnvironmentState(
            step=self.step,
            tickets_processed=self.tickets_processed,
            total_resolution_time=self.total_resolution_time,
            agent_workloads=agent_workloads.copy(),
            current_ticket=self.current_ticket,
            queue=deque(self.queue),  # Create a copy
            active_assignments=active_assignments,
        )

    def get_metrics(self) -> Dict[str, float]:
        """Calculate episode performance metrics.
        
        Returns:
            Dictionary of episode metrics
        """
        metrics: Dict[str, float] = {
            'step': float(self.step),
            'tickets_processed': float(self.tickets_processed),
            'total_resolution_time': self.total_resolution_time,
            'queue_size': float(len(self.queue)),
        }
        
        # Average resolution time
        if self.tickets_processed > 0:
            metrics['avg_resolution_time'] = (
                self.total_resolution_time / self.tickets_processed
            )
        else:
            metrics['avg_resolution_time'] = 0.0
        
        # Ticket type distribution
        for ticket_type, count in self.episode_tickets_by_type.items():
            metrics[f'tickets_type_{ticket_type}'] = float(count)
        
        # Ticket priority distribution
        for priority, count in self.episode_tickets_by_priority.items():
            metrics[f'tickets_priority_{priority}'] = float(count)
        
        return metrics

    def get_episode_statistics(self) -> Dict[str, any]:
        """Get comprehensive episode statistics.
        
        Returns:
            Dictionary with detailed episode statistics
        """
        metrics = self.get_metrics()
        
        stats = {
            'step': self.step,
            'tickets_processed': self.tickets_processed,
            'total_assignments': len(self.assignment_history),
            'queue_size': len(self.queue),
            'avg_resolution_time': metrics.get('avg_resolution_time', 0.0),
            'tickets_by_type': dict(self.episode_tickets_by_type),
            'tickets_by_priority': dict(self.episode_tickets_by_priority),
        }
        
        return stats

    def reset(self) -> None:
        """Reset state manager to initial state.
        
        Clears all state, metrics, and history.
        """
        self.step = 0
        self.tickets_processed = 0
        self.total_resolution_time = 0.0
        self.current_ticket = None
        self.queue.clear()
        self.assignment_history.clear()
        self.episode_tickets_by_type.clear()
        self.episode_tickets_by_priority.clear()
