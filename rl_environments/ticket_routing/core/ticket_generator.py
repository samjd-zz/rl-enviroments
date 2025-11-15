"""Ticket generation using Poisson process."""

import uuid
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from rl_environments.ticket_routing.core.data_models import Ticket, TicketType, Priority, TicketConfig


class TicketGenerator:
    """Generates support tickets using a Poisson arrival process.
    
    The generator simulates ticket arrivals with configurable arrival rate and
    distributions for ticket types and priorities. Uses a Poisson process to
    model realistic arrival patterns.
    
    Args:
        config: Configuration for ticket generation
        rng: NumPy random number generator for reproducibility
        
    Example:
        >>> config = TicketConfig(arrival_rate=50.0)
        >>> rng = np.random.default_rng(seed=42)
        >>> generator = TicketGenerator(config, rng)
        >>> tickets = generator.generate_tickets(current_step=10)
        >>> len(tickets) >= 0  # Tickets generated stochastically
        True
    """

    def __init__(self, config: TicketConfig, rng: np.random.Generator) -> None:
        """Initialize ticket generator.
        
        Args:
            config: Ticket generation configuration
            rng: Random number generator for deterministic generation
        """
        self.config = config
        self.rng = rng
        self.queue: Deque[Ticket] = deque()
        self.last_step = 0
        self._ticket_count = 0

    def generate_tickets(self, current_step: int) -> List[Ticket]:
        """Generate tickets based on time elapsed since last generation.
        
        Uses Poisson process to determine number of arrivals, then samples
        ticket types and priorities from configured distributions.
        
        Args:
            current_step: Current simulation step
            
        Returns:
            List of newly generated tickets
            
        Raises:
            ValueError: If current_step < last_step (time going backwards)
            
        Example:
            >>> generator = TicketGenerator(TicketConfig(), np.random.default_rng(42))
            >>> tickets = generator.generate_tickets(100)
            >>> all(isinstance(t, Ticket) for t in tickets)
            True
        """
        if current_step < self.last_step:
            raise ValueError(
                f"current_step ({current_step}) cannot be less than "
                f"last_step ({self.last_step})"
            )
        
        # Calculate time elapsed in days (1000 steps = 1 day)
        time_delta = (current_step - self.last_step) / 1000.0
        
        # Calculate expected number of arrivals using Poisson process
        expected_arrivals = self.config.arrival_rate * time_delta
        num_arrivals = self.rng.poisson(expected_arrivals)
        
        # Generate tickets
        tickets: List[Ticket] = []
        for _ in range(num_arrivals):
            ticket = self._create_ticket(current_step)
            tickets.append(ticket)
            self.queue.append(ticket)
        
        self.last_step = current_step
        return tickets

    def _create_ticket(self, arrival_step: int) -> Ticket:
        """Create a single ticket with sampled attributes.
        
        Args:
            arrival_step: Step when ticket arrives
            
        Returns:
            Newly created ticket
        """
        # Sample ticket type from distribution
        ticket_types = list(self.config.type_distribution.keys())
        type_probs = list(self.config.type_distribution.values())
        ticket_type = self.rng.choice(ticket_types, p=type_probs)
        
        # Sample priority from distribution
        priorities = list(self.config.priority_distribution.keys())
        priority_probs = list(self.config.priority_distribution.values())
        priority = self.rng.choice(priorities, p=priority_probs)
        
        # Create ticket
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType(ticket_type),
            priority=Priority(priority),
            arrival_step=arrival_step
        )
        
        self._ticket_count += 1
        return ticket

    def get_next_ticket(self) -> Optional[Ticket]:
        """Pop and return the next ticket from the queue.
        
        Returns:
            Next ticket if queue is not empty, None otherwise
            
        Example:
            >>> generator = TicketGenerator(TicketConfig(), np.random.default_rng(42))
            >>> generator.generate_tickets(100)  # doctest: +ELLIPSIS
            [...]
            >>> ticket = generator.get_next_ticket()
            >>> ticket is None or isinstance(ticket, Ticket)
            True
        """
        if len(self.queue) == 0:
            return None
        return self.queue.popleft()

    def peek_next_ticket(self) -> Optional[Ticket]:
        """View the next ticket without removing it from queue.
        
        Returns:
            Next ticket if queue is not empty, None otherwise
        """
        if len(self.queue) == 0:
            return None
        return self.queue[0]

    def get_queue_size(self) -> int:
        """Get current number of tickets in queue.
        
        Returns:
            Number of tickets waiting in queue
        """
        return len(self.queue)

    def reset(self) -> None:
        """Reset generator state to initial conditions.
        
        Clears the queue and resets counters.
        
        Example:
            >>> generator = TicketGenerator(TicketConfig(), np.random.default_rng(42))
            >>> generator.generate_tickets(100)  # doctest: +ELLIPSIS
            [...]
            >>> generator.reset()
            >>> generator.get_queue_size()
            0
        """
        self.queue.clear()
        self.last_step = 0
        self._ticket_count = 0

    def get_statistics(self) -> dict:
        """Get generator statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return {
            'total_generated': self._ticket_count,
            'current_queue_size': len(self.queue),
            'last_step': self.last_step,
        }
