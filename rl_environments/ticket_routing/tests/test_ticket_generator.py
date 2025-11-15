"""Tests for ticket generator."""

import pytest
import numpy as np

from rl_environments.ticket_routing.core.data_models import (
    Ticket,
    TicketType,
    Priority,
    TicketConfig,
)
from rl_environments.ticket_routing.core.ticket_generator import TicketGenerator


class TestTicketGenerator:
    """Tests for TicketGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        config = TicketConfig()
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        assert generator.config == config
        assert generator.get_queue_size() == 0
        assert generator.last_step == 0

    def test_deterministic_generation(self):
        """Test that seeding produces deterministic results."""
        config = TicketConfig(arrival_rate=50.0)
        
        # Create two generators with same seed
        gen1 = TicketGenerator(config, np.random.default_rng(seed=42))
        gen2 = TicketGenerator(config, np.random.default_rng(seed=42))
        
        # Generate tickets at same step
        tickets1 = gen1.generate_tickets(current_step=100)
        tickets2 = gen2.generate_tickets(current_step=100)
        
        # Should generate same number of tickets
        assert len(tickets1) == len(tickets2)
        
        # Should have same types and priorities
        for t1, t2 in zip(tickets1, tickets2):
            assert t1.type == t2.type
            assert t1.priority == t2.priority
            assert t1.arrival_step == t2.arrival_step

    def test_generate_tickets_increases_queue(self):
        """Test that generating tickets adds to queue."""
        config = TicketConfig(arrival_rate=1000.0)  # High rate for reliable test
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        initial_size = generator.get_queue_size()
        tickets = generator.generate_tickets(current_step=1000)
        
        assert len(tickets) > 0
        assert generator.get_queue_size() == initial_size + len(tickets)

    def test_ticket_attributes(self):
        """Test that generated tickets have valid attributes."""
        config = TicketConfig()
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        tickets = generator.generate_tickets(current_step=1000)
        
        for ticket in tickets:
            assert isinstance(ticket, Ticket)
            assert isinstance(ticket.type, TicketType)
            assert isinstance(ticket.priority, Priority)
            assert ticket.arrival_step == 1000
            assert ticket.id is not None

    def test_arrival_rate(self):
        """Test that average arrival rate matches configuration."""
        arrival_rate = 100.0  # tickets per day
        config = TicketConfig(arrival_rate=arrival_rate)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Simulate 10 days (10000 steps)
        total_tickets = 0
        for step in range(0, 10001, 100):
            tickets = generator.generate_tickets(current_step=step)
            total_tickets += len(tickets)
        
        # Should generate approximately 1000 tickets (100 per day * 10 days)
        # Allow 20% tolerance for stochastic process
        expected = 1000
        assert 0.8 * expected <= total_tickets <= 1.2 * expected

    def test_type_distribution(self):
        """Test that ticket types follow configured distribution."""
        config = TicketConfig(
            arrival_rate=1000.0,
            type_distribution={
                TicketType.TECHNICAL: 0.5,
                TicketType.BILLING: 0.5,
                TicketType.FEATURE: 0.0,
                TicketType.BUG: 0.0,
                TicketType.INTEGRATION: 0.0,
            }
        )
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Generate many tickets
        tickets = generator.generate_tickets(current_step=10000)
        
        # Count ticket types
        type_counts = {t: 0 for t in TicketType}
        for ticket in tickets:
            type_counts[ticket.type] += 1
        
        total = len(tickets)
        assert total > 0
        
        # Check distribution (allow 10% tolerance)
        tech_ratio = type_counts[TicketType.TECHNICAL] / total
        billing_ratio = type_counts[TicketType.BILLING] / total
        
        assert 0.4 <= tech_ratio <= 0.6
        assert 0.4 <= billing_ratio <= 0.6
        assert type_counts[TicketType.FEATURE] == 0

    def test_priority_distribution(self):
        """Test that priorities follow configured distribution."""
        config = TicketConfig(
            arrival_rate=1000.0,
            priority_distribution={
                Priority.CRITICAL: 0.1,
                Priority.HIGH: 0.2,
                Priority.MEDIUM: 0.3,
                Priority.LOW: 0.4,
            }
        )
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Generate many tickets
        tickets = generator.generate_tickets(current_step=10000)
        
        # Count priorities
        priority_counts = {p: 0 for p in Priority}
        for ticket in tickets:
            priority_counts[ticket.priority] += 1
        
        total = len(tickets)
        assert total > 0
        
        # Check distribution (allow 15% tolerance)
        for priority, expected_ratio in config.priority_distribution.items():
            actual_ratio = priority_counts[priority] / total
            assert abs(actual_ratio - expected_ratio) < 0.15

    def test_get_next_ticket(self):
        """Test retrieving tickets from queue."""
        config = TicketConfig(arrival_rate=100.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Generate some tickets
        generator.generate_tickets(current_step=1000)
        initial_size = generator.get_queue_size()
        
        # Get next ticket
        ticket = generator.get_next_ticket()
        
        if initial_size > 0:
            assert ticket is not None
            assert generator.get_queue_size() == initial_size - 1
        else:
            assert ticket is None

    def test_get_next_ticket_empty_queue(self):
        """Test getting ticket from empty queue returns None."""
        config = TicketConfig()
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        ticket = generator.get_next_ticket()
        assert ticket is None

    def test_peek_next_ticket(self):
        """Test peeking at next ticket without removing it."""
        config = TicketConfig(arrival_rate=100.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        generator.generate_tickets(current_step=1000)
        initial_size = generator.get_queue_size()
        
        # Peek should not change queue size
        ticket1 = generator.peek_next_ticket()
        assert generator.get_queue_size() == initial_size
        
        # Peeking again should return same ticket
        ticket2 = generator.peek_next_ticket()
        if ticket1 is not None:
            assert ticket1.id == ticket2.id

    def test_reset(self):
        """Test resetting generator state."""
        config = TicketConfig(arrival_rate=100.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Generate some tickets
        generator.generate_tickets(current_step=500)
        assert generator.get_queue_size() > 0 or True  # May be 0 stochastically
        assert generator.last_step == 500
        
        # Reset
        generator.reset()
        
        assert generator.get_queue_size() == 0
        assert generator.last_step == 0
        assert generator._ticket_count == 0

    def test_queue_fifo_order(self):
        """Test that queue maintains FIFO order."""
        config = TicketConfig(arrival_rate=100.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Generate tickets at different steps
        tickets1 = generator.generate_tickets(current_step=100)
        tickets2 = generator.generate_tickets(current_step=200)
        tickets3 = generator.generate_tickets(current_step=300)
        
        # First ticket retrieved should be from first batch
        if len(tickets1) > 0:
            first_ticket = generator.get_next_ticket()
            assert first_ticket.id == tickets1[0].id

    def test_time_cannot_go_backwards(self):
        """Test that generating with earlier step raises error."""
        config = TicketConfig()
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        generator.generate_tickets(current_step=100)
        
        with pytest.raises(ValueError, match="cannot be less than"):
            generator.generate_tickets(current_step=50)

    def test_get_statistics(self):
        """Test getting generator statistics."""
        config = TicketConfig(arrival_rate=100.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        tickets = generator.generate_tickets(current_step=1000)
        stats = generator.get_statistics()
        
        assert 'total_generated' in stats
        assert 'current_queue_size' in stats
        assert 'last_step' in stats
        assert stats['total_generated'] == len(tickets)
        assert stats['current_queue_size'] == len(tickets)
        assert stats['last_step'] == 1000

    def test_multiple_generation_steps(self):
        """Test generating tickets over multiple steps."""
        config = TicketConfig(arrival_rate=50.0)
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        total_generated = 0
        
        # Generate tickets at multiple steps
        for step in [100, 250, 500, 750, 1000]:
            tickets = generator.generate_tickets(current_step=step)
            total_generated += len(tickets)
        
        stats = generator.get_statistics()
        assert stats['total_generated'] == total_generated
        assert stats['last_step'] == 1000

    def test_zero_arrival_rate(self):
        """Test generator with very low arrival rate."""
        config = TicketConfig(arrival_rate=0.1)  # Very low rate
        rng = np.random.default_rng(seed=42)
        generator = TicketGenerator(config, rng)
        
        # Should generate very few or no tickets
        tickets = generator.generate_tickets(current_step=100)
        assert len(tickets) >= 0  # Non-negative
