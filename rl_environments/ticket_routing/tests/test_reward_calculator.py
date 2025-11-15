"""Tests for RewardCalculator component."""

import uuid

import numpy as np
import pytest

from rl_environments.ticket_routing.core.data_models import (
    Assignment,
    Priority,
    RewardConfig,
    SupportAgent,
    Ticket,
    TicketType,
)
from rl_environments.ticket_routing.core.reward_calculator import RewardCalculator


@pytest.fixture
def default_config() -> RewardConfig:
    """Provide default reward configuration."""
    return RewardConfig(
        base_reward=10.0,
        expertise_multiplier=5.0,
        resolution_penalty_rate=0.1,
        workload_threshold=2.0,
        workload_penalty=5.0,
    )


@pytest.fixture
def reward_calculator(default_config: RewardConfig) -> RewardCalculator:
    """Provide initialized reward calculator."""
    return RewardCalculator(config=default_config)


@pytest.fixture
def sample_agent() -> SupportAgent:
    """Provide sample agent for testing."""
    return SupportAgent(
        id=0,
        expertise=np.array([0.8, 0.6, 0.7, 0.5, 0.4]),  # High technical expertise
        current_workload=5,
        max_capacity=10,
    )


@pytest.fixture
def sample_ticket() -> Ticket:
    """Provide sample ticket for testing."""
    return Ticket(
        id=uuid.uuid4(),
        type=TicketType.TECHNICAL,
        priority=Priority.MEDIUM,
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


class TestRewardCalculatorInitialization:
    """Tests for RewardCalculator initialization."""
    
    def test_initialization(self, default_config: RewardConfig) -> None:
        """Test valid initialization."""
        calculator = RewardCalculator(config=default_config)
        
        assert calculator.config == default_config
        assert calculator.config.base_reward == 10.0
        assert calculator.config.expertise_multiplier == 5.0


class TestBasicRewardCalculation:
    """Tests for basic reward calculation."""
    
    def test_calculate_reward_medium_priority(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward calculation with medium priority ticket."""
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=sample_ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Expected: base(10) + expertise(0.8*5=4.0) - resolution(24*0.1=2.4) - workload(0) = 11.6
        # With medium priority multiplier (1.0): 11.6
        assert reward == pytest.approx(11.6, rel=1e-6)
    
    def test_calculate_reward_high_priority(
        self,
        reward_calculator: RewardCalculator,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward calculation with high priority ticket."""
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.HIGH,
            arrival_step=0,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Subtotal: 11.6, with high priority multiplier (1.5): 11.6 * 1.5 = 17.4
        assert reward == pytest.approx(17.4, rel=1e-6)
    
    def test_calculate_reward_critical_priority(
        self,
        reward_calculator: RewardCalculator,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward calculation with critical priority ticket."""
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.CRITICAL,
            arrival_step=0,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Subtotal: 11.6, with critical priority multiplier (2.0): 11.6 * 2.0 = 23.2
        assert reward == pytest.approx(23.2, rel=1e-6)
    
    def test_calculate_reward_low_priority(
        self,
        reward_calculator: RewardCalculator,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward calculation with low priority ticket."""
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.LOW,
            arrival_step=0,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Subtotal: 11.6, with low priority multiplier (0.8): 11.6 * 0.8 = 9.28
        assert reward == pytest.approx(9.28, rel=1e-6)


class TestExpertiseBonus:
    """Tests for expertise bonus calculation."""
    
    def test_high_expertise_gives_high_bonus(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that high expertise yields high bonus."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([1.0, 0.5, 0.5, 0.5, 0.5]),  # Max expertise in TECHNICAL
            current_workload=0,
            max_capacity=10,
        )
        
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        bonus = reward_calculator._calculate_expertise_bonus(ticket, agent)
        
        # Expertise (1.0) * multiplier (5.0) = 5.0
        assert bonus == pytest.approx(5.0)
    
    def test_low_expertise_gives_low_bonus(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that low expertise yields low bonus."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.2, 0.5, 0.5, 0.5, 0.5]),  # Low expertise in TECHNICAL
            current_workload=0,
            max_capacity=10,
        )
        
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        bonus = reward_calculator._calculate_expertise_bonus(ticket, agent)
        
        # Expertise (0.2) * multiplier (5.0) = 1.0
        assert bonus == pytest.approx(1.0)
    
    def test_expertise_bonus_varies_by_ticket_type(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that bonus varies based on ticket type."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.9, 0.3, 0.5, 0.5, 0.5]),
            current_workload=0,
            max_capacity=10,
        )
        
        technical_ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        billing_ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.BILLING,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        technical_bonus = reward_calculator._calculate_expertise_bonus(
            technical_ticket, agent
        )
        billing_bonus = reward_calculator._calculate_expertise_bonus(
            billing_ticket, agent
        )
        
        # Technical: 0.9 * 5.0 = 4.5
        # Billing: 0.3 * 5.0 = 1.5
        assert technical_bonus == pytest.approx(4.5)
        assert billing_bonus == pytest.approx(1.5)
        assert technical_bonus > billing_bonus


class TestResolutionPenalty:
    """Tests for resolution time penalty calculation."""
    
    def test_longer_resolution_higher_penalty(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
    ) -> None:
        """Test that longer resolution time yields higher penalty."""
        short_assignment = Assignment(
            id=uuid.uuid4(),
            ticket_id=sample_ticket.id,
            agent_id=0,
            assignment_step=0,
            estimated_resolution_time=10.0,
        )
        
        long_assignment = Assignment(
            id=uuid.uuid4(),
            ticket_id=sample_ticket.id,
            agent_id=0,
            assignment_step=0,
            estimated_resolution_time=40.0,
        )
        
        short_penalty = reward_calculator._calculate_resolution_penalty(short_assignment)
        long_penalty = reward_calculator._calculate_resolution_penalty(long_assignment)
        
        # Short: 10.0 * 0.1 = 1.0
        # Long: 40.0 * 0.1 = 4.0
        assert short_penalty == pytest.approx(1.0)
        assert long_penalty == pytest.approx(4.0)
        assert long_penalty > short_penalty
    
    def test_resolution_penalty_calculation(
        self,
        reward_calculator: RewardCalculator,
        sample_assignment: Assignment,
    ) -> None:
        """Test resolution penalty calculation."""
        penalty = reward_calculator._calculate_resolution_penalty(sample_assignment)
        
        # 24.0 hours * 0.1 rate = 2.4
        assert penalty == pytest.approx(2.4)


class TestWorkloadPenalty:
    """Tests for workload balance penalty calculation."""
    
    def test_balanced_workload_no_penalty(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that balanced workload incurs no penalty."""
        # All agents have same workload (std = 0)
        balanced_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        penalty = reward_calculator._calculate_workload_penalty(balanced_workloads)
        
        assert penalty == 0.0
    
    def test_slightly_imbalanced_workload_no_penalty(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that slight imbalance below threshold incurs no penalty."""
        # Slight variance but std < threshold (2.0)
        workloads = np.array([5, 5, 6, 5, 5], dtype=np.int32)
        
        penalty = reward_calculator._calculate_workload_penalty(workloads)
        
        # std ≈ 0.4, below threshold of 2.0
        assert penalty == 0.0
    
    def test_imbalanced_workload_penalty(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that high imbalance incurs penalty."""
        # High variance (std > threshold)
        imbalanced_workloads = np.array([1, 2, 5, 8, 10], dtype=np.int32)
        
        penalty = reward_calculator._calculate_workload_penalty(imbalanced_workloads)
        
        # std ≈ 3.35 > threshold (2.0), so penalty = 5.0
        assert penalty == 5.0
    
    def test_empty_workload_array(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test workload penalty with empty array."""
        empty_workloads = np.array([], dtype=np.int32)
        
        penalty = reward_calculator._calculate_workload_penalty(empty_workloads)
        
        assert penalty == 0.0


class TestPriorityMultipliers:
    """Tests for priority-based reward multipliers."""
    
    def test_critical_priority_multiplier(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test critical priority multiplier."""
        multiplier = reward_calculator._get_priority_multiplier(Priority.CRITICAL)
        assert multiplier == 2.0
    
    def test_high_priority_multiplier(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test high priority multiplier."""
        multiplier = reward_calculator._get_priority_multiplier(Priority.HIGH)
        assert multiplier == 1.5
    
    def test_medium_priority_multiplier(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test medium priority multiplier."""
        multiplier = reward_calculator._get_priority_multiplier(Priority.MEDIUM)
        assert multiplier == 1.0
    
    def test_low_priority_multiplier(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test low priority multiplier."""
        multiplier = reward_calculator._get_priority_multiplier(Priority.LOW)
        assert multiplier == 0.8
    
    def test_priority_ordering(
        self,
        reward_calculator: RewardCalculator,
    ) -> None:
        """Test that priority multipliers are properly ordered."""
        critical = reward_calculator._get_priority_multiplier(Priority.CRITICAL)
        high = reward_calculator._get_priority_multiplier(Priority.HIGH)
        medium = reward_calculator._get_priority_multiplier(Priority.MEDIUM)
        low = reward_calculator._get_priority_multiplier(Priority.LOW)
        
        assert critical > high > medium > low


class TestRewardBreakdown:
    """Tests for reward breakdown analysis."""
    
    def test_get_reward_breakdown(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test getting detailed reward breakdown."""
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        breakdown = reward_calculator.get_reward_breakdown(
            ticket=sample_ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Verify all components present
        assert 'base_reward' in breakdown
        assert 'expertise_bonus' in breakdown
        assert 'resolution_penalty' in breakdown
        assert 'workload_penalty' in breakdown
        assert 'subtotal' in breakdown
        assert 'priority_multiplier' in breakdown
        assert 'total_reward' in breakdown
        
        # Verify values
        assert breakdown['base_reward'] == 10.0
        assert breakdown['expertise_bonus'] == pytest.approx(4.0)  # 0.8 * 5.0
        assert breakdown['resolution_penalty'] == pytest.approx(2.4)  # 24.0 * 0.1
        assert breakdown['workload_penalty'] == 0.0
        assert breakdown['subtotal'] == pytest.approx(11.6)
        assert breakdown['priority_multiplier'] == 1.0  # Medium priority
        assert breakdown['total_reward'] == pytest.approx(11.6)
    
    def test_breakdown_matches_calculate_reward(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_agent: SupportAgent,
        sample_assignment: Assignment,
    ) -> None:
        """Test that breakdown total matches direct calculation."""
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        direct_reward = reward_calculator.calculate_reward(
            ticket=sample_ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        breakdown = reward_calculator.get_reward_breakdown(
            ticket=sample_ticket,
            agent=sample_agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        assert breakdown['total_reward'] == pytest.approx(direct_reward)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_expertise(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward with zero expertise."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            current_workload=0,
            max_capacity=10,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=sample_ticket,
            agent=agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Base(10) + expertise(0) - resolution(2.4) - workload(0) = 7.6
        assert reward == pytest.approx(7.6)
    
    def test_maximum_expertise(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_assignment: Assignment,
    ) -> None:
        """Test reward with maximum expertise."""
        agent = SupportAgent(
            id=0,
            expertise=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            current_workload=0,
            max_capacity=10,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=sample_ticket,
            agent=agent,
            assignment=sample_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Base(10) + expertise(5.0) - resolution(2.4) - workload(0) = 12.6
        assert reward == pytest.approx(12.6)
    
    def test_very_long_resolution_time(
        self,
        reward_calculator: RewardCalculator,
        sample_ticket: Ticket,
        sample_agent: SupportAgent,
    ) -> None:
        """Test reward with very long resolution time."""
        long_assignment = Assignment(
            id=uuid.uuid4(),
            ticket_id=sample_ticket.id,
            agent_id=0,
            assignment_step=0,
            estimated_resolution_time=100.0,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = reward_calculator.calculate_reward(
            ticket=sample_ticket,
            agent=sample_agent,
            assignment=long_assignment,
            agent_workloads=agent_workloads,
        )
        
        # Base(10) + expertise(4.0) - resolution(10.0) - workload(0) = 4.0
        assert reward == pytest.approx(4.0)
    
    def test_custom_config(self) -> None:
        """Test reward calculator with custom configuration."""
        custom_config = RewardConfig(
            base_reward=20.0,
            expertise_multiplier=10.0,
            resolution_penalty_rate=0.2,
            workload_threshold=1.0,
            workload_penalty=10.0,
        )
        
        calculator = RewardCalculator(config=custom_config)
        
        agent = SupportAgent(
            id=0,
            expertise=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            current_workload=0,
            max_capacity=10,
        )
        
        ticket = Ticket(
            id=uuid.uuid4(),
            type=TicketType.TECHNICAL,
            priority=Priority.MEDIUM,
            arrival_step=0,
        )
        
        assignment = Assignment(
            id=uuid.uuid4(),
            ticket_id=ticket.id,
            agent_id=0,
            assignment_step=0,
            estimated_resolution_time=20.0,
        )
        
        agent_workloads = np.array([5, 5, 5, 5, 5], dtype=np.int32)
        
        reward = calculator.calculate_reward(
            ticket=ticket,
            agent=agent,
            assignment=assignment,
            agent_workloads=agent_workloads,
        )
        
        # Base(20) + expertise(0.5*10=5.0) - resolution(20*0.2=4.0) - workload(0) = 21.0
        assert reward == pytest.approx(21.0)
