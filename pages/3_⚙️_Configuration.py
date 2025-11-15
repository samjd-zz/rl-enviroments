"""Configuration - Customize environment parameters."""

import streamlit as st
import yaml
from pathlib import Path

from rl_environments.ticket_routing.core.data_models import (
    EnvironmentConfig,
    AgentConfig,
    TicketConfig,
    RewardConfig,
)

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Environment Configuration")
st.markdown("Customize environment parameters and export configurations")

# Initialize session state for config
if 'custom_config' not in st.session_state:
    st.session_state.custom_config = EnvironmentConfig()

# Tabs for different configuration sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Environment",
    "👥 Agents",
    "🎫 Tickets",
    "🎁 Rewards",
    "💾 Export/Import"
])

with tab1:
    st.header("Environment Settings")
    st.markdown("Configure overall environment parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_agents = st.number_input(
            "Number of Agents",
            min_value=1,
            max_value=20,
            value=st.session_state.custom_config.num_agents,
            help="Total number of support agents in the system"
        )
        
        episode_length = st.number_input(
            "Episode Length (steps)",
            min_value=10,
            max_value=1000,
            value=st.session_state.custom_config.episode_length,
            help="Number of steps per episode"
        )
    
    with col2:
        max_queue_size = st.number_input(
            "Maximum Queue Size",
            min_value=1,
            max_value=200,
            value=st.session_state.custom_config.max_queue_size,
            help="Maximum number of tickets that can be queued"
        )
    
    st.markdown("---")
    
    # Preview
    st.subheader("Environment Preview")
    st.info(f"""
    - **Agents:** {num_agents}
    - **Episode Length:** {episode_length} steps
    - **Max Queue Size:** {max_queue_size} tickets
    - **Expected Throughput:** ~{episode_length * 0.8:.0f} tickets per episode
    """)

with tab2:
    st.header("Agent Configuration")
    st.markdown("Configure agent capacity and expertise settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_capacity = st.number_input(
            "Agent Max Capacity",
            min_value=1,
            max_value=20,
            value=st.session_state.custom_config.agent_config.max_capacity,
            help="Maximum number of tickets an agent can handle simultaneously"
        )
    
    with col2:
        st.markdown("**Expertise Range**")
        st.markdown("Define expertise level range for agents (0.0 to 1.0)")
    
    min_expertise, max_expertise = st.session_state.custom_config.agent_config.expertise_range
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_expertise_input = st.slider(
            "Minimum Expertise",
            min_value=0.0,
            max_value=1.0,
            value=min_expertise,
            step=0.1,
            help="Minimum expertise level across all ticket types"
        )
    
    with col2:
        max_expertise_input = st.slider(
            "Maximum Expertise",
            min_value=min_expertise_input,
            max_value=1.0,
            value=max_expertise,
            step=0.1,
            help="Maximum expertise level across all ticket types"
        )
    
    st.markdown("---")
    
    # Agent preview
    st.subheader("Agent Profile Preview")
    st.info(f"""
    - **Capacity:** {max_capacity} tickets
    - **Expertise Range:** {min_expertise_input:.1f} - {max_expertise_input:.1f}
    - **Average Expertise:** {(min_expertise_input + max_expertise_input) / 2:.1f}
    """)

with tab3:
    st.header("Ticket Configuration")
    st.markdown("Configure ticket generation and distribution")
    
    arrival_rate = st.slider(
        "Ticket Arrival Rate (per day)",
        min_value=1.0,
        max_value=200.0,
        value=st.session_state.custom_config.ticket_config.arrival_rate,
        step=5.0,
        help="Average number of tickets per day"
    )
    
    st.markdown("---")
    
    # Ticket preview
    st.subheader("Ticket Generation Preview")
    st.info(f"""
    - **Arrival Rate:** {arrival_rate:.0f} tickets/day
    - **Expected per Episode:** ~{arrival_rate * episode_length / 1000:.0f} tickets
    
    Note: Type and priority distributions are configured in the data models.
    """)

with tab4:
    st.header("Reward Configuration")
    st.markdown("Configure reward function parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_reward = st.slider(
            "Base Reward",
            min_value=0.0,
            max_value=20.0,
            value=st.session_state.custom_config.reward_config.base_reward,
            step=1.0,
            help="Base reward for valid assignment"
        )
        
        expertise_multiplier = st.slider(
            "Expertise Multiplier",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.custom_config.reward_config.expertise_multiplier,
            step=0.5,
            help="Multiplier for expertise bonus"
        )
        
        resolution_penalty_rate = st.slider(
            "Resolution Penalty Rate",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.custom_config.reward_config.resolution_penalty_rate,
            step=0.05,
            help="Rate for resolution time penalty"
        )
    
    with col2:
        workload_threshold = st.slider(
            "Workload Threshold",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.custom_config.reward_config.workload_threshold,
            step=0.5,
            help="Standard deviation threshold for balance penalty"
        )
        
        workload_penalty = st.slider(
            "Workload Penalty",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.custom_config.reward_config.workload_penalty,
            step=0.5,
            help="Penalty for imbalanced workload"
        )
    
    st.markdown("---")
    
    # Reward preview
    st.subheader("Reward Function Preview")
    
    st.info(f"""
    **Reward Components:**
    - Base Reward: {base_reward:.1f}
    - Expertise Multiplier: {expertise_multiplier:.1f}x
    - Resolution Penalty Rate: {resolution_penalty_rate:.2f}
    - Workload Threshold: {workload_threshold:.1f} std dev
    - Workload Penalty: {workload_penalty:.1f}
    """)

with tab5:
    st.header("Export & Import Configuration")
    
    # Apply current configuration
    if st.button("💾 Apply Current Configuration", type="primary"):
        # Create new config with current values
        agent_config = AgentConfig(
            max_capacity=max_capacity,
            expertise_range=(min_expertise_input, max_expertise_input),
        )
        
        ticket_config = TicketConfig(
            arrival_rate=arrival_rate,
        )
        
        reward_config = RewardConfig(
            base_reward=base_reward,
            expertise_multiplier=expertise_multiplier,
            resolution_penalty_rate=resolution_penalty_rate,
            workload_threshold=workload_threshold,
            workload_penalty=workload_penalty,
        )
        
        st.session_state.custom_config = EnvironmentConfig(
            num_agents=num_agents,
            episode_length=episode_length,
            max_queue_size=max_queue_size,
            agent_config=agent_config,
            ticket_config=ticket_config,
            reward_config=reward_config,
        )
        
        st.success("✅ Configuration applied successfully!")
    
    st.markdown("---")
    
    # Current configuration display
    st.subheader("📋 Current Configuration")
    
    st.json({
        'num_agents': num_agents,
        'episode_length': episode_length,
        'max_queue_size': max_queue_size,
        'agent_config': {
            'max_capacity': max_capacity,
            'expertise_range': [min_expertise_input, max_expertise_input],
        },
        'ticket_config': {
            'arrival_rate': arrival_rate,
        },
        'reward_config': {
            'base_reward': base_reward,
            'expertise_multiplier': expertise_multiplier,
            'resolution_penalty_rate': resolution_penalty_rate,
            'workload_threshold': workload_threshold,
            'workload_penalty': workload_penalty,
        }
    })
    
    st.info("Configuration is applied to the current session. Use the preset buttons below to quickly switch between difficulty modes.")

# Preset configurations
st.markdown("---")
st.header("🎛️ Preset Configurations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔰 Easy Mode", use_container_width=True):
        st.session_state.custom_config = EnvironmentConfig(
            num_agents=10,
            episode_length=100,
            max_queue_size=100,
        )
        st.success("Easy mode applied!")
        st.rerun()

with col2:
    if st.button("⚖️ Balanced Mode", use_container_width=True):
        st.session_state.custom_config = EnvironmentConfig()
        st.success("Balanced mode applied!")
        st.rerun()

with col3:
    if st.button("� Hard Mode", use_container_width=True):
        st.session_state.custom_config = EnvironmentConfig(
            num_agents=3,
            episode_length=200,
            max_queue_size=20,
        )
        st.success("Hard mode applied!")
        st.rerun()

# Info section
with st.expander("ℹ️ Configuration Tips"):
    st.markdown("""
    **Agent Configuration Tips:**
    - Higher capacity = agents can handle more tickets but may reduce specialization
    - Lower resolution time = faster turnover but may reduce quality
    - Broader expertise range = more flexibility but less specialization
    
    **Ticket Configuration Tips:**
    - Higher arrival rate = more challenging environment
    - Higher priority probability = more pressure on the system
    - Balanced type distribution = tests agent versatility
    
    **Reward Configuration Tips:**
    - Emphasize expertise for specialized team structures
    - Emphasize balance for fair workload distribution
    - Emphasize priority for critical-ticket handling
    - Higher penalties encourage more careful action selection
    """)
