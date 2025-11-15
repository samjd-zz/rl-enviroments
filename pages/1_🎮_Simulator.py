"""Environment Simulator - Interactive step-by-step simulation."""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from rl_environments.ticket_routing import TicketRoutingEnv
from rl_environments.ticket_routing.core.data_models import EnvironmentConfig

st.set_page_config(page_title="Environment Simulator", page_icon="🎮", layout="wide")

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'episode_data' not in st.session_state:
    st.session_state.episode_data = []
if 'current_obs' not in st.session_state:
    st.session_state.current_obs = None
if 'current_info' not in st.session_state:
    st.session_state.current_info = None
if 'terminated' not in st.session_state:
    st.session_state.terminated = False
if 'truncated' not in st.session_state:
    st.session_state.truncated = False
if 'total_reward' not in st.session_state:
    st.session_state.total_reward = 0.0

st.title("🎮 Environment Simulator")
st.markdown("Interactive step-by-step simulation of the Ticket Routing Environment")

# Sidebar controls
st.sidebar.header("Simulation Controls")

# Environment initialization
if st.sidebar.button("🔄 Initialize Environment", type="primary", use_container_width=True):
    config = EnvironmentConfig()
    st.session_state.env = TicketRoutingEnv(config=config)
    st.session_state.current_obs, st.session_state.current_info = st.session_state.env.reset(seed=42)
    st.session_state.episode_data = []
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.total_reward = 0.0
    st.success("✅ Environment initialized!")

if st.session_state.env is None:
    st.info("👆 Click 'Initialize Environment' to start")
    st.stop()

# Check if episode is done
episode_done = st.session_state.terminated or st.session_state.truncated

# Action selection
st.sidebar.markdown("---")
st.sidebar.subheader("Take Action")

if not episode_done:
    num_agents = st.session_state.env.config.num_agents
    action = st.sidebar.selectbox(
        "Select Agent to Assign Ticket",
        range(num_agents),
        format_func=lambda x: f"Agent {x}",
    )
    
    if st.sidebar.button("▶️ Step", use_container_width=True):
        obs, reward, terminated, truncated, info = st.session_state.env.step(action)
        
        # Store step data
        st.session_state.episode_data.append({
            'step': st.session_state.current_info['step'],
            'action': action,
            'reward': reward,
            'queue_size': info['queue_size'],
            'available_agents': info['available_agents'],
        })
        
        st.session_state.current_obs = obs
        st.session_state.current_info = info
        st.session_state.terminated = terminated
        st.session_state.truncated = truncated
        st.session_state.total_reward += reward
        
        if reward > 0:
            st.sidebar.success(f"✅ Reward: +{reward:.2f}")
        else:
            st.sidebar.error(f"❌ Reward: {reward:.2f}")

else:
    st.sidebar.warning("Episode finished!")
    if st.sidebar.button("🔄 Reset Episode", use_container_width=True):
        st.session_state.current_obs, st.session_state.current_info = st.session_state.env.reset(seed=42)
        st.session_state.episode_data = []
        st.session_state.terminated = False
        st.session_state.truncated = False
        st.session_state.total_reward = 0.0
        st.rerun()

# Random policy option
st.sidebar.markdown("---")
if st.sidebar.button("🎲 Random Action", disabled=episode_done, use_container_width=True):
    action = st.session_state.env.action_space.sample()
    obs, reward, terminated, truncated, info = st.session_state.env.step(action)
    
    st.session_state.episode_data.append({
        'step': st.session_state.current_info['step'],
        'action': action,
        'reward': reward,
        'queue_size': info['queue_size'],
        'available_agents': info['available_agents'],
    })
    
    st.session_state.current_obs = obs
    st.session_state.current_info = info
    st.session_state.terminated = terminated
    st.session_state.truncated = truncated
    st.session_state.total_reward += reward
    st.rerun()

# Auto-play option
if st.sidebar.button("⏩ Auto-play Episode (Random Policy)", disabled=episode_done, use_container_width=True):
    with st.spinner("Running episode..."):
        while not (st.session_state.terminated or st.session_state.truncated):
            action = st.session_state.env.action_space.sample()
            obs, reward, terminated, truncated, info = st.session_state.env.step(action)
            
            st.session_state.episode_data.append({
                'step': st.session_state.current_info['step'],
                'action': action,
                'reward': reward,
                'queue_size': info['queue_size'],
                'available_agents': info['available_agents'],
            })
            
            st.session_state.current_obs = obs
            st.session_state.current_info = info
            st.session_state.terminated = terminated
            st.session_state.truncated = truncated
            st.session_state.total_reward += reward
    
    st.success(f"Episode complete! Total reward: {st.session_state.total_reward:.2f}")
    st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Current State")
    
    if st.session_state.current_info:
        # Episode metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Step", st.session_state.current_info['step'])
        with metric_cols[1]:
            st.metric("Total Reward", f"{st.session_state.total_reward:.2f}")
        with metric_cols[2]:
            st.metric("Queue Size", st.session_state.current_info['queue_size'])
        with metric_cols[3]:
            st.metric("Available Agents", st.session_state.current_info['available_agents'])
        
        # Current ticket info
        st.markdown("#### 🎫 Current Ticket")
        if st.session_state.current_obs and st.session_state.current_obs['current_ticket'][0] > 0:
            ticket_type = int(st.session_state.current_obs['current_ticket'][0])
            ticket_priority = int(st.session_state.current_obs['current_ticket'][1])
            
            ticket_types = ['Bug', 'Feature Request', 'Question', 'Configuration', 'Integration']
            priorities = ['Low', 'Medium', 'High', 'Critical']
            
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.info(f"**Type:** {ticket_types[ticket_type] if ticket_type < len(ticket_types) else 'Unknown'}")
            with tcol2:
                priority_name = priorities[ticket_priority] if ticket_priority < len(priorities) else 'Unknown'
                if ticket_priority >= 2:
                    st.error(f"**Priority:** {priority_name}")
                else:
                    st.warning(f"**Priority:** {priority_name}")
        else:
            st.info("No ticket in queue")
        
        # Agent workload visualization
        st.markdown("#### 👥 Agent Workload")
        if st.session_state.current_obs:
            agents_data = st.session_state.current_obs['agents']
            
            agent_df = pd.DataFrame({
                'Agent': [f"Agent {i}" for i in range(len(agents_data))],
                'Workload': [agents_data[i][1] for i in range(len(agents_data))],
                'Capacity': [agents_data[i][2] for i in range(len(agents_data))],
            })
            
            # Color-code based on capacity
            def color_workload(val):
                if val >= 0.8:
                    return 'background-color: #ff6b6b'
                elif val >= 0.6:
                    return 'background-color: #ffd93d'
                else:
                    return 'background-color: #6bcf7f'
            
            styled_df = agent_df.style.map(color_workload, subset=['Workload'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Progress bars for workload
            for i in range(len(agents_data)):
                workload_pct = float(agents_data[i][1])  # Convert numpy float32 to Python float
                st.progress(workload_pct, text=f"Agent {i}")

with col2:
    st.subheader("📈 Episode Progress")
    
    if st.session_state.episode_data:
        # Reward over time
        df = pd.DataFrame(st.session_state.episode_data)
        
        st.markdown("**Rewards**")
        st.line_chart(df.set_index('step')['reward'])
        
        st.markdown("**Queue Size**")
        st.line_chart(df.set_index('step')['queue_size'])
        
        st.markdown("**Available Agents**")
        st.line_chart(df.set_index('step')['available_agents'])
        
        # Summary statistics
        st.markdown("**Statistics**")
        st.metric("Mean Reward", f"{df['reward'].mean():.2f}")
        st.metric("Max Queue", int(df['queue_size'].max()))
        st.metric("Steps Taken", len(df))

# Episode history table
if st.session_state.episode_data:
    st.markdown("---")
    st.subheader("📋 Episode History")
    
    history_df = pd.DataFrame(st.session_state.episode_data)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "step": st.column_config.NumberColumn("Step", format="%d"),
            "action": st.column_config.NumberColumn("Agent Selected", format="Agent %d"),
            "reward": st.column_config.NumberColumn("Reward", format="%.2f"),
            "queue_size": st.column_config.NumberColumn("Queue Size", format="%d"),
            "available_agents": st.column_config.NumberColumn("Available Agents", format="%d"),
        }
    )
    
    # Download data
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Episode Data",
        data=csv,
        file_name="episode_data.csv",
        mime="text/csv",
    )

# Info section
with st.expander("ℹ️ About This Simulator"):
    st.markdown("""
    This simulator allows you to interact with the Ticket Routing Environment step-by-step:
    
    - **Initialize**: Start a new environment instance
    - **Select Agent**: Choose which agent should handle the current ticket
    - **Step**: Execute the action and observe the result
    - **Random Action**: Let the environment choose randomly
    - **Auto-play**: Run a full episode with random actions
    
    The environment simulates a support ticket routing system where you must assign
    incoming tickets to available agents based on their workload and expertise.
    """)
