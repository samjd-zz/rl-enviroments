"""Analytics Dashboard - Performance metrics and visualizations."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from rl_environments.ticket_routing import TicketRoutingEnv
from rl_environments.ticket_routing.core.data_models import EnvironmentConfig

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Analytics Dashboard")
st.markdown("Performance metrics and analysis for the Ticket Routing Environment")

# Sidebar configuration
st.sidebar.header("Analysis Settings")

num_episodes = st.sidebar.slider("Number of Episodes", 1, 50, 10)
episode_length = st.sidebar.slider("Episode Length", 50, 500, 100)
num_agents = st.sidebar.slider("Number of Agents", 2, 10, 5)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Initialize session state for results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if run_analysis:
    with st.spinner("Running environment analysis..."):
        # Create environment
        config = EnvironmentConfig(
            num_agents=num_agents,
            episode_length=episode_length,
        )
        env = TicketRoutingEnv(config=config)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        all_step_data = []
        
        for episode in range(num_episodes):
            obs, info = env.reset(seed=seed + episode)
            episode_reward = 0.0
            step_count = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()  # Random policy
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                all_step_data.append({
                    'episode': episode,
                    'step': step_count,
                    'reward': reward,
                    'queue_size': info['queue_size'],
                    'tickets_processed': info['tickets_processed'],
                    'available_agents': info['available_agents'],
                    'agent_workload_mean': info['agent_workload_mean'],
                    'agent_workload_std': info['agent_workload_std'],
                })
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
        
        # Store results
        st.session_state.analysis_results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'step_data': pd.DataFrame(all_step_data),
            'config': {
                'num_episodes': num_episodes,
                'episode_length': episode_length,
                'num_agents': num_agents,
                'seed': seed,
            }
        }
    
    st.success(f"✅ Analysis complete! Ran {num_episodes} episodes.")

# Display results if available
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    
    # Summary metrics
    st.header("📈 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_reward = np.mean(results['episode_rewards'])
        st.metric("Mean Episode Reward", f"{mean_reward:.2f}")
    
    with col2:
        std_reward = np.std(results['episode_rewards'])
        st.metric("Reward Std Dev", f"{std_reward:.2f}")
    
    with col3:
        mean_length = np.mean(results['episode_lengths'])
        st.metric("Mean Episode Length", f"{mean_length:.0f}")
    
    with col4:
        total_tickets = results['step_data']['tickets_processed'].max()
        st.metric("Total Tickets Processed", int(total_tickets))
    
    st.markdown("---")
    
    # Episode rewards
    st.header("🎯 Episode Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Episode Rewards")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results['episode_rewards'], marker='o', linewidth=2, markersize=6)
        ax.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Reward Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(results['episode_rewards'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Step-level analysis
    st.header("🔍 Step-Level Analysis")
    
    step_data = results['step_data']
    
    tab1, tab2, tab3, tab4 = st.tabs(["Rewards", "Queue Dynamics", "Agent Workload", "Efficiency"])
    
    with tab1:
        st.subheader("Reward Distribution Over Time")
        
        # Reward heatmap by episode
        pivot_data = step_data.pivot_table(
            values='reward',
            index='step',
            columns='episode',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_data.T, cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Reward'})
        ax.set_xlabel('Step')
        ax.set_ylabel('Episode')
        ax.set_title('Reward Heatmap Across Episodes')
        st.pyplot(fig)
        plt.close()
        
        # Average reward per step
        st.markdown("**Average Reward per Step (All Episodes)**")
        avg_reward_by_step = step_data.groupby('step')['reward'].mean()
        st.line_chart(avg_reward_by_step)
    
    with tab2:
        st.subheader("Queue Size Dynamics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average queue size over steps
            avg_queue = step_data.groupby('step')['queue_size'].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(avg_queue.index, avg_queue.values, linewidth=2)
            ax.fill_between(avg_queue.index, avg_queue.values, alpha=0.3)
            ax.set_xlabel('Step')
            ax.set_ylabel('Queue Size')
            ax.set_title('Average Queue Size Over Time')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Queue size distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(step_data['queue_size'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Queue Size')
            ax.set_ylabel('Frequency')
            ax.set_title('Queue Size Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Queue statistics
        st.markdown("**Queue Statistics**")
        queue_stats = step_data['queue_size'].describe()
        st.dataframe(queue_stats.to_frame().T, use_container_width=True)
    
    with tab3:
        st.subheader("Agent Workload Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Workload balance over time
            st.markdown("**Mean Workload**")
            avg_workload = step_data.groupby('step')['agent_workload_mean'].mean()
            st.line_chart(avg_workload)
        
        with col2:
            # Workload variance (balance)
            st.markdown("**Workload Standard Deviation (Balance)**")
            avg_std = step_data.groupby('step')['agent_workload_std'].mean()
            st.line_chart(avg_std)
        
        # Workload balance quality
        st.markdown("**Workload Balance Quality**")
        balance_score = 1 - step_data['agent_workload_std'].mean()
        st.metric(
            "Balance Score",
            f"{balance_score:.2%}",
            help="Higher is better. Calculated as 1 - mean(std_dev) of agent workloads"
        )
    
    with tab4:
        st.subheader("System Efficiency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tickets processed over time
            st.markdown("**Tickets Processed (Cumulative)**")
            for episode in step_data['episode'].unique()[:5]:  # Show first 5 episodes
                episode_data = step_data[step_data['episode'] == episode]
                st.line_chart(episode_data.set_index('step')['tickets_processed'])
        
        with col2:
            # Available agents over time
            st.markdown("**Available Agents**")
            avg_available = step_data.groupby('step')['available_agents'].mean()
            st.line_chart(avg_available)
        
        # Efficiency metrics
        st.markdown("**Efficiency Metrics**")
        efficiency_cols = st.columns(3)
        
        with efficiency_cols[0]:
            throughput = step_data['tickets_processed'].max() / episode_length
            st.metric("Throughput", f"{throughput:.2f} tickets/step")
        
        with efficiency_cols[1]:
            avg_utilization = 1 - (step_data['available_agents'].mean() / num_agents)
            st.metric("Agent Utilization", f"{avg_utilization:.1%}")
        
        with efficiency_cols[2]:
            avg_queue = step_data['queue_size'].mean()
            st.metric("Average Queue Size", f"{avg_queue:.1f}")
    
    st.markdown("---")
    
    # Detailed data table
    st.header("📋 Detailed Data")
    
    with st.expander("View Raw Data"):
        st.dataframe(
            step_data,
            use_container_width=True,
            hide_index=True,
        )
        
        # Download button
        csv = step_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="analytics_data.csv",
            mime="text/csv",
        )
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        st.dataframe(step_data.describe(), use_container_width=True)

else:
    st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to start")
    
    st.markdown("""
    ### About This Dashboard
    
    This analytics dashboard provides comprehensive insights into environment performance:
    
    - **Summary Metrics**: High-level performance indicators
    - **Episode Performance**: Track rewards and consistency across episodes
    - **Step-Level Analysis**: Deep dive into:
        - Reward patterns and distributions
        - Queue dynamics and efficiency
        - Agent workload balance
        - System utilization metrics
    
    **Random Policy Baseline**: The analysis uses a random action policy to establish
    baseline performance metrics. This helps identify the difficulty of the environment
    and provides a benchmark for trained agents.
    """)

# Info section
with st.expander("ℹ️ Interpreting the Metrics"):
    st.markdown("""
    **Episode Rewards**
    - Higher is better
    - Look for consistency (low standard deviation)
    - Random policy establishes baseline
    
    **Queue Size**
    - Lower average queue = better efficiency
    - High variance might indicate bottlenecks
    
    **Agent Workload**
    - Low std deviation = balanced workload
    - High balance score = efficient utilization
    
    **Throughput**
    - Tickets processed per step
    - Higher = more efficient system
    
    **Agent Utilization**
    - Percentage of agents actively working
    - Too high = risk of queue overflow
    - Too low = underutilized resources
    """)
