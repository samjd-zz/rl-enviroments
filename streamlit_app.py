"""Streamlit UI for RL Environments - Main Application.

This is the entry point for the Streamlit web interface that provides
interactive visualization and control of RL environments.
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="RL Environments Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page content
st.title("🤖 Reinforcement Learning Environments Dashboard")

st.markdown("""
Welcome to the **RL Environments Interactive Dashboard**! This application provides
an intuitive interface to explore, visualize, and interact with custom reinforcement
learning environments.

### Available Features

""")

# Create three columns for feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🎮 Environment Simulator
    - Run interactive simulations
    - Step through episodes
    - Visualize agent decisions
    - Real-time metrics
    """)

with col2:
    st.markdown("""
    #### 📊 Analytics Dashboard
    - Performance metrics
    - Agent workload analysis
    - Queue statistics
    - Reward tracking
    """)

with col3:
    st.markdown("""
    #### ⚙️ Configuration
    - Customize environment
    - Agent parameters
    - Reward shaping
    - Export settings
    """)

st.markdown("---")

# Quick start guide
st.header("🚀 Quick Start")

st.markdown("""
1. **Navigate** using the sidebar to access different features
2. **Simulator** - Run step-by-step or full episode simulations
3. **Analytics** - View detailed performance metrics and visualizations
4. **Configuration** - Adjust environment parameters to your needs

### Current Environments

**Ticket Routing Environment** 🎫
- Simulates a B2B SaaS support ticket system
- Learn optimal ticket assignment strategies
- Multi-agent system with expertise matching
- Real-time workload balancing

""")

# System information
with st.expander("📋 System Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Framework:**
        - Gymnasium API Compatible
        - Python 3.8+
        - NumPy-based computation
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - Type-safe implementation
        - Comprehensive testing
        - Configurable parameters
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | RL Environments v1.0</p>
</div>
""", unsafe_allow_html=True)
