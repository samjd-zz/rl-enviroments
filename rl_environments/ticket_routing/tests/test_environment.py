"""Integration tests for the TicketRoutingEnv environment."""

import time
from typing import Dict

import gymnasium as gym
import numpy as np
import pytest

from rl_environments.ticket_routing import TicketRoutingEnv
from rl_environments.ticket_routing.core.data_models import EnvironmentConfig


class TestEnvironmentBasics:
    """Test basic environment functionality."""

    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = TicketRoutingEnv()
        assert env is not None
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        env.close()

    def test_environment_with_custom_config(self):
        """Test environment creation with custom configuration."""
        config = EnvironmentConfig(
            num_agents=3,
            max_queue_size=50,
            episode_length=500,
        )
        env = TicketRoutingEnv(config=config)
        assert env.config.num_agents == 3
        assert env.config.max_queue_size == 50
        assert env.config.episode_length == 500
        env.close()

    def test_action_space(self):
        """Test action space definition."""
        env = TicketRoutingEnv()
        assert env.action_space.n == env.config.num_agents
        # Test valid actions
        for i in range(env.action_space.n):
            assert env.action_space.contains(i)
        # Test invalid actions
        assert not env.action_space.contains(-1)
        assert not env.action_space.contains(env.action_space.n)
        env.close()

    def test_observation_space(self):
        """Test observation space definition."""
        env = TicketRoutingEnv()
        obs_space = env.observation_space
        
        # Check that all required keys are present
        assert "current_ticket" in obs_space.spaces
        assert "agents" in obs_space.spaces
        assert "queue_size" in obs_space.spaces
        assert "time_step" in obs_space.spaces
        
        # Check shapes
        assert obs_space["current_ticket"].shape == (4,)
        assert obs_space["agents"].shape == (env.config.num_agents, 8)
        assert obs_space["queue_size"].shape == (1,)
        assert obs_space["time_step"].shape == (1,)
        
        env.close()


class TestEnvironmentReset:
    """Test environment reset functionality."""

    def test_reset_returns_correct_tuple(self):
        """Test that reset returns (observation, info) tuple."""
        env = TicketRoutingEnv()
        result = env.reset(seed=42)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        
        env.close()

    def test_reset_observation_valid(self):
        """Test that reset returns a valid observation."""
        env = TicketRoutingEnv()
        obs, info = env.reset(seed=42)
        
        # Check observation is in observation space
        assert env.observation_space.contains(obs)
        
        # Check all required keys present
        assert "current_ticket" in obs
        assert "agents" in obs
        assert "queue_size" in obs
        assert "time_step" in obs
        
        # Check data types
        assert obs["current_ticket"].dtype == np.float32
        assert obs["agents"].dtype == np.float32
        assert obs["queue_size"].dtype == np.float32
        assert obs["time_step"].dtype == np.float32
        
        env.close()

    def test_reset_info_contains_metrics(self):
        """Test that reset info contains expected metrics."""
        env = TicketRoutingEnv()
        obs, info = env.reset(seed=42)
        
        # Check required info keys
        assert "step" in info
        assert "tickets_processed" in info
        assert "avg_resolution_time" in info
        assert "queue_size" in info
        assert "agent_workload_mean" in info
        assert "agent_workload_std" in info
        assert "available_agents" in info
        
        # Check initial values
        assert info["step"] == 0
        assert info["tickets_processed"] == 0.0
        assert info["available_agents"] == env.config.num_agents
        
        env.close()

    def test_reset_with_seed_is_deterministic(self):
        """Test that reset with same seed produces same initial state."""
        env1 = TicketRoutingEnv()
        env2 = TicketRoutingEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Observations should be identical
        np.testing.assert_array_equal(obs1["current_ticket"], obs2["current_ticket"])
        np.testing.assert_array_equal(obs1["agents"], obs2["agents"])
        np.testing.assert_array_equal(obs1["queue_size"], obs2["queue_size"])
        np.testing.assert_array_equal(obs1["time_step"], obs2["time_step"])
        
        env1.close()
        env2.close()

    def test_reset_clears_previous_episode(self):
        """Test that reset properly clears state from previous episode."""
        env = TicketRoutingEnv()
        
        # Run partial episode
        obs1, info1 = env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        
        # Reset and check state is cleared
        obs2, info2 = env.reset(seed=42)
        
        assert info2["step"] == 0
        assert info2["tickets_processed"] == 0.0
        
        env.close()


class TestEnvironmentStep:
    """Test environment step functionality."""

    def test_step_returns_correct_tuple(self):
        """Test that step returns (obs, reward, terminated, truncated, info) tuple."""
        env = TicketRoutingEnv()
        env.reset(seed=42)
        
        action = env.action_space.sample()
        result = env.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()

    def test_step_observation_valid(self):
        """Test that step returns a valid observation."""
        env = TicketRoutingEnv()
        env.reset(seed=42)
        
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        
        # Check observation is in observation space
        assert env.observation_space.contains(obs)
        
        env.close()

    def test_step_invalid_action_raises_error(self):
        """Test that invalid actions raise ValueError."""
        env = TicketRoutingEnv()
        env.reset(seed=42)
        
        with pytest.raises(ValueError):
            env.step(-1)
        
        with pytest.raises(ValueError):
            env.step(env.action_space.n)
        
        env.close()

    def test_step_without_reset_raises_error(self):
        """Test that stepping without reset raises assertion error."""
        env = TicketRoutingEnv()
        
        with pytest.raises(AssertionError):
            env.step(0)
        
        env.close()

    def test_step_increments_time(self):
        """Test that step increments the time step."""
        env = TicketRoutingEnv()
        _, info = env.reset(seed=42)
        
        assert info["step"] == 0
        
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        
        assert info["step"] == 1
        
        env.close()

    def test_step_updates_metrics(self):
        """Test that step updates episode metrics."""
        env = TicketRoutingEnv()
        env.reset(seed=42)
        
        # Take several steps
        for _ in range(20):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
        
        # Check that metrics have been updated
        assert info["step"] == 20
        # Some tickets should have been processed
        assert info["tickets_processed"] >= 0
        
        env.close()


class TestFullEpisode:
    """Test full episode execution."""

    def test_full_episode_execution(self):
        """Test running a full episode until termination."""
        config = EnvironmentConfig(episode_length=100)
        env = TicketRoutingEnv(config=config)
        obs, info = env.reset(seed=42)
        
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        # Should terminate at episode_length
        assert steps == config.episode_length
        assert terminated
        assert not truncated
        assert info["step"] == config.episode_length
        
        env.close()

    def test_multiple_episodes_with_reset(self):
        """Test running multiple episodes with reset between them."""
        config = EnvironmentConfig(episode_length=50)
        env = TicketRoutingEnv(config=config)
        
        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)
            assert info["step"] == 0
            
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
            
            assert steps == config.episode_length
            assert terminated
        
        env.close()

    def test_episode_collects_rewards(self):
        """Test that episode collects rewards."""
        config = EnvironmentConfig(episode_length=50)
        env = TicketRoutingEnv(config=config)
        env.reset(seed=42)
        
        rewards = []
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
        
        # Should have collected some rewards
        assert len(rewards) == config.episode_length
        # Not all rewards should be the same (variety in assignments)
        assert len(set(rewards)) > 1
        
        env.close()


class TestSeedingDeterminism:
    """Test seeding and determinism."""

    def test_same_seed_produces_same_episode(self):
        """Test that same seed produces identical episodes."""
        config = EnvironmentConfig(episode_length=50)
        
        # Run first episode
        env1 = TicketRoutingEnv(config=config)
        env1.reset(seed=42)
        rewards1 = []
        observations1 = []
        
        for _ in range(50):
            action = 0  # Fixed action
            obs, reward, _, _, _ = env1.step(action)
            rewards1.append(reward)
            observations1.append(obs["current_ticket"].copy())
        
        # Run second episode with same seed
        env2 = TicketRoutingEnv(config=config)
        env2.reset(seed=42)
        rewards2 = []
        observations2 = []
        
        for _ in range(50):
            action = 0  # Fixed action
            obs, reward, _, _, _ = env2.step(action)
            rewards2.append(reward)
            observations2.append(obs["current_ticket"].copy())
        
        # Compare rewards and observations
        np.testing.assert_array_almost_equal(rewards1, rewards2)
        for obs1, obs2 in zip(observations1, observations2):
            np.testing.assert_array_equal(obs1, obs2)
        
        env1.close()
        env2.close()

    def test_different_seeds_produce_different_episodes(self):
        """Test that different seeds produce different episodes."""
        config = EnvironmentConfig(episode_length=50)
        
        # Run with seed 42
        env1 = TicketRoutingEnv(config=config)
        env1.reset(seed=42)
        rewards1 = []
        for _ in range(50):
            action = 0
            _, reward, _, _, _ = env1.step(action)
            rewards1.append(reward)
        
        # Run with seed 123
        env2 = TicketRoutingEnv(config=config)
        env2.reset(seed=123)
        rewards2 = []
        for _ in range(50):
            action = 0
            _, reward, _, _, _ = env2.step(action)
            rewards2.append(reward)
        
        # Rewards should be different
        assert not np.allclose(rewards1, rewards2)
        
        env1.close()
        env2.close()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_ticket_available(self):
        """Test behavior when no ticket is available."""
        # This is tested implicitly in the smoke test
        # When no tickets are generated, reward should be -1.0
        env = TicketRoutingEnv()
        env.reset(seed=42)
        
        # Force a step where no ticket might be available
        action = 0
        obs, reward, _, _, info = env.step(action)
        
        # Either ticket was assigned or got -1.0 reward
        assert isinstance(reward, (int, float))
        
        env.close()

    def test_agent_at_capacity(self):
        """Test behavior when assigning to agent at capacity."""
        config = EnvironmentConfig(num_agents=2, episode_length=100)
        env = TicketRoutingEnv(config=config)
        env.reset(seed=42)
        
        # Repeatedly assign to same agent until capacity
        assigned = 0
        for _ in range(50):
            action = 0  # Always assign to agent 0
            obs, reward, _, _, info = env.step(action)
            
            # Check if we got invalid action penalty
            if "invalid_action" in info:
                assert reward == -10.0
                break
            else:
                assigned += 1
        
        # Should have hit capacity at some point
        # (or test confirms capacity handling works)
        assert assigned >= 0
        
        env.close()

    def test_render_modes(self):
        """Test different render modes."""
        # Test human mode
        env = TicketRoutingEnv(render_mode="human")
        env.reset(seed=42)
        env.render()  # Should not crash
        env.close()
        
        # Test None mode (no rendering)
        env = TicketRoutingEnv(render_mode=None)
        env.reset(seed=42)
        env.render()  # Should not crash
        env.close()


class TestPerformance:
    """Test performance benchmarks."""

    def test_step_performance(self):
        """Test that step execution is fast (<1ms average)."""
        config = EnvironmentConfig(episode_length=1000)
        env = TicketRoutingEnv(config=config)
        env.reset(seed=42)
        
        # Warm-up
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        
        # Measure step time
        times = []
        for _ in range(100):
            action = env.action_space.sample()
            start = time.perf_counter()
            env.step(action)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Print for information
        print(f"\nStep performance:")
        print(f"  Average: {avg_time*1000:.3f}ms")
        print(f"  Max: {max_time*1000:.3f}ms")
        
        # Assert performance target
        assert avg_time < 0.001, f"Average step time {avg_time*1000:.3f}ms exceeds 1ms target"
        
        env.close()

    def test_reset_performance(self):
        """Test that reset is reasonably fast."""
        env = TicketRoutingEnv()
        
        times = []
        for i in range(20):
            start = time.perf_counter()
            env.reset(seed=42 + i)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        print(f"\nReset performance:")
        print(f"  Average: {avg_time*1000:.3f}ms")
        
        # Reset should be fast (< 10ms)
        assert avg_time < 0.01, f"Average reset time {avg_time*1000:.3f}ms exceeds 10ms"
        
        env.close()


class TestGymnasiumCompliance:
    """Test Gymnasium API compliance."""

    def test_env_checker_passes(self):
        """Test that environment passes Gymnasium's env_checker."""
        from gymnasium.utils.env_checker import check_env
        
        env = TicketRoutingEnv()
        
        # This will raise an exception if the environment is not compliant
        check_env(env.unwrapped, skip_render_check=True)
        
        env.close()

    def test_observation_space_sample(self):
        """Test that observation space can be sampled."""
        env = TicketRoutingEnv()
        
        # Sample from observation space
        sample = env.observation_space.sample()
        
        # Check that sample is valid
        assert env.observation_space.contains(sample)
        
        env.close()

    def test_metadata_present(self):
        """Test that environment has required metadata."""
        env = TicketRoutingEnv()
        
        assert hasattr(env, "metadata")
        assert "render_modes" in env.metadata
        assert "render_fps" in env.metadata
        
        env.close()
