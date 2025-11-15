"""Smoke test to verify environment can be instantiated and run."""

from rl_environments.ticket_routing import TicketRoutingEnv

def main():
    print("Creating environment...")
    env = TicketRoutingEnv()
    
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    
    print("\nObservation keys:", obs.keys())
    print("Observation shapes:")
    for key, value in obs.items():
        print(f"  {key}: {value.shape}")
    
    print("\nInfo keys:", info.keys())
    print("Info values:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nRunning 10 steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print("Episode ended early!")
            break
    
    print("\nSmoke test passed! ✓")
    env.close()

if __name__ == "__main__":
    main()
