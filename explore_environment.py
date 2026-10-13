import gymnasium as gym
import numpy as np

env = gym.make("InvertedPendulum-v5", render_mode="human")

print("-- Inverted Pendulum Environment --")
print(f"Observation Space: {env.observation_space}")
print(f" Shape: {env.observation_space.shape}")
print(f"  Observations: [cart_position, pole_angle, cart_velocity, pole_angular_velocity]")
print(f"\nAction Space: {env.action_space}")
print(f"  Shape: {env.action_space.shape}")
print(f"  Range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
print(f"  Action: force applied to cart in Newtons") 

num_episodes = 10
episode_rewards = []

for episode in range (num_episodes):
    observation, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample() #random action
        observation, reward, terminated, truncated, info = env.step(action)
        total reward += reward
        done = terminated or truncated
    
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.0f}")

env.close()

print(f"\n=== Random Agent Results ===")
print(f"Mean Reward: {np.mean(episode_rewards):.1f} / 1000")
print(f"The pole falls almost immediately without a learned policy!")