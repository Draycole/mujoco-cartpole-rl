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