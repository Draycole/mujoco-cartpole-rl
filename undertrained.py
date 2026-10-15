import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("InvertedPendulum-v5")

print("Training PPO for 10000 timesteps (exploring undertraining)")
model = PPO("MlpPolicy", env, verbose = 1, device="auto")
model.learn(total_timesteps=10_000)

model.save("ppo_pendulum_undertrained")
print("Model saved as 'ppo_pendulum_undertrained.zip")