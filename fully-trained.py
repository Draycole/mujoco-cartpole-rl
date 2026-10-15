import gymnasium as gym
import numpy as np
import matplotlib as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("InvertedPendulum-v5")
eval_env = gym.make("InvertedPendulum-v5")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/"
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True,
)

print("Training PPO for 100,000 timesteps...")
model = PPO(
    "MlpPolicy",
    env,
    verbose = 1,
    device = "auto",
    learning_rate = 3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./tensorboard_log/",
)

model.learn(total_timesteps=100_000, callback=eval_callback)

model.save("ppo_pendulum_trained")
print("Model saved as 'ppo_pendulum_trained.zip'")