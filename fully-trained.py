import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("InvertedPendulum-v5")
eval_env = gym.make("InvertedPendulum-v5")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
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

# agent evaluation
print("\n=== Evaluating Fully Trained Agent ===")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.1f} +/- {std_reward:.1f}")

# plot training curve from eval logs
eval_results = np.load("./logs/evaluations.npz")
timesteps = eval_results["timesteps"]
results = eval_results["results"]
mean_rewards = np.mean(results, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, linewidth=2, color="blue")
plt.fill_between(
    timesteps,
    mean_rewards - np.std(results, axis=1),
    mean_rewards + np.std(results, axis=1),
    alpha=0.2,
    color="blue",
)
plt.xlabel("Training Timesteps", fontsize=12)
plt.ylabel("Mean Episode Reward", fontsize=12)
plt.title("PPO Training Progress: Inverted Pendulum", fontsize=14)
plt.axhline(y=1000, color="green", linestyle="--", label="Maximum Reward (1000)")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print("Training curve saved as 'training_curve.png'")

eval_env.close()
env.close()

# render
render_env = gym.make("InvertedPendulum-v5", render_mode="human")
obs, info = render_env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = render_env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Rendered episode reward: {total_reward:.0f}")
render_env.close()