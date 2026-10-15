import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

model_undertrained = PPO.load("ppo_pendulum_undertrained")
model_trained = PPO.load("./best_model/best_model")

eval_env = Monitor(gym.make("InvertedPendulum-v5"))

random_rewards = []
for _ in range(10):
    obs, info = eval_env.reset()
    total_reward = 0
    done = False
    while not done:
        action = eval_env.action_space.sample()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    random_rewards.append(total_reward)

undertrained_reward, _ = evaluate_policy(model_undertrained, eval_env, n_eval_episodes=10)

trained_reward, _ = evaluate_policy(model_trained, eval_env, n_eval_episodes=10)

eval_env.close()

# chart
labels = ["Random Agent", "Undertrained\n(10K steps)", "Fully Trained\n(100K steps)"]
rewards = [np.mean(random_rewards), undertrained_reward, trained_reward]
colors = ["#e74c3c", "#f39c12", "#27ae60"]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rewards, color=colors, edgecolor="black", linewidth=0.8)
plt.ylabel("Mean Episode Reward", fontsize=12)
plt.title("RL Agent Performance Comparison: Inverted Pendulum", fontsize=14)
plt.axhline(y=1000, color="green", linestyle="--", alpha=0.5, label="Maximum (1000)")
plt.legend(fontsize=11)
plt.ylim(0, 1100)

for bar, reward in zip(bars, rewards):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 20,
        f"{reward:.0f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=150)
plt.show()
print("Comparison chart saved as 'performance_comparison.png'")