import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("InvertedPendulum-v5")

print("Training PPO for 10000 timesteps (exploring undertraining)")
model = PPO("MlpPolicy", env, verbose = 1, device="auto")
model.learn(total_timesteps=10_000)

model.save("ppo_pendulum_undertrained")
print("Model saved as 'ppo_pendulum_undertrained.zip")

print("\n--- Evaluating the Undertrained Agent ---")
eval_env = gym.make("InvertedPendulum-v5")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.1f} +/- {std_reward:.1f}")
print(f"The agent is learning but not god enough yet")

eval_env.close()
env.close()

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