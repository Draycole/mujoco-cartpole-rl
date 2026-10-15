from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("./best_model/best_model")

env = gym.make(
    "InvertedPendulum-v5",
    render_mode="human",
    max_episode_steps=10000
)

obs, info = env.reset()

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            obs, info = env.reset()

except KeyboardInterrupt:
    pass
finally:
    env.close()