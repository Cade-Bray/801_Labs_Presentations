import time
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step: {step}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        print("Episode finished!")
        time.sleep(1)
        obs, info = env.reset()

env.close()