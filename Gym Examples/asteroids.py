from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gymnasium as gym
import ale_py
import time

TRAINING = False
MODEL_PATH = "dqn_asteroids"
TOTAL_TIMESTEPS = 1_000_000
# See all Atari Environments at https://ale.farama.org/environments/
gym.register_envs(ale_py)

def training():
    """
    Trains a DQN agent on the Atari Asteroids environment. The environment is created with 4 parallel instances to speed
    up training, and frames are stacked to give the model temporal context. The DQN model is configured with specific
    hyperparameters for learning, exploration, and memory optimization. After training for a total of 1 million
    timesteps, the model is saved to disk. Finally, the environment is closed to free up resources.
    :return:
    """
    env = make_atari_env("ALE/Asteroids-v5", n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model = DQN(
        "CnnPolicy",          # Use a convolutional neural network policy suitable for image environments
        env,                        # The environment to train on which is the Atari Asteroids game
        verbose=1,                  # Print training progress and information
        buffer_size=100_000,        # Size of the replay buffer to store past experiences for training
        learning_starts=10_000,     # Num of steps to collect before starting training to ensure a good variety of exp
        batch_size=32,              # Number of samples to use for each training update
        gamma=0.99,                 # Discount factor for future rewards, close to 1 means future rewards are important
        exploration_fraction=0.1,   # Frac of total training to reduce exploration rate from initial to final value
        exploration_final_eps=0.01, # Final value of exploration rate after decay, meaning 1% of the time the agent will take random actions
        optimize_memory_usage=True, # Optimize memory usage by storing only the most recent frames in the replay buffer
        replay_buffer_kwargs={
            "handle_timeout_termination": False  # Needs to be false if optimize_memory_usage is true to avoid bug in SB3
        }
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # After training, close the environment to free up resources
    env.close()

def testing():
    """
    Loads the trained DQN model and runs it in the environment to see how it performs. The agent will take the best
    action according to the model's predictions at each step, without any exploration. The environment is rendered in
    human mode so you can visually observe the agent's performance. The loop runs for a fixed number of steps, and if an
    episode ends, it resets the environment and starts a new episode. After testing, the
    environment is closed to free up resources.
    :return: None
    """
    # Simply use SB3's native factory, but turn rendering on.
    env = make_atari_env(
        "ALE/Asteroids-v5",
        n_envs=1,
        seed=0,
        env_kwargs={"render_mode": "human"},
    )

    # Needs to match the frame stacking of the training environment
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(MODEL_PATH, env=env)

    obs = env.reset()
    for step in range(1_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            print(f"Episode finished after {step + 1} steps with reward {reward[0]}")
            time.sleep(2)
            obs = env.reset()

        # Add a tiny visual sleep to slow down the 4-frame execution speed so it looks more natural
        time.sleep(0.015)

    env.close()

if __name__ == "__main__":
    if TRAINING:
        training()
    else:
        # Trained on every 4 frames and is replayed on every 4 frames. This means that it will appear to be choppy.
        testing()
