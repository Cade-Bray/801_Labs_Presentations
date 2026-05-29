from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
import gymnasium as gym
import ale_py
import time

TRAINING = False
MODEL_PATH = "dqn_mario"
TOTAL_TIMESTEPS = 1_000_000
# See all Atari Environments at https://ale.farama.org/environments/
gym.register_envs(ale_py)

def training():
    """
    Trains a DQN agent on the Atari Mario environment. The environment is created with 4 parallel instances to speed
    up training, and frames are stacked to give the model temporal context. The DQN model is configured with specific
    hyperparameters for learning, exploration, and memory optimization. After training for a total of 1 million
    timesteps, the model is saved to disk. Finally, the environment is closed to free up resources.
    :return:
    """
    env = make_atari_env("ALE/MarioBros-v5", n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=1)
    model = DQN(
        "CnnPolicy",          # Use a convolutional neural network policy suitable for image environments
        env,                        # The environment to train on which is the Atari mario game
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
    episode ends, it resets the environment and starts a new episode. After testing, the environment is closed to free
    up resources.
    :return: None
    """
    env = make_atari_env("ALE/MarioBros-v5", n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=1) # Stack 1 frame only since Mario is a fast paced gam, and we want the model to react to the most recent frame without needing temporal context from previous frames
    model = DQN.load(MODEL_PATH, env=env) # Load the trained model and set the environment for testing
    obs = env.reset() # Reset the environment to get the initial observation for testing
    for step in range(1_000):
        # Deterministic action because we want to see the best action according to the trained model
        # This means the agent is not exploring it is always exploiting the best action according to the model's predictions
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment and observe the results
        obs, reward, done, info = env.step(action)

        if done:
            print(f"Episode finished after {step + 1} steps with reward {reward}")
            time.sleep(2)  # Pause to allow the user to see the final state before resetting
            obs = env.reset()

    # After testing, close the environment to free up resources
    env.close()

if __name__ == "__main__":
    if TRAINING:
        training()
    else:
        testing()
