import torch
import gym
from gym.envs.registration import register

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

from env import BPP
from model import Agent


def train():
    register(
        id='Bpp-v1',
        entry_point='env:BPP',
    )

    # Create and wrap the environment
    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}
    env = gym.make('Bpp-v1', env_config=env_config)

    num_envs = 16

    env = SubprocVecEnv([lambda: env for _ in range(num_envs)], start_method="spawn")

    model = PPO(Agent, env, verbose=1, n_epochs = 5, n_steps = 8192 // num_envs)

    # Train the agent
    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        model.save("bpp")


if __name__ == '__main__':
    train()