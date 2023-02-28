import argparse

import gym
from gym.envs.registration import register
from stable_baselines3 import PPO

from env import BPP
from model import Agent


def test():
    parser = argparse.ArgumentParser(description='Testing script for ranked reward on bin packing problem')
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path for trained policy', required=True)
    args = parser.parse_args()

    register(
        id='Bpp-v1',
        entry_point='env:BPP',
    )

    # Create and wrap the environment
    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}
    env = gym.make('Bpp-v1', env_config=env_config)

    model = PPO(Agent, env, verbose=1)
    model = PPO.load(args.checkpoint_path, env)

    while True:
        done = False

        obs = env.reset()

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()


if __name__ == '__main__':
    test()