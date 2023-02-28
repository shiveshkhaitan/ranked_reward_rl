import copy

import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from typing import List, Tuple, Union


class BPP(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self.bin_size = env_config['bin_size']
        self.max_bin_size = env_config['max_bin_size']
        self.dim = len(self.bin_size)
        self.num_items = env_config['num_items']

        self.items = dict()
        self.bin = np.zeros(self.max_bin_size)

        self.observation_space = spaces.Dict(
            {
                'obs': spaces.Dict(
                        {
                            'states': spaces.Box(-1, self.max_bin_size[0],
                                                 shape=(self.num_items, 2 * self.dim + 1)),
                            'actions': spaces.Box(-1, self.max_bin_size[0],
                                                  shape=(self.num_items * self.max_bin_size[0], 2)),
                        }
                    ),

                'action_mask': spaces.Box(low=0, high=1, shape=(self.num_items * self.max_bin_size[0], )),
            }
        )

        self.action_space = spaces.Discrete(self.num_items * self.max_bin_size[0])
        self.running_reward = 0

        # Workaround for max_episode_steps
        class Spec:
            def __init__(self, max_episode_steps):
                self.id = getattr(env_config, 'worker_index', 0)
                self.max_episode_steps = max_episode_steps

        self.spec = Spec(max_episode_steps=self.num_items)

        # Colors for rendering
        self.colors = np.random.rand(self.num_items, 3)

    def _get_obs(self):

        obs = dict()
        obs['states'] = np.array(self.items)

        items = np.arange(1, self.num_items + 1)
        placements = np.arange(0, self.max_bin_size[0])

        actions = np.vstack((np.repeat(items, self.max_bin_size[0]), np.tile(placements, self.num_items))).T
        for action in actions:
            already_placed = self.items[action[0] - 1][-1] != -1
            if already_placed:
                action[:] = [-1, -1]
            else:
                size = self.items[action[0] - 1][1 : 3]

                block = self.bin[action[1] : action[1] + size[0], :]
                y = np.where(block.sum(0) == 0)[0]
                if len(y) == 0:
                    action[:] = [-1, -1]
                else:
                    anchor = np.array([action[1], y[0]])
                    block = self.bin[anchor[0] : anchor[0] + size[0], anchor[1] : anchor[1] + size[1]]

                    # Out of bounds
                    if anchor[0] + size[0] > self.max_bin_size[0] or anchor[1] + size[1] > self.max_bin_size[1]:
                        action[:] = [-1, -1]

                    # Overlapping
                    elif np.any(block):
                        action[:] = [-1, -1]

                    # Not supported
                    elif anchor[1] > 0:
                        left_supported = np.any(self.bin[anchor[0] : anchor[0] + int(np.ceil(size[0] / 2)),
                                                anchor[1] - 1])
                        right_supported = np.any(self.bin[anchor[0] + int(np.floor(size[0] / 2)) : anchor[0] + size[0],
                                                 anchor[1] - 1])

                        if not (left_supported and right_supported):
                            action[:] = [-1, -1]

        obs['actions'] = actions
        self.actions = actions

        action_mask = np.ones(len(actions))
        action_mask[np.where(actions[:, 0] == -1)[0]] = 0
        self.action_mask = action_mask

        observation = dict(obs=obs, action_mask=action_mask)

        return observation

    def reset(self):
        self.items = []
        items = [[0] * self.dim + self.bin_size + [(self.bin_size[0] * self.bin_size[1])]]

        while len(items) < self.num_items:
            weights = [item[4] * 100 for item in items]
            weight_sum = sum(weights)
            weights = [item / weight_sum for item in weights]

            random_item = items.pop(np.random.choice(len(items), p=weights))
            random_axis = np.random.choice(self.dim)

            while random_item[random_axis] + 1 == random_item[random_axis + self.dim]:
                items.append(random_item)
                random_item = items.pop(np.random.choice(len(items)))
                random_axis = np.random.choice(self.dim)

            random_pos = np.random.choice(np.arange(random_item[random_axis] + 1,
                                                    random_item[random_axis + self.dim]))

            item1 = copy.deepcopy(random_item)
            item1[random_axis + self.dim] = random_pos
            item1[4] = (item1[2] - item1[0]) * (item1[3] - item1[1])

            item2 = copy.deepcopy(random_item)
            item2[random_axis] = random_pos
            item2[4] = (item2[2] - item2[0]) * (item2[3] - item2[1])

            items.extend([item1, item2])

        self.initial_setting = items

        for i, item in enumerate(items):
            self.items.append([i + 1, item[2] - item[0], item[3] - item[1], -1, -1]) 

        self.bin = np.zeros(self.max_bin_size)

        self.num_placed = 0

        return self._get_obs()

    def step(self, action):
        action += 1
        action = self.actions[(action * self.action_mask).argmax()]

        size = self.items[action[0] - 1][1:3]

        block = self.bin[action[1] : action[1] + size[0], :]
        anchor = np.array([action[1], np.where(block.sum(0) == 0)[0][0]])        

        block = self.bin[anchor[0] : anchor[0] + size[0], anchor[1] : anchor[1] + size[1]]

        block[:, :] = 1

        self.items[action[0] - 1][3:5] = anchor

        self.num_placed += 1

        if self.num_placed == 10:
            self.running_reward += 1
            return self._get_obs(), 1, True, {}

        obs = self._get_obs()

        # No more boxes can be accomodated
        if self.action_mask.sum() == 0:
            self.running_reward += -1
            return obs, -1, True, {}

        return obs, 0, False, {}

    def render(self, mode='human'):
        if self.dim != 2:
            raise NotImplementedError('Rendering only supported for 2D bins')

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.set_xlim(0, self.max_bin_size[0])
        ax1.set_ylim(0, self.max_bin_size[1])
        ax1.set_title('Original placement')
        ax1.set_aspect('equal', 'box')

        ax2.set_xlim(0, self.max_bin_size[0])
        ax2.set_ylim(0, self.max_bin_size[1])
        ax2.set_title('Achieved placement')
        ax2.set_aspect('equal', 'box')

        for i, item in enumerate(self.initial_setting):
            ax1.add_patch(Rectangle((item[0], item[1]), item[2] - item[0], item[3] - item[1],
                          color=self.colors[i]))

        for i, item in enumerate(self.items):
            if item[3] != -1:
                ax2.add_patch(Rectangle((item[3], item[4]), item[1], item[2], color=self.colors[i]))

        plt.show()

    def set_state(self, state):
        self.items = copy.deepcopy(state[0][0])
        self.bin = copy.deepcopy(state[0][1])
        self.num_placed = state[0][2]
        self.running_reward = state[1]
        return self._get_obs()

    def get_state(self):
        return copy.deepcopy([self.items, self.bin, self.num_placed]), self.running_reward
