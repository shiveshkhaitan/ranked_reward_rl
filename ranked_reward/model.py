from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel

from gym import spaces
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple, Type, Union


class Agent(ActorCriticModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs,
        model_config,
        name,
    ):
        super().__init__(
            observation_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        self.state_encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(50, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            )

        self.action_encoder = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 128),
            )

        self.action_pool = torch.nn.AdaptiveMaxPool2d((1, 128))

        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            )

        self.mlp_value = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )

    def forward(self, input_dict, state, seq_lens):
        states, actions = input_dict['obs']['states'].float(), input_dict['obs']['actions'].float()

        state_emdedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])

        embedding = torch.hstack((state_emdedding, action_pool))
        final_embedding = self.mlp(embedding)

        action_out = torch.matmul(action_embedding, final_embedding.reshape(-1, action_embedding.shape[-1], 1))[:, :, 0]

        self._value_out = self.mlp_value(embedding)

        return action_out, None
