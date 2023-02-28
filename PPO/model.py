import torch

from stable_baselines3.common.policies import ActorCriticPolicy


class AgentPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim_pi = 100
        self.latent_dim_vf = 1

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

    def forward(self, features):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        states, actions = features['states'].float(), features['actions'].float()

        state_emdedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])

        embedding = torch.hstack((state_emdedding, action_pool))
        final_embedding = self.mlp(embedding)

        action_out = torch.matmul(action_embedding, final_embedding.reshape(-1, action_embedding.shape[-1], 1))[:, :, 0]

        value_out = self.mlp_value(embedding)

        return action_out, value_out


    def forward_actor(self, features):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        states, actions = features['states'].float(), features['actions'].float()

        state_emdedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])

        embedding = torch.hstack((state_emdedding, action_pool))
        final_embedding = self.mlp(embedding)

        action_out = torch.matmul(action_embedding, final_embedding.reshape(-1, action_embedding.shape[-1], 1))[:, :, 0]

        return action_out

    def forward_critic(self, features):
        states, actions = features['states'].float(), features['actions'].float()

        state_emdedding = self.state_encoder(states)
        action_embedding = self.action_encoder(actions)

        action_pool = self.action_pool(action_embedding).reshape(-1, action_embedding.shape[-1])

        embedding = torch.hstack((state_emdedding, action_pool))

        value_out = self.mlp_value(embedding)

        return value_out


class Agent(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AgentPolicy()

    def extract_features(self, obs):
        return obs
