import ray
from ray.rllib.algorithms import alpha_zero
from ray.rllib.utils.test_utils import check_train_results
from ray.tune.registry import register_env

from env import BPP
from model import Agent

register_env('Bpp-v1', BPP)


def train():
    ray.init(num_gpus=1)

    mcts_config = {
                "puct_coefficient": 1.0,
                "num_simulations": 300,
                "temperature": 1.5,
                "dirichlet_epsilon": 0.25,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": True,
                "add_dirichlet_noise": True,
            }

    env_config = {'bin_size': [10, 10], 'max_bin_size': [10, 10], 'num_items': 10}

    ranked_rewards = {
        "enable": True,
        "percentile": 75,
        "buffer_max_length": 1000,
        "initialize_buffer": True,
        "num_init_rewards": 100,
    }

    config = (
                alpha_zero.AlphaZeroConfig()
                .environment(env='Bpp-v1', env_config=env_config)
                .training(model={"custom_model": Agent},
                          mcts_config=mcts_config,
                          num_sgd_iter=10,
                          ranked_rewards=ranked_rewards)
                .rollouts(num_rollout_workers=16)
             )

    num_iterations = 50

    algo = config.build()

    for i in range(num_iterations):
        print(f'Training iteration: {i + 1}')
        results = algo.train()

    path = algo.save()

    print(f'Model saved to {path}')
    algo.stop()

    ray.shutdown()


if __name__  == '__main__':
    train()