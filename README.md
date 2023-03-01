# Ranked Reward Reinforcement Learning (Bin Packing Problem) 

This contains experimental code for the Bin Packing Problem with PPO and [ranked reward MCTS](https://arxiv.org/pdf/1807.01672.pdf) methods. Bin packing problem is a combinatorial optimization problem consisting of a set of bins of varied sizes to be filled into a container. There are multiple versions of the bin packing problem. This repository considers the offline problem where we have a fixed size container and all bins to be fit into the container are visible throughout. The goal is to fit the maximum number of bins possible into the container. Currently it implements a 2D problem (which can be extended) where the state space consists of the features of all items `{id, size and placement(if already placed)}` and a set of `feasible actions`. There are constraints on the actions as the items cannot be placed when not supported below or go beyond the container. The actions available are based on the left over items to be filled and valid placements in the container. The problem sets are generated using a method similar to that mentioned in the ranked reward paper. This considers a discrete action space. At each step, the agent selects an item and its placement in the container. The reward is +1 for fitting all the items in the container, -1 when no actions are feasible and there are items remaining to be filled and 0 otherwise. An episode either ends when all items are placed or no more items can be placed.

The policy and value network architecture is based on [`Ranked Reward`](https://arxiv.org/pdf/1807.01672.pdf). There are assumptions made about the architecture as the exact architecture is not mentioned in the paper. Currently there are two implementations:
1) using a PPO agent
2) using the ranked reward method 

Both the methods have a similar policy and value network architecture but the ranked reward method uses MCTS.

## Installation

```
git clone https://github.com/shiveshkhaitan/ranked_reward_rl
cd ranked_reward_rl
conda env create -f environment.yml
conda activate ranked_reward
```

#### Example placement using Ranked Reward

<p align="center">
<img src="https://user-images.githubusercontent.com/33219837/222007831-55a91995-7160-414b-88f9-47a7e460557f.png" width="750" height="500" />
</p>
