import os
import numpy as np
from torch import manual_seed
from mushroom_rl_extensions.environments.mujoco_envs.three_linked_reacher import (
    ThreeLinkedReacher,
)
from mushroom_rl.core import Core
from mushroom_rl.core.serialization import Serializable
from mushroom_rl.utils.dataset import compute_J
import numpy as np

# Script to render your trained agent in the environment


def main():
    # Seed
    np.random.seed(0)
    manual_seed(0)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Agent of your choice
    agent_path = "<your agent path>"
    agent = Serializable.load(agent_path)

    core = Core(agent, mdp)
    dataset = core.evaluate(n_steps=1000, render=True)
    J = compute_J(dataset, mdp.info.gamma)
    R = compute_J(dataset)
    print(J)
    print(R)


if __name__ == "__main__":
    main()
