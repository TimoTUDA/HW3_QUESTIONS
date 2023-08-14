import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from mushroom_rl_extensions.environments.mujoco_envs.three_linked_reacher import (
    ThreeLinkedReacher,
)
from mushroom_rl.core import Core
from mushroom_rl.core.serialization import Serializable
from mushroom_rl.utils.dataset import compute_J
from torch import manual_seed


def predict_agent_state_value(theta_0, theta_1, agent):
    # TODO: Create a test state for the agent
    # We will constrain the third link to be held fixed at an angle of 0 and the goal position should be fixed at (-0.2, -0.2).
    # The test state must be changed accordingly in the following ways:
    # 1) The sin & cos values of the three joint angles must be modified in the test state
    # 2) The target x and y coordinates must be modified in the test state
    # Keep the joint velocities at 0.
    # Use the value network of a trained PPO agent to estimate the value of the test state

    test_state = np.zeros(11)
    # [YOUR CODE: MODIFY TEST STATE]

    v = #[YOUR CODE]
    return v


def main():
    # Seed
    np.random.seed(0)
    manual_seed(0)

    # MDP
    horizon = 250
    gamma = 0.99
    mdp = ThreeLinkedReacher(horizon, gamma)

    # Load agent
    agent_path = "<your agent path>"
    agent = Serializable.load(agent_path)

    # Extracting (action) value function
    theta_0 = np.linspace(-np.pi, np.pi, 100)
    theta_1 = np.linspace(-np.pi, np.pi, 100)
    theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0, theta_1)
    values = np.zeros_like(theta_0_mesh)

    for i in range(len(theta_0)):
        for j in range(len(theta_1)):
            values[i, j] = predict_agent_state_value(theta_0[i], theta_1[j], agent)

    # Create heatmap
    fig, ax = plt.subplots()
    ax.pcolormesh(theta_0_mesh, theta_1_mesh, values, cmap="jet")
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet), label="Action value"
    )
    fig.suptitle(f"Value HeatMap for target at (-0.2,-0.2): {type(agent).__name__}")
    ax.set_ylabel("theta_1")
    ax.set_xlabel("theta_0")

    # Save plot
    plot_dir = "<your path>/plots"
    fn = plot_dir + "/action_value_heatmap.jpg"
    plt.savefig(fn)


if __name__ == "__main__":
    main()
