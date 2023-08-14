import numpy as np
from pathlib import Path
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box


class ThreeLinkedReacher(MuJoCo):
    """
    Mujoco simulation of Three-Linked Reacher environment
    """

    def __init__(self, horizon=250, gamma=0.99):
        """
        Contructor of Three-Linked Reacher environment.
        See https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/environments/mujoco_envs/ball_in_a_cup.py
        for reference.

        Args:
            gamma (float): discount factor of the MDP.
            horizon (int): the horizon.
        """

        # TODO: add xml_path & action_spec according to the provided three_linked_reacher.xml
        # [YOUR CODE]
        xml_path = "C:/Users/Timo/Desktop/HW3_QUESTIONS/mushroom_rl_extensions/environments/mujoco_envs/data/three_linked_reacher.xml"
        action_spec = (3, Box(-1, 1, (3,)))

        observation_spec = [
            ("joint0", "joint0", ObservationType.JOINT_POS),
            ("joint1", "joint1", ObservationType.JOINT_POS),
            ("joint2", "joint2", ObservationType.JOINT_POS),
            ("joint0_vel", "joint0", ObservationType.JOINT_VEL),
            ("joint1_vel", "joint1", ObservationType.JOINT_VEL),
            ("joint2_vel", "joint2", ObservationType.JOINT_VEL),
            ("target_x", "target_x", ObservationType.JOINT_POS),
            ("target_y", "target_y", ObservationType.JOINT_POS),
        ]

        additional_data_spec = [
            ("joint0", "joint0", ObservationType.JOINT_POS),
            ("joint1", "joint1", ObservationType.JOINT_POS),
            ("joint2", "joint2", ObservationType.JOINT_POS),
            ("joint0_vel", "joint0", ObservationType.JOINT_VEL),
            ("joint1_vel", "joint1", ObservationType.JOINT_VEL),
            ("joint2_vel", "joint2", ObservationType.JOINT_VEL),
            ("target_x", "target_x", ObservationType.JOINT_POS),
            ("target_y", "target_y", ObservationType.JOINT_POS),
            ("target", "target", ObservationType.BODY_POS),
            ("fingertip", "fingertip", ObservationType.BODY_POS),
        ]

        collision_groups = [("reacher", ["root", "link0", "link1", "link2"])]

        super().__init__(
            xml_path,
            action_spec,
            observation_spec,
            gamma,
            horizon,
            n_substeps=4,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
        )

        self.init_robot_pos = np.array([0.0, 0.0, 0.0])

    def reward(self, state, action, next_state, absorbing):
        # TODO: implement the reward function
        # reward = - ||finger_position - target_position||
        # [YOUR CODE]
        # Extract finger_position and target_position from additional_data
        fingertip_pos = self._read_additional_data("fingertip", ObservationType.BODY_POS)
        target_pos = self._read_additional_data("target", ObservationType.BODY_POS)

        # Calculate the negative Euclidean distance
        reward = -np.linalg.norm(fingertip_pos[:2] - target_pos[:2])

        return reward

    def is_absorbing(self, state):
        return False

    def setup(self, obs=None):
        # Initialise robot state
        self._write_data("joint0", 0)
        self._write_data("joint1", 0)
        self._write_data("joint2", 0)
        self._write_data("joint0_vel", 0)
        self._write_data("joint1_vel", 0)
        self._write_data("joint2_vel", 0)

        # Initialise goal state
        target_x = -0.2 + np.clip(np.random.normal(scale=0.01), -0.05, 0.05)
        target_y = -0.2 + np.clip(np.random.normal(scale=0.01), -0.05, 0.05)
        self._write_data("target_x", target_x)
        self._write_data("target_y", target_y)

    def _preprocess_action(self, action):
        action_clipped = np.clip(
            action, self.info.action_space.low, self.info.action_space.high
        )
        return action_clipped

    def _modify_mdp_info(self, mdp_info):
        # Modify mdp_info to contain low/high of sin & cos joint angles
        low = mdp_info.observation_space.low
        high = mdp_info.observation_space.high
        angles_low = -1 * np.ones(6)
        angles_high = 1 * np.ones(6)
        low_new = np.concatenate((angles_low, low[3:]))
        high_new = np.concatenate((angles_high, high[3:]))
        new_observation_space = Box(low=low_new, high=high_new)
        mdp_info.observation_space = new_observation_space
        return mdp_info

    def _modify_observation(self, obs):
        # TODO: Change raw angle readings in obs to sin and cos angle readings
        # First three obs should we removed, sines and cosines of angle readings
        # should be prepended to the observation
        # [YOUR CODE]
        # Extract raw angles for three joints
        raw_angles = obs[:3]

        # Calculate sine and cosine values of the angles
        sin_values = np.sin(raw_angles)
        cos_values = np.cos(raw_angles)

        # Concatenate sin, cos values and rest of the observations
        obs_modified = np.concatenate([sin_values, cos_values, obs[3:]])

        return obs_modified
