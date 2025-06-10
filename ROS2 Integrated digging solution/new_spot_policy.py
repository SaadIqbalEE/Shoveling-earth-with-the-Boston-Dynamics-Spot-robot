# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

import numpy as np
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController

class SpotFlatTerrainPolicy(PolicyController):
    """The Spot quadruped — policy controls only the legs."""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None,
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0

        # Get full joint list
        all_joint_names = self.robot.dof_names
        self.leg_joint_names = [
            'fl_hx', 'fl_hy', 'fl_kn',
            'fr_hx', 'fr_hy', 'fr_kn',
            'hl_hx', 'hl_hy', 'hl_kn',
            'hr_hx', 'hr_hy', 'hr_kn',
        ]
        self.leg_joint_indices = [all_joint_names.index(name) for name in self.leg_joint_names]

        # Store default full joint positions
        joint_positions = self.robot.get_joint_positions()
        self.default_pos = np.array(joint_positions, dtype=np.float32)

    def _compute_observation(self, command):
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.T
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        all_joint_pos = np.array(self.robot.get_joint_positions())
        all_joint_vel = np.array(self.robot.get_joint_velocities())
        joint_delta = all_joint_pos - self.default_pos

        obs = np.zeros(69)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        obs[12:31] = joint_delta
        obs[31:50] = all_joint_vel
        obs[50:69] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation.

        Only applies control to the 12 leg joints.
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            full_action = self._compute_action(obs)  # Full 19D from policy
            self._previous_action = full_action.copy()

        # Only apply action to leg joints
        full_joint_targets = np.array(self.default_pos.copy())
        full_joint_targets[self.leg_joint_indices] += (
            self._previous_action[self.leg_joint_indices] * self._action_scale
        )

        action = ArticulationAction(joint_positions=full_joint_targets)
        self.robot.apply_action(action)
        self._policy_counter += 1

class SpotArmFlatTerrainPolicy(PolicyController):
    """The Spot quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: str = None,
        policy_path: str = None, 
        policy_params_path: str = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(policy_path, policy_params_path)
        self._action_scale = 0.2
        self._previous_action = np.zeros(19)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(69)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:31] = current_joint_pos - self.default_pos
        obs[31:50] = current_joint_vel
        # Previous Action
        obs[50:69] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        full_action = np.array(self.default_pos)  # Start from current default pos (19 DOFs)
        full_action[self.leg_joint_indices] += self.action * self._action_scale

        action = ArticulationAction(joint_positions=full_action)
        self.robot.apply_action(action)

        self._policy_counter += 1
