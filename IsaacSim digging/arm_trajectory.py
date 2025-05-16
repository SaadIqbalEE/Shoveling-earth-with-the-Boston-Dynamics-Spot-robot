import numpy as np
import os

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path

from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import lula

class spot_arm_trajectory():
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None

    def setup(self):
        robot_prim_path = "/World/spot_with_arm"
        self._articulation = Articulation(robot_prim_path)

        # Config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        #mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        #rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        #Initialize a LulaCSpaceTrajectoryGenerator object
        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.yaml",
            urdf_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.urdf"
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.yaml",
            urdf_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.urdf"
        )

        self._end_effector_name = "arm0_link_ee"

    def setup_cspace_trajectory(self):
        c_space_points = np.array([
            [1 , 0.5 , 1 , 1 , 1 , 0 , ],
            [1 , 0.5 , 1.1 , 1.1 , 1 , 0 , ],
            [1 , 0.4 , 1.2 , 1.1 , 1 , 0, ],
            [1 , 0.4 , 1.3 , 1.2 , 1 , 0 , ]
            ])

        timestamps = np.array([0,5,10,13])

        trajectory_time_optimal = self._c_space_trajectory_generator.compute_c_space_trajectory(c_space_points)
        trajectory_timestamped = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points,timestamps)

        # Visualize c-space targets in task space
        # for i,point in enumerate(c_space_points):
        #     position,rotation = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, point)
        #     add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}")
        #     frame = XFormPrim(f"/visualized_frames/target_{i}",scale=[.04,.04,.04])
        #     frame.set_world_pose(position,rot_matrices_to_quats(rotation))

        if trajectory_time_optimal is None or trajectory_timestamped is None:
                carb.log_warn("No trajectory could be computed")
                self._action_sequence = []
        else:
            physics_dt = 1/60
            self._action_sequence = []

            # Follow both trajectories in a row

            articulation_trajectory_time_optimal = ArticulationTrajectory(self._articulation, trajectory_time_optimal, physics_dt)
            self._action_sequence.extend(articulation_trajectory_time_optimal.get_action_sequence())

            articulation_trajectory_timestamped = ArticulationTrajectory(self._articulation, trajectory_timestamped, physics_dt)
            self._action_sequence.extend(articulation_trajectory_timestamped.get_action_sequence())

    def update(self, step: float):
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= len(self._action_sequence) + 10 # Wait 10 frames before repeating trajectories
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])

        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10 # Wait 10 frames before repeating trajectories

    def reset(self):
        # Delete any visualized frames
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))