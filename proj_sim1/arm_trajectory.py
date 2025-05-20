import numpy as np
import os

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
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

    def setup(self, articulation):
        self._articulation = articulation

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

    def setup_cspace_trajectory(self, position, dig_type):
        arm_joint_names = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]
        joint_name_to_index = {name: i for i, name in enumerate(self._articulation.dof_names)}
        full_joint_positions = self._articulation.get_joint_positions()
        arm_joint_positions = [full_joint_positions[joint_name_to_index[name]] for name in arm_joint_names]

        if dig_type == "dig":
            p1 = arm_joint_positions
            p2 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1], position[2]+0.3]), np.array([1, 0, 0, 0]))[0]
            p3 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, position, np.array([1, 0, 0, 0]))[0]
            p4 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1]-0.2, position[2]]), np.array([1, 0, 0, 0]))[0]
            p5 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1]-0.2, position[2]+0.3]), np.array([1, 0, 0, 0]))[0]
            c_space_points = np.array([p1, p2, p3, p4, p5])
            timestamps = np.array([0, 5, 10, 13, 18])

        elif dig_type == "dump":
            p1 = arm_joint_positions
            p2 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1], position[2]+0.2]), np.array([1, 1, 0, 0]))[0]
            p3 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, position, np.array([1, 1, 0, 0]))[0]
            p4 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1]-0.1, position[2]]), np.array([1, 0, 0, 0]))[0]
            p5 = self._kinematics_solver.compute_inverse_kinematics(self._end_effector_name, np.array([position[0], position[1]-0.1, position[2]+0.2]), np.array([1, 0, 0, 0]))[0]
            c_space_points = np.array([p1, p2, p3, p4, p5])
            timestamps = np.array([0, 5, 10, 13, 18])

        trajectory_timestamped = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points,timestamps)

        # Visualize c-space targets in task space
        # for i,point in enumerate(c_space_points):
        #     position,rotation = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, point)
        #     add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}")
        #     frame = XFormPrim(f"/visualized_frames/target_{i}",scale=[.04,.04,.04])
        #     frame.set_world_pose(position,rot_matrices_to_quats(rotation))

        if trajectory_timestamped is None:
                carb.log_warn("No trajectory could be computed")
                self._action_sequence = []
        else:
            physics_dt = 1/60
            self._action_sequence = []

            articulation_trajectory_timestamped = ArticulationTrajectory(self._articulation, trajectory_timestamped, physics_dt)
            self._action_sequence.extend(articulation_trajectory_timestamped.get_action_sequence())

    def update(self):
        if len(self._action_sequence) == 0:
            print("zero length trajectory") # no trajectory found
            return True

        if self._action_sequence_index >= len(self._action_sequence):
            print("trajectory done")
            self._action_sequence_index = 0 # finished trajectory
            return True

        if self._action_sequence_index == 0:
            none_indices = [i for i, x in enumerate(self._action_sequence) if x is None]
            if none_indices:
                print(f"action sequence contains None at indices: {none_indices}")
            print("at teleport step, action sequence has length: " + str(len(self._action_sequence)))
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])
        self._action_sequence_index += 1
        return False

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