import numpy as np
import os

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats, euler_angles_to_quats, quats_to_euler_angles
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from scipy.spatial.transform import Rotation as R

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
            robot_description_path = "./modified_asset/spot.yaml",
            urdf_path = "./modified_asset/spot.urdf"
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = "./modified_asset/spot.yaml",
            urdf_path = "./modified_asset/spot.urdf"
        )

        self._end_effector_name = "arm0_link_ee"

    def setup_cspace_trajectory(self, position, dig_type):
        arm_joint_names = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"]
        joint_name_to_index = {name: i for i, name in enumerate(self._articulation.dof_names)}
        full_joint_positions = self._articulation.get_joint_positions()
        arm_joint_positions = [full_joint_positions[joint_name_to_index[name]] for name in arm_joint_names]
        cur_pos, cur_rot = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, arm_joint_positions)
        dig_start_positions = [0.0, -0.3, 1.0, 0.0, 0.0, 0.0]
        dig_pos, dig_rot = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, dig_start_positions)
        dig_start_positions_from_coord = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([position[0]-0.15, position[1], dig_pos[2]]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 50, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
        dig_pos_c, dig_rot_c = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, dig_start_positions_from_coord)
        rest_positions = [0.0, -3, 3, 0.0, 0.0, 0.0]

        if dig_type == "dig":
            # Trajectory points. Arguments for IK are end effector, position (x, y, z), orientation quaternion (w, x, y, z)
            p1 = arm_joint_positions
            p2 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos[0]-0.1, dig_pos[1], dig_pos[2]+0.35]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p3 = dig_start_positions
            p4 = dig_start_positions_from_coord
            p5 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos_c[0]-0.12, dig_pos_c[1], dig_pos_c[2]-0.165]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 50, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p6 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos_c[0]+0.165, dig_pos_c[1], dig_pos_c[2]-0.22]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p7 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos[0]-0.1, dig_pos[1], dig_pos[2]+0.35]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            c_space_points_1 = np.array([p1, p2])
            c_space_points_2 = np.array([p2, p3])
            c_space_points_3 = np.array([p3, p4])
            c_space_points_4 = np.array([p4, p5, p6, p7])
            timestamps_1 = np.array([0, 1])
            timestamps_2 = np.array([0, 1])
            timestamps_3 = np.array([0, 1])
            timestamps_4 = np.array([0, 1, 3, 5])
            trajectory_timestamped_1 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_1,timestamps_1)
            trajectory_timestamped_2 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_2,timestamps_2)
            trajectory_timestamped_3 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_3,timestamps_3)
            trajectory_timestamped_4 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_4,timestamps_4)

        elif dig_type == "dump":
            p1 = arm_joint_positions
            p2 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos[0]-0.1, dig_pos[1], dig_pos[2]+0.35]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p3 = [p2[0]+np.pi/2, p2[1], p2[2], p2[3], p2[4], p2[5]]
            p4 = [p3[0], p3[1], p3[2], p3[3], p3[4], p3[5]+2.8]
            p5 = rest_positions
            c_space_points_1 = np.array([p1, p2])
            timestamps_1 = np.array([0, 1])
            c_space_points_2 = np.array([p2, p3])
            timestamps_2 = np.array([0, 3])
            c_space_points_3 = np.array([p3, p4])
            timestamps_3 = np.array([0, 1])
            c_space_points_4 = np.array([p4, p3])
            timestamps_4 = np.array([0, 1])
            c_space_points_5 = np.array([p3, p5])
            timestamps_5 = np.array([0, 2])
            trajectory_timestamped_5 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_1,timestamps_1)
            trajectory_timestamped_6 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_2,timestamps_2)
            trajectory_timestamped_7 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_3,timestamps_3)
            trajectory_timestamped_8 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_4,timestamps_4)
            trajectory_timestamped_9 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_5,timestamps_5)

        elif dig_type == "test":
            p1 = arm_joint_positions
            p2 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos[0]-0.1, dig_pos[1], dig_pos[2]+0.35]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p3 = dig_start_positions
            p4 = dig_start_positions_from_coord
            p5 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos_c[0]-0.12, dig_pos_c[1], dig_pos_c[2]-0.165]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 50, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p6 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos_c[0]+0.165, dig_pos_c[1], dig_pos_c[2]-0.22]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p7 = self._kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                np.array([dig_pos[0]-0.1, dig_pos[1], dig_pos[2]+0.35]), # position (x, y, z)
                euler_angles_to_quats(np.array([0, 0, 0]), degrees=True))[0] # orientation euler angle in degrees (x, y, z)
            p8 = [p7[0]+np.pi/2+0.5, p7[1], p7[2], p7[3], p7[4], p7[5]]
            p9 = [p8[0], p8[1], p8[2], p8[3], p8[4], p8[5]+2.8]
            p10 = rest_positions
            c_space_points_1 = np.array([p1, p2])
            c_space_points_2 = np.array([p2, p3])
            c_space_points_3 = np.array([p3, p4])
            c_space_points_4 = np.array([p4, p5, p6, p7])
            timestamps_1 = np.array([0, 1])
            timestamps_2 = np.array([0, 1])
            timestamps_3 = np.array([0, 1])
            timestamps_4 = np.array([0, 1, 3, 5])
            trajectory_timestamped_1 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_1,timestamps_1)
            trajectory_timestamped_2 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_2,timestamps_2)
            trajectory_timestamped_3 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_3,timestamps_3)
            trajectory_timestamped_4 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_4,timestamps_4)
            c_space_points_2 = np.array([p7, p8])
            timestamps_2 = np.array([0, 3])
            c_space_points_3 = np.array([p8, p9])
            timestamps_3 = np.array([0, 1])
            c_space_points_4 = np.array([p9, p8])
            timestamps_4 = np.array([0, 1])
            c_space_points_5 = np.array([p8, p10])
            timestamps_5 = np.array([0, 2])
            trajectory_timestamped_6 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_2,timestamps_2)
            trajectory_timestamped_7 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_3,timestamps_3)
            trajectory_timestamped_8 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_4,timestamps_4)
            trajectory_timestamped_9 = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points_5,timestamps_5)
            
        # Visualize c-space targets in task space
        #for i,point in enumerate(c_space_points):
        #    position,rotation = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, point)
        #    add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}")
        #    frame = XFormPrim(f"/visualized_frames/target_{i}",scales=np.array([[.04,.04,.04]]))
        #    frame.set_world_poses(
        #        positions=np.array([position]),
        #        orientations=np.array([rot_matrices_to_quats(rotation)])
        #    )

        #if trajectory_timestamped is None:
        #    carb.log_warn("No trajectory could be computed")
        #    self._action_sequence = []
        #else:
        physics_dt = 1/60
        self._action_sequence = []

        if dig_type == "dig":
            try:
                articulation_trajectory_timestamped_1 = ArticulationTrajectory(self._articulation, trajectory_timestamped_1, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_2 = ArticulationTrajectory(self._articulation, trajectory_timestamped_2, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_3 = ArticulationTrajectory(self._articulation, trajectory_timestamped_3, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_4 = ArticulationTrajectory(self._articulation, trajectory_timestamped_4, physics_dt).get_action_sequence()
            except:
                return False
            if (None in articulation_trajectory_timestamped_1 or None in articulation_trajectory_timestamped_2 or None in articulation_trajectory_timestamped_3 or None in articulation_trajectory_timestamped_4):
                return False
            self._action_sequence.extend(articulation_trajectory_timestamped_1)
            self._action_sequence.extend(articulation_trajectory_timestamped_2)
            self._action_sequence.extend(articulation_trajectory_timestamped_3)
            self._action_sequence.extend(articulation_trajectory_timestamped_4)

        elif dig_type == "dump":
            try:
                articulation_trajectory_timestamped_5 = ArticulationTrajectory(self._articulation, trajectory_timestamped_5, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_6 = ArticulationTrajectory(self._articulation, trajectory_timestamped_6, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_7 = ArticulationTrajectory(self._articulation, trajectory_timestamped_7, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_8 = ArticulationTrajectory(self._articulation, trajectory_timestamped_8, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_9 = ArticulationTrajectory(self._articulation, trajectory_timestamped_9, physics_dt).get_action_sequence()
            except:
                return False
            if (None in articulation_trajectory_timestamped_5 or None in articulation_trajectory_timestamped_6 or None in articulation_trajectory_timestamped_7 or None in articulation_trajectory_timestamped_8 or None in articulation_trajectory_timestamped_9):
                return False
            self._action_sequence.extend(articulation_trajectory_timestamped_5)
            self._action_sequence.extend(articulation_trajectory_timestamped_6)
            self._action_sequence.extend(articulation_trajectory_timestamped_7)
            self._action_sequence.extend(articulation_trajectory_timestamped_8)
            self._action_sequence.extend(articulation_trajectory_timestamped_9)

        elif dig_type == "test":
            try:
                articulation_trajectory_timestamped_1 = ArticulationTrajectory(self._articulation, trajectory_timestamped_1, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_2 = ArticulationTrajectory(self._articulation, trajectory_timestamped_2, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_3 = ArticulationTrajectory(self._articulation, trajectory_timestamped_3, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_4 = ArticulationTrajectory(self._articulation, trajectory_timestamped_4, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_6 = ArticulationTrajectory(self._articulation, trajectory_timestamped_6, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_7 = ArticulationTrajectory(self._articulation, trajectory_timestamped_7, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_8 = ArticulationTrajectory(self._articulation, trajectory_timestamped_8, physics_dt).get_action_sequence()
                articulation_trajectory_timestamped_9 = ArticulationTrajectory(self._articulation, trajectory_timestamped_9, physics_dt).get_action_sequence()
            except:
                return False
            if (None in articulation_trajectory_timestamped_1 or None in articulation_trajectory_timestamped_2 or None in articulation_trajectory_timestamped_3 or None in articulation_trajectory_timestamped_4 or None in articulation_trajectory_timestamped_6 or None in articulation_trajectory_timestamped_7 or None in articulation_trajectory_timestamped_8 or None in articulation_trajectory_timestamped_9):
                return False
            self._action_sequence.extend(articulation_trajectory_timestamped_1)
            self._action_sequence.extend(articulation_trajectory_timestamped_2)
            self._action_sequence.extend(articulation_trajectory_timestamped_3)
            self._action_sequence.extend(articulation_trajectory_timestamped_4)
            self._action_sequence.extend(articulation_trajectory_timestamped_6)
            self._action_sequence.extend(articulation_trajectory_timestamped_7)
            self._action_sequence.extend(articulation_trajectory_timestamped_8)
            self._action_sequence.extend(articulation_trajectory_timestamped_9)

        return True

    def update(self):
        if len(self._action_sequence) == 0:
            print("zero length trajectory") # no trajectory found
            return True

        if self._action_sequence_index >= len(self._action_sequence):
            print("\nTrajectory done\n")
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

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        #self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))