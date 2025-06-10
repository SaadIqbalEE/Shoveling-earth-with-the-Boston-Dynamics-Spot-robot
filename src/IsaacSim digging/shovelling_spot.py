# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse
import sys
import numpy as np

from isaacsim import SimulationApp

# Specify if test mode is wanted for simulation
test = False

# This sample loads a usd stage and starts simulation
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RaytracedLighting"}

kit = SimulationApp(launch_config=CONFIG)

import carb
import omni

import arm_trajectory
# Locate Isaac Sim assets folder to load sample
from isaacsim.storage.native import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
    
# usd path to spot environment    
usd_path = "/home/rllab/Desktop/25P24/IsaacEnvironments/testEnvironment.usd"

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {usd_path} is a valid usd file."
    )
    kit.close()
    sys.exit()
# Wait two frames so that stage starts loading
kit.update()
kit.update()

print("Loading stage...")
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory,
    ArticulationKinematicsSolver
)

world = World()
world.reset()
spot_arm = arm_trajectory.spot_arm_trajectory()

while is_stage_loading():
    kit.update()
print("Loading Complete")
omni.timeline.get_timeline_interface().play()
arm_setup = False

#testing
#robot_prim_path = "/World/spot_with_arm"
#articulation = Articulation(robot_prim_path)
#kinematics_solver = LulaKinematicsSolver(
#    robot_description_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.yaml",
#    urdf_path = "/home/rllab/Desktop/25P24/Isaac_spot_tutorials/asset/spot.urdf"
#)
#articulation_kinematics = ArticulationKinematicsSolver(
#    articulation,
#    kinematics_solver,
#    "arm0_link_ee"
#)
def prim_exists(path):
    stage = omni.usd.get_context().get_stage()
    return stage.GetPrimAtPath(path).IsValid()
assert prim_exists("/World/spot"), "Robot prim /World/spot_with_arm does not exist!"

# Run in test mode, exit after a fixed number of steps
if test is True:
    for i in range(10):
        # Run in realtime mode, we don't specify the step size
        kit.update()
else:
    while kit.is_running():
        # Run in realtime mode, we don't specify the step size
        kit.update()
        # Execute defined arm trajectory
        if arm_setup is False:
            #action, truthValue = articulation_kinematics.compute_inverse_kinematics(
            #    np.array([0, 0, 1.0]),
            #    np.array([0, 0, 0])
            #)
            #articulation.apply_action(action)
            spot_arm.setup()
            print("DOFs in robot:", spot_arm._articulation.dof_names)
            spot_arm.setup_cspace_trajectory()
            arm_setup = True
        spot_arm.update()
        

omni.timeline.get_timeline_interface().stop()
kit.close()
