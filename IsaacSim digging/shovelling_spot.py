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
usd_path = "/home/rllab/Desktop/25P24/IsaacEnvironments/Environment1.usd"

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

world = World()
world.reset()
spot_arm = arm_trajectory.spot_arm_trajectory()

while is_stage_loading():
    kit.update()
print("Loading Complete")
omni.timeline.get_timeline_interface().play()
arm_setup = False
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
        #if arm_setup is False:
        #    spot_arm.setup()
        #    spot_arm.setup_cspace_trajectory()
        #    arm_setup = True
        #spot_arm.update()
        

omni.timeline.get_timeline_interface().stop()
kit.close()
