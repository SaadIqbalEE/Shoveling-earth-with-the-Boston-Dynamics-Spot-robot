# only when the Isaac Sim has been opened (A Unit Test)
# For group member (only written in English is for your reference, something in Chinese is just my own thinking)
# How to launch (script):
# /home/rllab/IsaacSim/python.sh print_object.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
# False: Open GUI

import omni.usd
from pxr import UsdGeom, Gf

# Path for the real simulation (which include the sphere and cube)
SCENE_PATH = "/home/rllab/Desktop/25P24/proj_sim1/testEnvironment.usd"
omni.usd.get_context().open_stage(SCENE_PATH)

# Waiting for loading complete (However there is contradiction...)
# while not omni.usd.get_context().is_stage_loading_complete():
#     simulation_app.update()

# 获取 Stage
stage = omni.usd.get_context().get_stage()

print("=== Listing Cubes and Spheres ===")
for prim in stage.Traverse():
    name = prim.GetName()
    type_name = prim.GetTypeName()
    path = prim.GetPath()
    if prim.IsA(UsdGeom.Camera):
        print(f"Camera: {prim.GetPath()}")
    if type_name in ["Cube", "Sphere"]:
        # 获取变换信息
        try:
            xform = UsdGeom.Xformable(prim)
            scale = Gf.Vec3f(1.0, 1.0, 1.0)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = op.Get()
                    break
        except Exception:
            scale = Gf.Vec3f(1.0, 1.0, 1.0)

        print(f"{type_name}: {path} | name: {name} | scale: {scale}")

# if you don't want to close the GUI
simulation_app.close()
