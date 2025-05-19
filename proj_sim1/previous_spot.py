# Launch Isaac Sim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Updated Imports
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Sdf, Usd, UsdPhysics, PhysxSchema
# from omni.isaac.ros2_bridge.scripts.ros2_camera_helper import CameraHelper

import omni.usd
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from threading import Thread

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from omni.isaac.sensor import Camera

class CameraPublisher(Node):
    def __init__(self, camera_prim_path):
        super().__init__('camera_publisher')
        self.publisher = self.create_publisher(Image, '/front_cam/image_raw', 10)
        self.bridge = CvBridge()
        self.camera = Camera(camera_prim_path)
        self.camera.initialize()

    def publish(self):
        image = self.camera.get_rgba()  # shape: (H, W, 4)
        if image is None:
            return

        # Convert RGBA to RGB
        rgb_image = (image[:, :, :3] * 255).astype(np.uint8)
        ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        self.publisher.publish(ros_image)


#For ROS2 integration
class SpotROSInterface(Node):
    def __init__(self, spot, world):
        super().__init__('spot_ros_interface')
        self.spot = spot
        self.world = world

        #self.sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/spot/joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/spot/world_pose', 10)

        #self.cmd_vel = Twist()

    # def cmd_vel_callback(self, msg):
    #     self.cmd_vel = msg  # Save latest command

    def publish_state(self):
        # Publish robot's pose in the world
        pose_msg = PoseStamped()
        position, orientation = self.spot.get_world_pose()

        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        # orientation assumed to be [x, y, z, w]
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])

        self.pose_pub.publish(pose_msg)

        # Publish joint angles
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.spot.dof_names
        joint_msg.position = [float(p) for p in self.spot.get_joint_positions()]
        self.joint_pub.publish(joint_msg)

    # def apply_cmd(self):
    #     # Apply velocity to base of robot
    #     lin = self.cmd_vel.linear
    #     ang = self.cmd_vel.angular

    #     # Translate X/Y and rotate around Z (Yaw)
    #     pose = self.spot.get_world_pose()
    #     delta = np.array([lin.x, lin.y, 0.0]) * 0.01  # Scale factor
    #     new_position = pose.p + delta
    #     self.spot.set_world_pose(position=new_position)

# Stage Setup
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)

# === FILE PATHS ===
ENVIRONMENT_USD_PATH = "./Simp_Env.usd"
OBJECT_USD_PATH = "./spot.usd"
ENV_PRIM_PATH = "/World/Environment"
OBJ_PRIM_PATH = "/World/MyObject"

OBJECT_POSITION = (1.0, 4.0, 0.3)

# === LOAD ENVIRONMENT ===
add_reference_to_stage(usd_path=ENVIRONMENT_USD_PATH, prim_path=ENV_PRIM_PATH)

# === LOAD OBJECT ===
add_reference_to_stage(usd_path=OBJECT_USD_PATH, prim_path=OBJ_PRIM_PATH)

# camera_helper = CameraHelper(
#     camera_prim_path=camera_prim_path,
#     rgb_enabled=True,
#     publish_camera_info=True,
#     ros_camera_prefix="front_cam"
# )

# === SET OBJECT TRANSFORM ===
object_prim = stage.GetPrimAtPath(OBJ_PRIM_PATH)
if object_prim.IsValid():
    xform = UsdGeom.Xformable(object_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(OBJECT_POSITION)

# === ENSURE PHYSICS IS ENABLED ===
# Optionally attach physics root if it's not present
if not UsdPhysics.ArticulationRootAPI.Get(stage, OBJ_PRIM_PATH):
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(OBJ_PRIM_PATH))



# Optional: freeze rotation for testing
physx_root = PhysxSchema.PhysxRigidBodyAPI.Apply(object_prim)
physx_root.CreateDisableGravityAttr().Set(False)
#physx_root.CreateStartAsleepAttr().Set(False)

# === START SIMULATION ===
world.reset()

# === Set Joint Angles ===
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation

# Wrap Spot as Articulation
spot = Articulation(prim_path=OBJ_PRIM_PATH)
spot.initialize()

for _ in range(30):
    world.step(render=False)

# Example joint targets
joint_targets = {
    "fl_hip_joint": 0.2,
    "fl_knee_joint": -0.9,
    "fr_hip_joint": 0.2,
    "fr_knee_joint": -0.9,
    "hl_hip_joint": 0.2,
    "hl_knee_joint": -0.9,
    "hr_hip_joint": 0.2,
    "hr_knee_joint": -0.9,

    # "arm0_sh0": -1.2,
    # "arm0_sh1": 0.3,
    # "arm0_el0": -1.5,
    # "arm0_el1": 0.0,
    # "arm0_wr0": 0.8,
    # "arm0_wr1": 0.0,
}

# Print available DOF names
print("Available DOF names:", spot.dof_names)

# Build the full joint position array
num_dofs = spot.num_dof
target_positions = np.zeros(num_dofs)

# for joint_name, angle in joint_targets.items():
#     try:
#         idx = spot.get_dof_index(joint_name)
#         target_positions[idx] = angle
#     except Exception as e:
#         print(f"Warning: Joint '{joint_name}' not found. {e}")

# Apply joint positions
spot.set_joint_positions(target_positions)

# Let physics settle
for _ in range(300):
    world.step(render=True)


# Initialize ROS
rclpy.init()
camera_pub = CameraPublisher("/World/MyObject/front_camera")
ros_interface = SpotROSInterface(spot, world)

def ros_spin():
    rclpy.spin(ros_interface)

ros_thread = Thread(target=ros_spin, daemon=True)
ros_thread.start()

rclpy_thread = Thread(target=lambda: rclpy.spin(camera_pub), daemon=True)
rclpy_thread.start()

while simulation_app.is_running():
    # ros_interface.apply_cmd()
    ros_interface.publish_state()
    camera_pub.publish()
    world.step(render=True)

# === CLEANUP ===
simulation_app.close()
ros_interface.destroy_node()
rclpy.shutdown()
