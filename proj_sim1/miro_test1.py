# Launch Isaac Sim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim and USD imports
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
from isaacsim.core.prims import RigidPrim
import omni.usd
import carb
import numpy as np

# ROS2 Imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# Threading
from threading import Thread

# Isaac Sim Camera
from isaacsim.sensors.camera import Camera


class CameraPublisher(Node):
    def __init__(self, camera_prim_path):
        super().__init__('camera_publisher')
        self.publisher = self.create_publisher(Image, '/front_cam/image_raw', 10)
        self.bridge = CvBridge()
        self.camera = Camera(camera_prim_path)
        self.camera.initialize()
        self.camera.set_resolution((640, 480)) # Ensure usable resolution
        print("Camera initialized:", self.camera.is_valid())

    def publish(self):
        image = self.camera.get_rgba()
        if image is None or image.ndim != 3 or image.shape[2] < 3:
            self.get_logger().warn("Invalid or missing camera image")
            return

        rgb_image = (image[:, :, :3] * 255).astype(np.uint8)
        ros_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        self.publisher.publish(ros_image)


class SpotROSInterface(Node):
    def __init__(self, spot, world):
        super().__init__('spot_ros_interface')
        self.spot = spot
        self.world = world

        self.joint_pub = self.create_publisher(JointState, '/spot/joint_states', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/spot/world_pose', 10)

    def publish_state(self):
        pose_msg = PoseStamped()
        position, orientation = self.spot.get_world_pose()

        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])

        self.pose_pub.publish(pose_msg)

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.spot.dof_names
        joint_msg.position = [float(p) for p in self.spot.get_joint_positions()]
        self.joint_pub.publish(joint_msg)



# === FILE PATHS ===
ENVIRONMENT_USD_PATH = "/home/rllab/Desktop/25P24/IsaacEnvironments/Environment_with_containers_nearby.usd"
#ENVIRONMENT_USD_PATH = "/home/rllab/Desktop/25P24/IsaacEnvironments/Environment_with_obstacles.usd"
# from david: test Environment1.usd for better adaptability (however, the stone is lost)
# ENVIRONMENT_USD_PATH = "/home/rllab/Desktop/25P24/IsaacEnvironments/Environment1.usd"
OBJECT_USD_PATH = "/home/rllab/Desktop/25P24/proj_sim1/spot_with_shovel.usd"
#OBJECT_USD_PATH = "/home/rllab/Desktop/25P24/moving/spot_arm_shovel.usd"
ENV_PRIM_PATH = "/World/Environment"
#OBJ_PRIM_PATH = "/World/MyObject"
OBJ_PRIM_PATH = "/World/spot"
#CAMERA_PRIM_PATHL = "/World/spot/body/frontright_fisheye"
#CAMERA_PRIM_PATHR = "/World/spot/body/frontright_fisheye"
CAMERA_PRIM_PATH = "/World/spot/front_camera"
#CAMERA_PRIM_PATH = "/World/MyObject/front_camera"
OBJECT_POSITION = (-3.40, -2, 0.71)

# === Load Environment USD Stage ===
try:
    result = omni.usd.get_context().open_stage(ENVIRONMENT_USD_PATH)
    if not result:
        raise RuntimeError(f"Failed to open environment stage at {ENVIRONMENT_USD_PATH}")
except Exception as e:
    carb.log_error(str(e))
    simulation_app.close()
    exit(1)

# Wait until the stage is fully loaded
simulation_app.update()
simulation_app.update()
while is_stage_loading():
    simulation_app.update()

# Load Spot robot
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
add_reference_to_stage(OBJECT_USD_PATH, OBJ_PRIM_PATH)

# Set Object Transform
object_prim = stage.GetPrimAtPath(OBJ_PRIM_PATH)
if object_prim.IsValid():
    xform = UsdGeom.Xformable(object_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(OBJECT_POSITION)

# Enable Physics
physx_root = PhysxSchema.PhysxRigidBodyAPI.Apply(object_prim)
physx_root.CreateDisableGravityAttr().Set(False)

# Start Simulation
world.reset()

# Articulation Setup
from omni.isaac.core.articulations import Articulation
spot = Articulation(prim_path=OBJ_PRIM_PATH)

for _ in range(30):
    world.step(render=False)


# Optional: Set Initial Joint Positions
joint_targets = {
    "fl_hx": 0.0, "fr_hx": 0.0, "hl_hx": 0.0, "hr_hx": 0.0,
    "fl_hy": 0.6, "fr_hy": 0.6, "hl_hy": 0.6, "hr_hy": 0.6,
    "fl_kn": -1.2, "fr_kn": -1.2, "hl_kn": -1.2, "hr_kn": -1.2,
    "arm0_sh0": 0.0, "arm0_sh1": -3,
    "arm0_el0": 3, "arm0_el1": 0.0,
    "arm0_wr0": 0.0, "arm0_wr1": 0.0
}
print("Available DOF names:", spot.dof_names)
target_positions = np.zeros(spot.num_dof)
for joint_name, angle in joint_targets.items():
    try:
        idx = spot.get_dof_index(joint_name)
        target_positions[idx] = angle
    except Exception as e:
        print(f"Warning: Joint '{joint_name}' not found. {e}")

spot.initialize()
controller = spot.get_articulation_controller()
spot.set_joint_positions(target_positions)

for _ in range(300):
    world.step(render=True)

# Initialize ROS2
rclpy.init()
ros_interface = SpotROSInterface(spot, world)
camera_pub = CameraPublisher(CAMERA_PRIM_PATH)
#camera_pubL = CameraPublisher(CAMERA_PRIM_PATHL, 'left')
#camera_pubR = CameraPublisher(CAMERA_PRIM_PATHR, 'right')

# Combined Executor
executor = MultiThreadedExecutor()
executor.add_node(ros_interface)
executor.add_node(camera_pub)
#executor.add_node(camera_pubL)
#executor.add_node(camera_pubR)

ros_thread = Thread(target=executor.spin, daemon=True)
ros_thread.start()

# Arm trajectory class
import arm_trajectory
spot_arm = arm_trajectory.spot_arm_trajectory()
setup_arm = True
do_trajectory = False
arm_action = "test" # Arm action can be "dig", "dump", "static" or "test". "static" keeps arm static and "test" does both dig and dump
# Coordinate conversion
prim_spot = stage.GetPrimAtPath(OBJ_PRIM_PATH)
xform = UsdGeom.Xformable(prim_spot)
local_to_world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
world_to_local_matrix = local_to_world_matrix.GetInverse()
#world_point = Gf.Vec3d(-2.46, -1.75, 0) # Dig point in world coordinates x, y, z
world_point = Gf.Vec3d(-2.49, -1.70, 0) # Dig point in world coordinates x, y, z
local_point = world_to_local_matrix.Transform(world_point)

# Main Loop
while simulation_app.is_running():
    ros_interface.publish_state()
    camera_pub.publish()
    world.step(render=True)

    if setup_arm is True:
        # Setup arm trajectory
        spot_arm.setup(spot) # Setup with articulation as argument
        setup_arm = False

    if arm_action == "dig":
        do_trajectory = True
        arm_action = "static"
        print("\nDigging position: " + str(world_point) + "\n")
        trajectory = spot_arm.setup_cspace_trajectory(local_point, "dig") # Create trajectory, "dig" for digging
        if trajectory is False:
            print("\nCannot reach digging position " + str(world_point) + "\n")
            do_trajectory = False

    elif arm_action == "dump":
        do_trajectory = True
        arm_action = "static"
        trajectory = spot_arm.setup_cspace_trajectory(np.array([0, 0, 0]), "dump") # Create trajectory, "dump" for dumping earth. in dumping, position argument is not used
        if trajectory is False:
            print("\nCannot reach dumping position\n")
            do_trajectory = False

    elif arm_action == "test":
        do_trajectory = True
        arm_action = "static"
        print("\nDigging position: " + str(world_point) + "\n")
        trajectory = spot_arm.setup_cspace_trajectory(local_point, "test") # Create trajectory, "dig" for digging
        if trajectory is False:
            print("\nCannot reach digging position " + str(world_point) + "\n")
            do_trajectory = False

    if do_trajectory: # Check if we want to execute trajectory
        arm_step = spot_arm.update() # Move arm one step
        if arm_step is True: # Check if trajectory is complete
            do_trajectory = False

# Cleanup
simulation_app.close()
ros_interface.destroy_node()
camera_pub.destroy_node()
rclpy.shutdown()
