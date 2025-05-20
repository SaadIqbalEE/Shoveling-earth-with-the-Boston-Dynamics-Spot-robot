# Launch Isaac Sim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim and USD imports
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.usd
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


# Stage Setup
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)

# === FILE PATHS ===
# ENVIRONMENT_USD_PATH = "./Simp_Env.usd"
ENVIRONMENT_USD_PATH = "./testEnvironment.usd"
OBJECT_USD_PATH = "./spot.usd"
ENV_PRIM_PATH = "/World/Environment"
OBJ_PRIM_PATH = "/World/MyObject"
CAMERA_PRIM_PATH = "/World/MyObject/front_camera"
OBJECT_POSITION = (1.0, 1.0, 0.5)

# Load Environment and Object
add_reference_to_stage(ENVIRONMENT_USD_PATH, ENV_PRIM_PATH)
add_reference_to_stage(OBJECT_USD_PATH, OBJ_PRIM_PATH)

# Set Object Transform
object_prim = stage.GetPrimAtPath(OBJ_PRIM_PATH)
if object_prim.IsValid():
    xform = UsdGeom.Xformable(object_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(OBJECT_POSITION)

# Enable Physics
# if not UsdPhysics.ArticulationRootAPI.Get(stage, OBJ_PRIM_PATH):
#     UsdPhysics.ArticulationRootAPI.Apply(object_prim)
physx_root = PhysxSchema.PhysxRigidBodyAPI.Apply(object_prim)
physx_root.CreateDisableGravityAttr().Set(False)

# arm_ee_prim_path = "/World/MyObject/arm_ee"
# arm_ee_prim = stage.GetPrimAtPath(arm_ee_prim_path)
# if not arm_ee_prim.IsValid():
#     print(f"Error: Prim at {arm_ee_prim_path} not found in the stage.")
# else:
#     UsdPhysics.MassAPI.Apply(arm_ee_prim)
#     mass_api = UsdPhysics.MassAPI(arm_ee_prim)
#     mass_api.CreateMassAttr().Set(1.0)
#     mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.1, 0.1, 0.1))

# Start Simulation
world.reset()

# Articulation Setup
from omni.isaac.core.articulations import Articulation
spot = Articulation(prim_path=OBJ_PRIM_PATH)

for _ in range(30):
    world.step(render=False)

#Optional: Set Initial Joint Positions
joint_targets = {
    # Hip abduction/adduction – zero keeps legs straight outward
    "fl_hx": 0.0,
    "fr_hx": 0.0,
    "hl_hx": 0.0,
    "hr_hx": 0.0,

    # Hip pitch – slight forward/downward bend
    "fl_hy": 0.6,
    "fr_hy": 0.6,
    "hl_hy": 0.6,
    "hr_hy": 0.6,

    # Knee joints – bent backward for a standing crouch
    "fl_kn": -1.2,
    "fr_kn": -1.2,
    "hl_kn": -1.2,
    "hr_kn": -1.2,

    # Arm shoulder pan – aligned forward
    "arm0_sh0": 0.0,

    # Arm shoulder lift – slightly raised
    "arm0_sh1": -0.3,

    # Elbows – slightly bent
    "arm0_el0": 1.0,
    "arm0_el1": 0.0,

    # Wrist joints – neutral
    "arm0_wr0": 0.0,
    "arm0_wr1": 0.0,
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

# Combined Executor
executor = MultiThreadedExecutor()
executor.add_node(ros_interface)
executor.add_node(camera_pub)

ros_thread = Thread(target=executor.spin, daemon=True)
ros_thread.start()

# Arm trajectory class
import arm_trajectory
spot_arm = arm_trajectory.spot_arm_trajectory()
setup_arm = True # True if need for arm setup, False if already setup
trajectory_done = False

# Main Loop
while simulation_app.is_running():
    ros_interface.publish_state()
    camera_pub.publish()
    world.step(render=True)
    if setup_arm is True:
        # Setup arm trajectory
        spot_arm.setup(spot) # Articulation as argument
        dig_position = np.array([1, 0, 0]) # x, y, z coordinates for dig target
        spot_arm.setup_cspace_trajectory(dig_position, "dig") # Create trajectory, "dig" for digging, "dump" for dumping earth
        setup_arm = False
    if trajectory_done is False:
        arm_step = spot_arm.update()
        if arm_step is True: # Check if trajectory is complete
            trajectory_done = True

# Cleanup
simulation_app.close()
ros_interface.destroy_node()
camera_pub.destroy_node()
rclpy.shutdown()
