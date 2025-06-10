# Launch Isaac Sim
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim and USD imports
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from spot_policy import SpotFlatTerrainPolicy, SpotArmFlatTerrainPolicy
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
from std_msgs.msg import String

# Threading
from threading import Thread

# Isaac Sim Camera
from isaacsim.sensors.camera import Camera

class CameraPublisher(Node):
    def __init__(self, camera_prim_path, name):
        super().__init__('camera_publisher')
        self.publisher = self.create_publisher(Image, '/front_cam/'+name, 10)
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

        # Subscriber
        self.control_sub = self.create_subscription(
            String,
            '/robot_control',
            self.control_callback,
            10
        )

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

    def control_callback(self, msg: String):
        """Parses command string and applies it to the Spot robot."""
        self.get_logger().info(f"Received command: {msg.data}")
        try:
            command_parts = msg.data.strip().split(',')
            mode = command_parts[0]

            if mode == "walk":
                execute_walk(command_parts[1:])
            elif mode == "dig":
                execute_dig()
            elif mode == "dmp":
                execute_dump()
            elif mode == "rot_cw":
                execute_rotcw()
            elif mode == "rot_ccw":
                execute_rotccw()
            else:
                self.get_logger().warn(f"Unknown command: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse command: {e}")

_base_command = np.zeros(3)
needs_reset = False
first_step = True

bool_dig = False
bool_dmp = False
world_point = [0,0,0]
local_point = [0,0,0]
dig_err = False
dmp_err = False
executing_trejectory = False
setup_arm = True
def execute_walk(directon):
    global _base_command
    dir = ''.join(directon)
    direction_map = {
        '1000':'upward',
        '0100':'left',
        '0010':'downward',
        '0001':'right'
        }
    print(f'I\'m moving in {direction_map[dir]} direction, man')
    if direction_map[dir] == 'upward':
        _base_command += np.array([1,0,0])
    elif direction_map[dir] == 'downward':
        _base_command += np.array([-1,0,0])
    elif direction_map[dir] == 'left':
        _base_command += np.array([0,1,0])
    elif direction_map[dir] == 'right':
        _base_command += np.array([0,-1,0])
    else:
        _base_command = np.array([0,0,0])

def execute_dig(): #world_cor
    world_cor = [-2.49, -1.70, 0]
    global bool_dig, world_point, local_point
    print('I\'m Digging the earth, man')
    world.remove_physics_callback("spot_forward")
    #digging!
    prim_spot = stage.GetPrimAtPath(OBJ_PRIM_PATH)
    xform = UsdGeom.Xformable(prim_spot)
    local_to_world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_local_matrix = local_to_world_matrix.GetInverse()
    world_point = Gf.Vec3d(world_cor[0], world_cor[1], world_cor[2]) # Dig point in world coordinates x, y, z
    local_point = world_to_local_matrix.Transform(world_point)
    bool_dig = True
    pass

def execute_dump():
    global bool_dmp
    print('I\'m dumping the soil, man')
    world.remove_physics_callback("spot_forward")
    #dumping!
    bool_dmp = True
    pass

def execute_rotcw():
    global _base_command
    print('I\'m rotating clockwise, man')
    _base_command += np.array([0,0,-1])
    pass

def execute_rotccw():
    global _base_command
    print('I\'m rotating anti-clockwise, man')
    _base_command += np.array([0,0,1])
    pass

# === FILE PATHS ===
ENVIRONMENT_USD_PATH = "./Environment_with_obstacles.usd"
OBJECT_USD_PATH = "./spot_arm_plastic_shovel.usd" #spot USD
ENV_PRIM_PATH = "/World/Environment"
OBJ_PRIM_PATH = "/World/spot"
CAMERA_PRIM_PATHL = "/World/spot/body/frontright_fisheye"
CAMERA_PRIM_PATHR = "/World/spot/body/frontright_fisheye"
OBJECT_POSITION = (-3.4, -2, 0.7)

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
physics_dt = 1 / 200.0
render_dt = 1 / 60.0
world = World(stage_units_in_meters=1.0,physics_dt=physics_dt, rendering_dt=render_dt)

# Get current stage
stage = omni.usd.get_context().get_stage()


# Modify or create PhysicsScene at correct path
physics_scene_path = "/World/physicsScene"
# Get or define the PhysicsScene prim
physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
if not physics_scene_prim.IsValid():
    physics_scene_prim = stage.DefinePrim(physics_scene_path, "PhysicsScene")

# Add the attribute manually
attr = physics_scene_prim.CreateAttribute("physxEnableGpuDynamics", Sdf.ValueTypeNames.Bool)
attr.Set(True)

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

policy_path = "./spot_arm/models/spot_arm_policy.pt"
policy_params_path = "./spot_arm/params/env.yaml"

_spot = SpotArmFlatTerrainPolicy(
    prim_path="/World/spot",
    name="spot",
    usd_path=OBJECT_USD_PATH,
    policy_path=policy_path,
    policy_params_path=policy_params_path,
    position=np.array(OBJECT_POSITION),
)

def on_physics_step(step_size) -> None:
    global first_step, needs_reset  # <-- Add this line
    if first_step:
        _spot.initialize()
        first_step = False
    elif needs_reset:
        world.reset(True)
        needs_reset = False
        first_step = True
    else:
        _spot.forward(step_size, _base_command)

world.add_physics_callback("spot_forward", callback_fn=on_physics_step)

spot.initialize()
controller = spot.get_articulation_controller()

def stable_joint():
    current_positions_ = spot.get_joint_positions()
    current_positions = {name: float(current_positions_[i]) for i, name in enumerate(spot.dof_names)}
    joint_targets = current_positions
    for joint_name, angle in joint_targets_.items():
        try:
            idx = spot.get_dof_index(joint_name)
            joint_targets[idx] = angle
        except Exception as e:
            print(f"Warning: Joint '{joint_name}' not found. {e}")
    return np.array([joint_targets[name] for name in spot.dof_names])

# Optional: Set Initial Joint Positions
joint_targets_ = {
    "fl_hx": 0.0, "fr_hx": 0.0, "hl_hx": 0.0, "hr_hx": 0.0,
    "fl_hy": 0.5, "fr_hy": 0.5, "hl_hy": 0.8, "hr_hy": 0.8,
    "fl_kn": -1.2, "fr_kn": -1.2, "hl_kn": -1.5, "hr_kn": -1.5,
}

# Initialize ROS2
rclpy.init()
ros_interface = SpotROSInterface(spot, world)
camera_pubL = CameraPublisher(CAMERA_PRIM_PATHL, 'left')
camera_pubR = CameraPublisher(CAMERA_PRIM_PATHR, 'right')

# Combined Executor
executor = MultiThreadedExecutor()
executor.add_node(ros_interface)
executor.add_node(camera_pubL)
executor.add_node(camera_pubR)

ros_thread = Thread(target=executor.spin, daemon=True)
ros_thread.start()

# Arm trajectory class
import arm_trajectory
spot_arm = arm_trajectory.spot_arm_trajectory()

# Main Loop
while simulation_app.is_running():
    ros_interface.publish_state()
    camera_pubL.publish()
    camera_pubR.publish()
    world.step(render=True)

    if setup_arm is True:
        # Setup arm trajectory
        spot_arm.setup(spot) # Setup with articulation as argument
        setup_arm = False

    if bool_dig:
        bool_dig = False
        dig_err = False
        spot.set_joint_positions(stable_joint())
        print("\nDigging position: " + str(world_point) + "\n")
        trajectory = spot_arm.setup_cspace_trajectory(local_point, "dig") # Create trajectory, "dig" for digging
        if trajectory is False:
            if not world.physics_callback_exists("spot_forward"):
                world.add_physics_callback("spot_forward", callback_fn=on_physics_step)
            print("\nCannot reach digging position " + str(world_point) + "\n")
            dig_err = True
        else:
            executing_trejectory = True

    if bool_dmp:
        bool_dmp = False
        dmp_err = False
        spot.set_joint_positions(stable_joint())
        trajectory = spot_arm.setup_cspace_trajectory(np.array([0, 0, 0]), "dump") # Create trajectory, "dump" for dumping earth
        if trajectory is False:
            print("\nCannot reach dumping position\n")
            if not world.physics_callback_exists("spot_forward"):
                world.add_physics_callback("spot_forward", callback_fn=on_physics_step)
            dmp_err = True
        else:
            executing_trejectory = True

    
    if executing_trejectory: # Check if we want to execute trajectory
        arm_step = spot_arm.update() # Move arm one step
        if arm_step is True: # Check if trajectory is complete
            if not world.physics_callback_exists("spot_forward"):
                world.add_physics_callback("spot_forward", callback_fn=on_physics_step)
            executing_trejectory = False

# Cleanup
simulation_app.close()
ros_interface.destroy_node()
camera_pubL.destroy_node()
camera_pubR.destroy_node()
rclpy.shutdown()