import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import sys
import os

# === 确保 RAFT-Stereo 可导入 ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAFT-Stereo/core'))
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.bridge = CvBridge()

        # Subscribe to camera topics
        self.sub_left = self.create_subscription(Image, '/front_left/image_raw', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/front_right/image_raw', self.right_callback, 10)

        # Publishers
        self.point_pub = self.create_publisher(PointStamped, '/desired_world_point', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

        # Buffers
        self.left_img = None
        self.right_img = None

        # Load RAFT-Stereo model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RAFTStereo()
        ckpt_path = '/home/rllab/Desktop/25P24/Shoveling-earth-with-the-Boston-Dynamics-Spot-robot/proj_sim1/RAFT-Stereo/models/raftstereo-middlebury.pth'
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.get_logger().info("✅ RAFT-Stereo model loaded.")

        # Camera parameters (make sure these match your setup!)
        self.focal_length = 700  # pixels
        self.baseline = 0.1      # meters

    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_compute()

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_compute()

    def try_compute(self):
        if self.left_img is None or self.right_img is None:
            return

        left_norm = (self.left_img.astype(np.float32) / 255.0)
        right_norm = (self.right_img.astype(np.float32) / 255.0)

        left_tensor = torch.from_numpy(left_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        right_tensor = torch.from_numpy(right_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            padder = InputPadder(left_tensor.shape)
            left_pad, right_pad = padder.pad(left_tensor, right_tensor)
            _, disparity = self.model(left_pad, right_pad, iters=20, test_mode=True)
            disparity = padder.unpad(disparity).squeeze().cpu().numpy()

        depth_map = (self.focal_length * self.baseline) / (disparity + 1e-6)

        # Step 1: Extract blue sandbox
        hsv = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        sandbox_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Step 2: Extract red balls
        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_in_sandbox = cv2.bitwise_and(red_mask, sandbox_mask)

        # Step 3: Find closest red point
        min_depth = np.inf
        best_point = None
        ys, xs = np.where(red_in_sandbox > 0)
        for x, y in zip(xs, ys):
            d = depth_map[y, x]
            if 0 < d < min_depth:
                min_depth = d
                best_point = (x, y, d)

        debug_img = self.left_img.copy()
        if best_point is not None:
            cx, cy, cz = best_point
            world_x = (cx - self.left_img.shape[1] / 2) * cz / self.focal_length
            world_y = (cy - self.left_img.shape[0] / 2) * cz / self.focal_length
            world_z = cz

            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "camera_link"
            point_msg.point.x = float(world_x)
            point_msg.point.y = float(world_y)
            point_msg.point.z = float(world_z)
            self.point_pub.publish(point_msg)

            self.get_logger().info(f"Published target point: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")

            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
        else:
            self.get_logger().warn("No valid red target found.")

        # Draw sandbox outline
        contours, _ = cv2.findContours(sandbox_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

        self.left_img = None
        self.right_img = None

# def main(args=None):
#     rclpy.init(args=args)
#     node = StereoDepthNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
