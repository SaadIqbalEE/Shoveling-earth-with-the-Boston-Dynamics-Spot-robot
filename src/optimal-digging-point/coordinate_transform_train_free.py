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

# RAFT-Stereo Path Addition
CKPT_PATH = '/home/rllab/Desktop/25P24/Shoveling-earth-with-the-Boston-Dynamics-Spot-robot/proj_sim1/RAFT-Stereo/models/raftstereo-middlebury.pth'

raft_stereo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RAFT-Stereo/core'))
sys.path.insert(0, raft_stereo_path)

from raft_stereo import RAFTStereo
from utils.utils import InputPadder
class ExcavationPointFinder(Node):
    def __init__(self):
        super().__init__('excavation_point_finder')
        self.bridge = CvBridge()  # ROS Image to OpenCV

        # ROS subscribers and publishers
        self.sub_left = self.create_subscription(Image, '/front_left/image_raw', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/front_right/image_raw', self.right_callback, 10)
        self.point_pub = self.create_publisher(PointStamped, '/desired_excavation_point', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_excavation_image', 10)

        self.left_img = None
        self.right_img = None

        # Load RAFT-Stereo model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RAFTStereo(self.device)
        self.model.load_state_dict(torch.load(CKPT_PATH))
        self.model.eval()

    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process()

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process()

    def try_process(self):
        if self.left_img is None or self.right_img is None:
            return

        # Compute disparity
        left_tensor = torch.from_numpy(self.left_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        right_tensor = torch.from_numpy(self.right_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        padder = InputPadder(left_tensor.shape)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        with torch.no_grad():
            _, disparity = self.model(left_tensor, right_tensor, iters=20, test_mode=True)
        disp = disparity.squeeze().cpu().numpy()

        # Mask blue sandbox (HSV threshold)
        hsv = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn("No sandbox found!")
            return

        sandbox_contour = max(contours, key=cv2.contourArea)
        sandbox_mask = np.zeros_like(mask)
        cv2.drawContours(sandbox_mask, [sandbox_contour], -1, 255, -1)

        # Within sandbox, find closest nonzero disparity
        masked_disp = np.where(sandbox_mask, disp, 0)
        min_disp = np.max(masked_disp)  # start from maximum, since disparity is inversely proportional to depth
        min_point = None
        for y in range(masked_disp.shape[0]):
            for x in range(masked_disp.shape[1]):
                d = masked_disp[y, x]
                if d > 0 and d < min_disp:
                    min_disp = d
                    min_point = (x, y)

        if min_point is None:
            self.get_logger().warn("No valid excavation point found in sandbox!")
            return

        ###############################################
        # Very Important to have exact value
        ##############################################
        focal_length = 700  # pixel (you need to set actual)
        baseline = 0.1       # meters (you need to set actual)
        depth = (focal_length * baseline) / min_disp

        cx, cy = min_point
        w, h = disp.shape[1], disp.shape[0]
        point_x = (cx - w / 2) * depth / focal_length
        point_y = (cy - h / 2) * depth / focal_length
        point_z = depth

        # Publish point
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "camera_link"
        point_msg.point.x = float(point_x)
        point_msg.point.y = float(point_y)
        point_msg.point.z = float(point_z)
        self.point_pub.publish(point_msg)

        self.get_logger().info(f"Recommended excavation point: ({point_x:.2f}, {point_y:.2f}, {point_z:.2f})")

        # Debug visualization
        debug_img = self.left_img.copy()
        cv2.drawContours(debug_img, [sandbox_contour], -1, (255, 0, 0), 2)
        cv2.circle(debug_img, min_point, 5, (0, 0, 255), -1)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

        # Reset buffers
        self.left_img = None
        self.right_img = None


def main(args=None):
    rclpy.init(args=args)
    node = ExcavationPointFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()