# from two camera input, giving the 3-D coordinate
# behave as a ROS topic
# someone modify
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
ckpt_path = '/home/rllab/Desktop/25P24/Shoveling-earth-with-the-Boston-Dynamics-Spot-robot/proj_sim1/RAFT-Stereo/models/raftstereo-middlebury.pth'

class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node') # The initialize logic
        self.bridge = CvBridge()

        # Subscribe to left/right camera
        self.sub_left = self.create_subscription(Image, '/front_left/image_raw', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/front_right/image_raw', self.right_callback, 10)

        # Publisher to send computed world point
        self.point_pub = self.create_publisher(PointStamped, '/desired_world_point', 10)

        # Buffers
        self.left_img = None
        self.right_img = None

        # Stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=7,
            P1=8 * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2
        )

    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_compute()

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_compute()

    def try_compute(self):
        # left camera is use to projected
        if self.left_img is not None and self.right_img is not None:
            # Compute disparity
            disparity = self.stereo.compute(self.left_img, self.right_img).astype(np.float32) / 16.0

            # Example: select center pixel
            h, w = disparity.shape
            cx, cy = w // 2, h // 2
            disp_value = disparity[cy, cx]
            if disp_value <= 0:
                self.get_logger().warn("Invalid disparity value at center pixel")
                return

            # Focal length and baseline (you must set these correctly!)
            focal_length = 700  # pixels
            baseline = 0.1       # meters

            depth = (focal_length * baseline) / disp_value

            # Backproject to 3D point in camera frame
            point_x = (cx - w/2) * depth / focal_length
            point_y = (cy - h/2) * depth / focal_length
            point_z = depth

            # Publish point (THis is important)
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "world"  # or "camera_link" depending on your system
            point_msg.point.x = float(point_x)
            point_msg.point.y = float(point_y)
            point_msg.point.z = float(point_z)
            self.point_pub.publish(point_msg)

            self.get_logger().info(f"Published world point: ({point_x:.2f}, {point_y:.2f}, {point_z:.2f})")

            # Reset for next pair
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

