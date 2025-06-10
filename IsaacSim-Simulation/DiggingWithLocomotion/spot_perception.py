# This python file is for integration of perception model (for Environment_with_obstacles.usd)


# luckily, the name is fisheye...but it is not fisheye camera
# first let me copy the camera parameter
# python 3.12.7 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

focal_length = 18.14756
focus_distance = 400
nominal_width = 1936
nominal_height = 1216
horizontal_aperture = 20.955
vertical_aperture = 15.2908
width_center = 970.9424438476562
height_center = 600.374817

fx = focal_length * (nominal_width / horizontal_aperture)
fy = focal_length * (nominal_height / vertical_aperture)
cx = width_center
cy = height_center

K = [
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
]

class StereoDisparityPublisher(Node):
    def __init__(self):
        super().__init__('stereo_disparity_publisher')
        self.bridge = CvBridge()

        # subscript left and right image from other matrix
        self.left_sub = Subscriber(self, Image, '/front_cam/left')
        self.right_sub = Subscriber(self, Image, '/front_cam/right')

        # disparity and combined
        self.disparity_pub = self.create_publisher(Image, '/stereo/disparity', 10)
        self.combined_pub = self.create_publisher(Image, '/stereo/combined', 10)

        # StereoSGBM (Traditional Disparity Matching, efficient)
        min_disp = 0
        num_disp = 16 * 5  # multiply of 16
        block_size = 5
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.get_logger().info("StereoDisparityPublisher initialized!")

    def callback(self, left_msg, right_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        combined = cv2.hconcat([left_img, right_img])
        combined_msg = self.bridge.cv2_to_imgmsg(combined, encoding='bgr8')
        self.combined_pub.publish(combined_msg)

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo_matcher.compute(left_gray, right_gray).astype('float32') / 16.0

        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_uint8 = np.uint8(disp_norm)

        disp_msg = self.bridge.cv2_to_imgmsg(disp_uint8, encoding='mono8')
        self.disparity_pub.publish(disp_msg)

        self.get_logger().info("Published disparity map and combined image.")

def main(args=None):
    rclpy.init(args=args)
    node = StereoDisparityPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
