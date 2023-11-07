#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class FramePublisher(Node):
    def __init__(self):
        super().__init__("frame_publisher")
        self.get_logger().info("Frame publisher started!")
        self.pub = self.create_publisher(Image, "/ball_pub", 10)
        self.bridge = CvBridge()

    def publishVideo(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 800)
        cap.set(4, 600)
        while True:
            ret, frame = cap.read()
            if ret:
                ros_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.pub.publish(ros_msg)
            else:
                break
    

def main(args=None):
    try:
        rclpy.init(args=args)
        node = FramePublisher()
        node.publishVideo()
    except rclpy.exceptions.ROSInterruptException:
        rclpy.shutdown()