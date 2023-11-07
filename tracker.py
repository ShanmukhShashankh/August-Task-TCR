#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker")
        self.subscriber = self.create_subscription(Image, "/ball_pub", self.image_callback, 10)
        self.ball_publisher = self.create_publisher(PoseStamped, '/ball_location', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Tracker started!")
    

    def image_callback(self, ros_msg):
        cv_img = self.bridge.imgmsg_to_cv2(ros_msg, "bgr8")
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        colorLower = (70, 150, 49)
        colorUpper = (87, 255, 255)
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)
        ball_x, ball_y = self.find_ball_location(mask_img)
        self.draw_ball_contours(cv_img)
        ball_pose = PoseStamped()
        ball_pose.header = ros_msg.header
        ball_pose.pose.position.x = float(ball_x)
        ball_pose.pose.position.y = float(ball_y)
        ball_pose.pose.position.z = 0.0
        ball_pose.pose.orientation.w = 1.0
        self.ball_publisher.publish(ball_pose)


    def find_ball_location(self, masked_image):
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        self.contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ball_x = 400
        ball_y = 300

        if self.contours:
            largest_contour = max(self.contours, key=cv2.contourArea)
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                ball_x = int(M["m10"] / M["m00"])
                ball_y = int(M["m01"] / M["m00"])

        return ball_x, ball_y


    def get_contour_center(self, contour):
        M = cv2.moments(contour)
        cx = -1
        cy = -1
        if (M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        return cx, cy


    def draw_ball_contours(self, rgb_img):
        black_img = np.zeros(rgb_img.shape, 'uint8')
        for c in self.contours:
            area = cv2.contourArea(c)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            if (area > 5000):
                cv2.drawContours(rgb_img, [c], -1, (255,0,255), 2)
                cx, cy = self.get_contour_center(c)
                cv2.circle(rgb_img, (cx,cy), (int)(radius), (0,255,255), 3)
                cv2.circle(black_img, (cx,cy), (int)(radius), (0,255,255), 3)
                cv2.circle(black_img, (cx,cy), 5, (150,0,255), -1)
        cv2.imshow("Ball Tracking", rgb_img)
        cv2.imshow("Black Background Tracking", black_img)
        cv2.waitKey(3)



def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()