#!/usr/bin/env python
# coding=utf-8
import rospy
from sensor_msgs.msg import Image
from wali.srv import RGBD_Image, RGBD_ImageResponse

# sensor_msgs/Image, std_msgs/Header can be seen in RGBD_ImageResponse class

res = RGBD_ImageResponse()


def rgb_callback(ros_data):  # ros_data, Image type
    res.rgb = ros_data  # '/camera/rgb/image_color' return Type: sensor_msgs/Image


def depth_callback(ros_data):
    res.depth = ros_data  # '/camera/depth/image_raw' return Type: sensor_msgs/Image
    # rospy.loginfo('send depth', res.depth.header.frame_id)


def get_rgbd_image(req):
    if req.start:  # request.start
        rospy.Subscriber('/camera/rgb/image_color', Image, rgb_callback)  # set response.rgb
        rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)  # set response.depth
        # if res.rgb.data and res.depth.data: (not sync rgb and depth, so can't set this if)
        # Error processing request: not all arguments converted during string formatting
        # rospy.loginfo('send rgb', res.rgb.header.frame_id)
        return res  # RGBD_ImageResponse
    # else:
    #     return None


def rgbd_server():
    rospy.init_node('rgbd_image_server')
    rospy.Service('get_rgbd_image', RGBD_Image, get_rgbd_image)  # return type, service callback function
    rospy.spin()


if __name__ == "__main__":
    rgbd_server()
