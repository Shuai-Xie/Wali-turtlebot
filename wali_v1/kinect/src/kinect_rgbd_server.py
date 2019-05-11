#!/usr/bin/env python
# coding=utf-8
import rospy
from sensor_msgs.msg import Image
from kinect.srv import RGBD_Image, RGBD_ImageResponse  # build on this pkg

# sensor_msgs/Image, std_msgs/Header can be seen in RGBD_ImageResponse class

res = RGBD_ImageResponse()


def rgb_callback(ros_data):  # ros_data, Image type
    res.rgb = ros_data  # '/camera/rgb/image_color' return Type: sensor_msgs/Image


def depth_callback(ros_data):
    res.depth = ros_data  # '/camera/depth/image_raw' return Type: sensor_msgs/Image


def get_rgbd_image(req):
    if req.start:  # request.start
        rospy.Subscriber('/camera/rgb/image_color', Image, rgb_callback)  # set response.rgb
        rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)  # set response.depth

        if res.rgb.header.seq > 0:  # and res.rgb.header.seq > 0: value same
            rospy.loginfo('send RGBD seq: %s', str(res.rgb.header.seq))
        else:
            rospy.loginfo('loading kinect...')

        return res  # RGBD_ImageResponse


def rgbd_server():
    while not rospy.is_shutdown():  # must have, then loginfo works!
        rospy.loginfo('beign rgdb image server!')
        rospy.Service('get_rgbd_image', RGBD_Image, get_rgbd_image)  # return type, service callback function
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node('kinect_rgbd_server')
    rgbd_server()
