#!/usr/bin/env python3
# coding=utf-8
import os
import rospy
from kinect.srv import RGBD_Image
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
from cv_bridge import CvBridge
from yolov3_detect import yolov3_detect

out_path = '/home/nvidia/wali_ws/src/detect/scripts/output.png'


def rgbd_client(start=True):
    rospy.wait_for_service('get_kinect_rgbd_image')
    try:
        get_rgbd_image = rospy.ServiceProxy('get_kinect_rgbd_image', RGBD_Image)  # get the func in client
        bridge = CvBridge()
        while not rospy.is_shutdown():
            res = get_rgbd_image(start)
            if res.rgb.header.seq > 0:  # valid frame
                rospy.loginfo('get RGBD seq: %s', res.rgb.header.seq)
                rgb_msg = res.rgb
                if rgb_msg.data:  # only use valid rgbd
                    rgb = bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
                    res = yolov3_detect(img=rgb)
                    cv2.imshow('res', res)
                    cmd = cv2.waitKey(10)
                    if cmd == ord('q'):
                        print('exit')
                        break
            else:
                rospy.loginfo('server kinect is loading...')

    except rospy.ServiceException:
        print("Service call failed")


if __name__ == '__main__':
    rospy.init_node('kinect_detect')
    rgbd_client()
