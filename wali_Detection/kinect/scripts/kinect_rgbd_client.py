#!/usr/bin/env python3
# coding=utf-8
import os
import shutil

# must from sys
import rospy
from kinect.srv import RGBD_Image

import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# must below to use python3
import cv2
from cv_bridge import CvBridge


def mk_record_dir(root):
    rgb_path = os.path.join(root, 'rgb')
    depth_path = os.path.join(root, 'depth')
    # remake
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)
    os.mkdir(rgb_path)
    os.mkdir(depth_path)


def rgbd_client(start=True):
    rospy.wait_for_service('get_kinect_rgbd_image')
    try:
        get_rgbd_image = rospy.ServiceProxy('get_kinect_rgbd_image', RGBD_Image)  # get the func in client
        root = 'kinect'
        RECORD_FLAG = False
        RECORD_CNT = 0
        cnt = 0
        bridge = CvBridge()
        while not rospy.is_shutdown():
            res = get_rgbd_image(start)

            # wrt kinect_rgbd_server.py, sign of useful img
            if res.rgb.header.seq > 0:
                rospy.loginfo('get RGBD seq: %s', res.rgb.header.seq)
                rgb_msg, depth_msg = res.rgb, res.depth
                # print rgb_msg.height, depth_msg.height
                # change process: 0 0 -> 0 480 -> 480 480
                # rgg, depth, not sync and not ready in the beginning
                if rgb_msg.data and depth_msg.data:  # only use valid rgbd
                    rgb = bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
                    cv2.imshow('rgb', rgb)
                    depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
                    cv2.imshow('depth', depth)
                    if RECORD_FLAG:
                        cv2.imwrite(root + str(RECORD_CNT) + '/rgb/' + str(cnt) + '.jpg', rgb)
                        cv2.imwrite(root + str(RECORD_CNT) + '/depth/' + str(cnt) + '.png', depth)
                        print (cnt)
                        cnt += 1
                    # receive cmd
                    cmd = cv2.waitKey(30)
                    if cmd == ord('r'):
                        RECORD_FLAG = not RECORD_FLAG
                        if RECORD_FLAG:
                            cnt = 0
                            print ('record begin')
                            # mkdir
                            mk_record_dir(root + str(RECORD_CNT))
                        else:
                            print ('record stop')
                            RECORD_CNT += 1  # prepare for next record
                    if cmd == ord('q'):
                        print ('exit')
                        break
            else:
                rospy.loginfo('server kinect is loading...')

    except rospy.ServiceException:
        print("Service call failed")


if __name__ == '__main__':
    rospy.init_node('kinect_rgbd_client')
    rgbd_client()
