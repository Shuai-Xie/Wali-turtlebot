# coding=utf-8
import os
import shutil
import cv2
import rospy
from cv_bridge import CvBridge
from wali.srv import RGBD_Image


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
    rospy.wait_for_service('get_rgbd_image')
    try:
        get_rgbd_image = rospy.ServiceProxy('get_rgbd_image', RGBD_Image)  # get the func in client
        root = 'kinect'
        RECORD_FLAG = False
        RECORD_CNT = 0
        cnt = 0
        bridge = CvBridge()
        while True:
            res = get_rgbd_image(start)
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
                    print cnt
                    cnt += 1
                # receive cmd
                cmd = cv2.waitKey(200)
                if cmd == ord('r'):
                    RECORD_FLAG = not RECORD_FLAG
                    if RECORD_FLAG:
                        cnt = 0
                        print 'record begin'
                        # mkdir
                        mk_record_dir(root + str(RECORD_CNT))
                    else:
                        print 'record stop'
                        RECORD_CNT += 1  # prepare for next record
                if cmd == ord('q'):
                    print 'exit'
                    break

    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


if __name__ == '__main__':
    rgbd_client()
