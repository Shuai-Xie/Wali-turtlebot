#!/usr/bin/env python
# coding=utf-8
import rospy
from cv_bridge import CvBridge
from wali.srv import Action_order, Action_orderResponse
import numpy as np

actions = {
    'forward': 1,
    'left': 2,
    'leftx2': 22,
    'right': 3,
    'back': 4,
    'stop': 0
}

bridge = CvBridge()
SAFE_DISTANCE = 500
MIN_DISTANCE = 400  # use to scale the roi side
left_cnt = 0
right_cnt = 0

res = Action_orderResponse()


def get_run_order(req):  # Image type
    depth_msg = req.depth_msg
    global MIN_DISTANCE, left_cnt, right_cnt
    if depth_msg.data:
        depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        depth[depth == 0] = 65535
        ROI_side = min(int(640 * 300 / MIN_DISTANCE), 640)
        roi_left, roi_right = int(320 - ROI_side / 2), int(320 + ROI_side / 2)
        mid_roi_d = np.min(depth[:, roi_left:roi_right])
        if mid_roi_d == 65535:
            print 'back'
            action = actions['back']
        else:
            if mid_roi_d > SAFE_DISTANCE:
                MIN_DISTANCE = mid_roi_d
                print 'forward', mid_roi_d
                action = actions['forward']
            else:
                if left_cnt == right_cnt and left_cnt * right_cnt >= 4:
                    left_cnt, right_cnt = 0, 0
                    print 'delemma'
                    action = actions['leftx2']
                else:
                    # choose left or right
                    left_roi_d = np.min(depth[:, :int(320 - 0.5 * ROI_side)])
                    right_roi_d = np.min(depth[:, int(320 + 0.5 * ROI_side):])
                    if left_roi_d > right_roi_d:
                        left_cnt += 1
                        print 'left', left_roi_d, '>', right_roi_d
                        action = actions['left']
                    else:
                        right_cnt += 1
                        print 'right', left_roi_d, '<', right_roi_d
                        action = actions['right']
        res.action_order = action
        return res
    else:
        return None


def run_server():
    rospy.init_node('run_server')
    rospy.Service('get_run_order', Action_order, get_run_order)
    rospy.spin()


if __name__ == '__main__':
    run_server()
