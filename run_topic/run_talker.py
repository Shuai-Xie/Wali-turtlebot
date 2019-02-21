# coding=utf-8
import rospy
from wali.srv import RGBD_Image
from cv_bridge import CvBridge
import numpy as np
from std_msgs.msg import Int8
import cv2

actions = {
    'forward': 1,
    'left': 2,
    'right': 3,
    'back': 4,
    'stop': 0
}


def talker():
    # pub action
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher(name='direction_topic', data_class=Int8, queue_size=10)
    r = rospy.Rate(2)
    while not rospy.is_shutdown():
        rospy.wait_for_service('get_rgbd_image')
        try:
            get_rgbd_image = rospy.ServiceProxy('get_rgbd_image', RGBD_Image)
            SAFE_DISTANCE = 550  # should be distance to object, not ground
            MIN_DISTANCE = 400  # use to scale the roi side
            bridge = CvBridge()
            left_cnt = 0
            right_cnt = 0
            while True:
                # 1.get rgbd
                res = get_rgbd_image(start=True)
                rgb_msg, depth_msg = res.rgb, res.depth
                if rgb_msg.data and depth_msg.data:
                    rgb = bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
                    depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
                    # cv2.imshow('depth', depth)
                    # 2.use depth to cal mid,left,right distance
                    depth[depth == 0] = 65535
                    ROI_side = min(int(640 * 300 / MIN_DISTANCE), 640)
                    roi_left, roi_right = int(320 - ROI_side / 2), int(320 + ROI_side / 2)
                    # print 'ROI', roi_left, roi_right
                    # cv2.rectangle(rgb, (roi_left, 0), (roi_right, 479), color=(0, 0, 255), thickness=5)
                    # cv2.imshow('rgb', rgb)
                    # receive order
                    # cmd = cv2.waitKey(100)
                    # if cmd == ord('q'):
                    #     break
                    mid_roi_d = np.min(depth[:, roi_left:roi_right])

                    # crash, back 3 times and turn right
                    if mid_roi_d == 65535:
                        print 'back'
                        pub.publish(actions['back'])
                        r.sleep()
                        pub.publish(actions['back'])
                        r.sleep()
                        pub.publish(actions['back'])
                        r.sleep()
                        pub.publish(actions['right'])
                        r.sleep()
                        pub.publish(actions['right'])
                        r.sleep()
                        pub.publish(actions['right'])
                        r.sleep()
                        pub.publish(actions['right'])
                        r.sleep()
                        pub.publish(actions['right'])
                        r.sleep()
                    else:
                        # go forward
                        if mid_roi_d > SAFE_DISTANCE:
                            print 'forward', mid_roi_d
                            pub.publish(actions['forward'])
                            r.sleep()
                            MIN_DISTANCE = mid_roi_d
                            left_cnt, right_cnt = 0, 0
                        else:
                            # choose left or right
                            left_roi_d = np.min(depth[:, :int(320 - 0.5 * ROI_side)])
                            right_roi_d = np.min(depth[:, int(320 + 0.5 * ROI_side):])
                            if left_roi_d > right_roi_d:
                                print 'left', left_roi_d, '>', right_roi_d
                                pub.publish(actions['left'])
                                r.sleep()
                                left_cnt += 1
                                if left_cnt > 2:
                                    left_cnt = 0
                            else:
                                print 'right', left_roi_d, '<', right_roi_d
                                pub.publish(actions['right'])
                                r.sleep()
                                right_cnt += 1
                                if right_cnt > 2:
                                    right_cnt = 0

                        # delemma, keep left until find a way out
                        print left_cnt, right_cnt
                        if left_cnt * right_cnt >= 4:  # loop twice
                            while True:
                                res = get_rgbd_image(start=True)
                                depth_msg = res.depth
                                depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
                                depth[depth == 0] = 65535
                                mid_roi_d = np.min(depth[:, roi_left:roi_right])
                                if mid_roi_d < SAFE_DISTANCE:
                                    print 'delemma', mid_roi_d
                                    pub.publish(actions['left'])
                                    r.sleep()
                                else:
                                    break
                            left_cnt = 0
                            right_cnt = 0

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
