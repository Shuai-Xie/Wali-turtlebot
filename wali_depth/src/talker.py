# coding=utf-8
import rospy
from wali.srv import RGBD_Image
from cv_bridge import CvBridge
import numpy as np
from math import radians
from geometry_msgs.msg import Twist

actions = {
    'forward': 1,
    'left': 2,
    'right': 3,
    'back': 4,
    'stop': 0
}

move_cmd = Twist()
move_cmd.linear.x = 0
move_cmd.angular.z = 0

std_X = 0.3
std_Z = 25
a = 0.02  # accelerate speed


def set_move_cmd(action=0):
    if action == 1:  # forward
        move_cmd.linear.x = std_X
        move_cmd.angular.z = 0
    elif action == 2:  # left
        move_cmd.linear.x = 0
        move_cmd.angular.z = radians(std_Z)
    elif action == 3:  # right
        move_cmd.linear.x = 0
        move_cmd.angular.z = radians(-std_Z)
    elif action == 4:  # back
        move_cmd.linear.x = -std_X
        move_cmd.angular.z = 0
    else:  # stop
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0
    return move_cmd


def talker():
    rospy.init_node('talker', anonymous=True)
    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)  # velocity params
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.wait_for_service('get_rgbd_image')
        try:
            get_rgbd_image = rospy.ServiceProxy('get_rgbd_image', RGBD_Image)
            bridge = CvBridge()

            # define a func in a func
            def get_min_depth(lr=False):
                while True:
                    res = get_rgbd_image(start=True)
                    depth_msg = res.depth
                    if depth_msg.data:  # cus some depth_msg=None
                        depth = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
                        depth[depth == 0] = 65535  # only use valid depth point
                        if lr:
                            return np.min(depth[:, :320]), np.min(depth[:, 320:])  # choose roi
                        else:
                            return np.min(depth[:, 160:480])  # roi size = 320

            def speedup():
                global move_cmd
                while move_cmd.linear.x < std_X:
                    print 'speed up'
                    move_cmd.linear.x += a
                    cmd_vel.publish(move_cmd)
                    r.sleep()
                print 'uniform'

            def slowdown():
                global move_cmd
                while move_cmd.linear.x > 0:
                    print 'slow down'
                    move_cmd.linear.x -= a
                    cmd_vel.publish(move_cmd)
                    r.sleep()
                print 'stop'

            SAFE_DISTANCE = 600  # should be distance to object, not ground
            while True:
                mid_d = get_min_depth()
                # go back
                if mid_d == 65535:  # crash
                    move_cmd = set_move_cmd(action=actions['back'])
                    while get_min_depth() < SAFE_DISTANCE:
                        cmd_vel.publish(move_cmd)
                        print 'back'
                        r.sleep()
                # go forward
                elif mid_d > SAFE_DISTANCE:
                    speedup()  # smoothly speed up
                    move_cmd = set_move_cmd(action=actions['forward'])
                    while get_min_depth() > SAFE_DISTANCE:
                        cmd_vel.publish(move_cmd)
                        print 'forward'
                        print move_cmd.linear.x, move_cmd.angular.z
                        r.sleep()
                # choose left or right
                else:
                    slowdown()  # smoothly slow down
                    left_d, right_d = get_min_depth(lr=True)
                    if left_d > right_d:
                        print 'left', left_d, '>', right_d
                        move_cmd = set_move_cmd(action=actions['left'])
                        while get_min_depth() < SAFE_DISTANCE:
                            cmd_vel.publish(move_cmd)
                            print 'left'
                            print move_cmd.linear.x, move_cmd.angular.z
                            r.sleep()
                    else:
                        print 'right', left_d, '<', right_d
                        move_cmd = set_move_cmd(action=actions['right'])
                        while get_min_depth() < SAFE_DISTANCE:
                            cmd_vel.publish(move_cmd)
                            print 'right'
                            print move_cmd.linear.x, move_cmd.angular.z
                            r.sleep()

                # todo: delemma, keep left until find a way out
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
