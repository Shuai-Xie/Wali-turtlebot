#!/usr/bin/env python
# coding=utf-8
import rospy
import sys
from std_msgs.msg import Int8
from action.go_directions import GoForward, GoLeft, GoRight, GoBack, shutdown

sys.path.append('/home/itx/wali_ws/src/wali')  # cmd exe

forward = GoForward()
left = GoLeft()
right = GoRight()
back = GoBack()


def callback(data):
    #  action space
    direction = data.data
    if direction == 1:
        rospy.loginfo('forward')
        forward.run()
    if direction == 2:
        rospy.loginfo('left')
        left.run()
    if direction == 3:
        rospy.loginfo('right')
        right.run()
    if direction == 4:
        back.run()
        rospy.loginfo('back')
    if direction == 0:
        shutdown()
        rospy.loginfo('stop')


def listener():
    rospy.init_node('direction_listener', anonymous=True)
    rospy.Subscriber('direction_topic', Int8, callback)
    rospy.spin()  # keeps python from exiting until this node is stopped


if __name__ == '__main__':
    listener()
