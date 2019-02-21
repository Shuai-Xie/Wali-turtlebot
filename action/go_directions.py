#!/usr/bin/env python
# coding=utf-8
import rospy
from geometry_msgs.msg import Twist
from math import radians

cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)


class GoForward:
    def __init__(self):
        self.move_cmd = Twist()  # Twist is a datatype for velocity
        self.move_cmd.linear.x = 0.2
        self.move_cmd.angular.z = 0  # w = 0

    def run(self):
        r = rospy.Rate(5)  # smooth run
        for i in range(2):
            cmd_vel.publish(self.move_cmd)
            r.sleep()


class GoBack:
    def __init__(self):
        self.move_cmd = Twist()
        self.move_cmd.linear.x = -0.2
        self.move_cmd.angular.z = 0

    def run(self):
        r = rospy.Rate(10)
        for i in range(3):
            cmd_vel.publish(self.move_cmd)
            r.sleep()


class GoLeft:
    def __init__(self):
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = radians(45)

    def run(self):
        r = rospy.Rate(10)
        for i in range(3):
            cmd_vel.publish(self.move_cmd)
            r.sleep()


class GoRight:
    def __init__(self):
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = radians(-45)

    def run(self):
        r = rospy.Rate(10)
        for i in range(3):
            cmd_vel.publish(self.move_cmd)
            r.sleep()


def shutdown():
    print 'stop!'
    cmd_vel.publish(Twist())
    rospy.sleep(1)
