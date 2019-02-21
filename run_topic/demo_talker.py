#!/usr/bin/env python
# coding=utf-8
import rospy
from std_msgs.msg import Int8


def talker():
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher(name='direction_topic', data_class=Int8, queue_size=5)  # 小车前进方向
    while not rospy.is_shutdown():
        direction = input('input a direction: 1-forward, 2-left, 3-right, 4-back, 0-stop:')
        pub.publish(direction)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
