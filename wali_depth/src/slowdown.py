# coding=utf-8
import rospy
from geometry_msgs.msg import Twist

"""
three stage to simulate a running process
loop:
    1.speed up to 0.2m/s, a = 0.02
    2.uniform speed 10 times, 2s
    3.slow down to 0, a = -0.02
"""


def talker():
    rospy.init_node('talker', anonymous=True)
    cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)  # velocity params
    r = rospy.Rate(5)
    move_cmd = Twist()
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    a = 0.02  # linear acceleration
    cnt = 0
    while not rospy.is_shutdown():
        # speed up
        while move_cmd.linear.x < 0.2:
            print 'speed up'
            move_cmd.linear.x += a
            cmd_vel.publish(move_cmd)
            r.sleep()

        # uniform speed
        print 'forward'
        cmd_vel.publish(move_cmd)
        cnt += 1
        r.sleep()

        # slow down
        if cnt > 9:
            while move_cmd.linear.x > 0:
                print 'slow down'
                move_cmd.linear.x -= a
                cmd_vel.publish(move_cmd)
                r.sleep()
            print 'stop'
            cnt = 0


if __name__ == '__main__':
    talker()
