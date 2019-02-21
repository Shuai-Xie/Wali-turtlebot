import rospy
from wali.srv import RGBD_Image, Action_order
from std_msgs.msg import Int8


def run_client():
    rospy.init_node('run_client')
    pub = rospy.Publisher(name='direction_topic', data_class=Int8, queue_size=10)
    r = rospy.Rate(1)
    rospy.wait_for_service('get_rgbd_image')
    try:
        get_rgbd_image = rospy.ServiceProxy('get_rgbd_image', RGBD_Image)
        while True:
            rgbd_res = get_rgbd_image(start=True)
            rospy.wait_for_service('get_run_order')
            try:
                get_run_order = rospy.ServiceProxy('get_run_order', Action_order)
                action_res = get_run_order(rgbd_res.depth)
                pub.publish(action_res.action_order)
                print 'action', action_res.action_order
                r.sleep()
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


if __name__ == '__main__':
    try:
        run_client()
    except rospy.ROSInterruptException:
        pass
