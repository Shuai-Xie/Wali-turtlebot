#!/usr/bin/env python
import rospy
from kobuki_msgs.msg import SensorState


class kobuki_battery:
    kobuki_base_max_charge = 160  # max battery charge

    def __init__(self):
        rospy.init_node("kobuki_battery", disable_signals=True)  # this param can be used to close node

        # monitor Kobuki's power and charging status. (low battery, charging, not charging etc)
        self.kobuki_monitor = rospy.Subscriber("/mobile_base/sensors/core", SensorState,
                                               self.SensorPowerEventCallback)
        rospy.spin()

    def SensorPowerEventCallback(self, data):  # write callback function inner the class
        rospy.loginfo("Kobuki's battery is now: " + str(
            round(float(data.battery) / float(self.kobuki_base_max_charge) * 100)) + "%")
        if int(data.charger) == 0:
            rospy.loginfo("Not charging at docking station")
        else:
            rospy.loginfo("Charging at docking station")
        # self.kobuki_monitor.unregister()  # cancel subscription
        rospy.signal_shutdown(reason='only look once')  # close node


if __name__ == '__main__':
    try:
        kobuki_battery()
    except rospy.ROSInterruptException:
        rospy.loginfo("exception")
