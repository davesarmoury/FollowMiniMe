#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from threading import Lock

twist_count = 10 #How many twists to average.  More will make smoother motion, but also make the platform response slower
lock = Lock()

def callback(data):
    global twists

    lock.acquire()
    twists.pop(0)
    twists.append(data)
    lock.release()

def avg_twist():
    global twists

    x_avg = 0.0
    yaw_avg = 0.0

    lock.acquire()
    for msg in twists:
        x_avg = x_avg + msg.linear.x
        yaw_avg = yaw_avg + msg.angular.z
    lock.release()

    x_avg = x_avg / twist_count
    yaw_avg = yaw_avg / twist_count

    avg = Twist()
    avg.linear.x = x_avg
    avg.angular.z = yaw_avg

    return avg

def main():
    global twists
    twists = [Twist()] * twist_count

    rospy.init_node('cmd_vel_smoother', anonymous=True)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/tracker_cmd_vel", Twist, callback)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(avg_twist())
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass