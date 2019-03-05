#!/usr/bin/env python

import numpy as np  # working with vectors and matrices
import math
import sympy
import time
import rospy
import symengine
import tf2_ros

import camera_model as cam
import ekf_model

from apriltags2_ros.msg import AprilTagDetectionArray
from numpy.linalg import inv
from tf.transformations import quaternion_from_euler  # defined q = (x, y, z, w)
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from ekf.msg import callback_data
from pyquaternion import Quaternion  # defined q = (w, x, y, z)
# testing
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo


def callback1_timer(msg):
    global testcam1

    testcam1.set_time_offset(msg.header.stamp.secs, msg.header.stamp.nsecs)


def callback2_timer(msg):
    global testcam2

    testcam2.set_time_offset(msg.header.stamp.secs, msg.header.stamp.nsecs)


def callback1(
        msg):  # the purpose of this function is to recieve the position data of all seen tags from camera 1, create the h, H and R Matrices according to the seen tags and give them to ekf for further processing while publishing the tf-frame of it's camera
    global testcam1, ekf, pub_tf_cam1, no_tag1

    t1 = time.time()
    testcam1.callcam(ekf, pub_tf_cam1, no_tag1, msg)
    # do stuff
    elapsed1 = time.time() - t1
    print('time elapsed callback1 = ' + str(elapsed1))

# print ('cam1' + str(testcam1.compensate_cam_time(msg.header.stamp.secs,msg.header.stamp.nsecs)))


def callback2(msg):  # sole purpose of this function is to recieve the position data of all seen tags from camera 2

    global testcam2, ekf, pub_tf_cam2, no_tag2
    t2 = time.time()
    testcam2.callcam(ekf, pub_tf_cam2, no_tag2, msg)
    elapsed2 = time.time() - t2
    print('time elapsed callback2 = ' + str(elapsed2))

# print ('cam2' + str(testcam2.compensate_cam_time(msg.header.stamp.secs,msg.header.stamp.nsecs)))


def callback3(msg):
    global testcam3, ekf, pub_tf_cam3, no_tag3

    testcam3.callcam(ekf, pub_tf_cam3, no_tag3, msg)


def callback4(msg):
    global testcam4, ekf, pub_tf_cam4, no_tag4
    t4 = time.time()
    testcam4.callcam(ekf, pub_tf_cam4, no_tag4, msg)
    # do stuff
    elapsed4 = time.time() - t4
    print('time elapsed callback4 = ' + str(elapsed4))

def callback5(msg):
    global testcam5, ekf, pub_tf_cam5, no_tag5

    testcam5.callcam(ekf, pub_tf_cam5, no_tag5, msg)


def callback6(msg):
    global testcam6, ekf, pub_tf_cam6, no_tag6
    #t6 = time.time()
    testcam6.callcam(ekf, pub_tf_cam6, no_tag6, msg)
    #elapsed6 = time.time() - t6
    #print('time elapsed callback6 = ' + str(elapsed6))

# initiating publishers
pub_ekf_output = rospy.Publisher('/ekf_position', PoseWithCovarianceStamped, queue_size=1)
pub_tf_object = tf2_ros.TransformBroadcaster()
pub_tf_cam1 = tf2_ros.TransformBroadcaster()
pub_tf_cam2 = tf2_ros.TransformBroadcaster()
pub_tf_cam3 = tf2_ros.TransformBroadcaster()
pub_tf_cam4 = tf2_ros.TransformBroadcaster()
pub_tf_cam5 = tf2_ros.TransformBroadcaster()
pub_tf_cam6 = tf2_ros.TransformBroadcaster()

# initiating globals for all cameras
# x_hat_0 = np.array([0.25, 0.45, -0.8, 0, 0, 0]).reshape(6, 1)
x_hat_0 = np.array([2.0, 0.8, 0.8, 0, 0, 0]).reshape(6, 1)
skalar = 100
P_mat_0 = np.diag([10 * skalar, 10 * skalar, 10 * skalar, 0.5, 0.5, 0.5])
process_noise_pos = 0.3
process_noise_angle = 0.1
Q_mat = np.diag([process_noise_pos ** 2,
                 process_noise_pos ** 2,
                 process_noise_pos ** 2,
                 process_noise_angle ** 2,
                 process_noise_angle ** 2,
                 process_noise_angle ** 2])
ekf = ekf_model.EkfLocalization(x_hat_0, P_mat_0, Q_mat, pub_ekf_output, pub_tf_object)

# initiating camera 1
R_cam1 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise

testcam1 = cam.camera(1, 0.21, (1.51 - 0.025), 0.82, 0, -0.09, 0, R_cam1, 
                      z_scale=1 / 1.09, x_scale=(1+0.04/1.2))


no_tag1 = 1  # init "no tag" counter

# initiating camera 2
R_cam2 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise
testcam2 = cam.camera(2, 0.20, (0.555 - 0.05), 0.81, 0, -0.08, -0.042, R_cam2,
                      z_scale=1 / 1.085, x_scale=(1+0.04/1.2))  # new pos: x = 0.062, y = 0.41, z = -0.76

no_tag2 = 1  # init "no tag" counter

# initiating camera 3
R_cam3 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise
testcam3 = cam.camera(3, 0, 0, 0, 0, 0, 0, R_cam3,
                      z_scale=1 / 1, x_scale=(1+0.04/1.2))
no_tag3 = 1  # init "no tag" counter

# initiating camera 4
R_cam4 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise
testcam4 = cam.camera(4, 3.985, 0.515, 0.805, 0, -0.044, (math.pi - 0.065), R_cam4,
                      z_scale=1 / 1.1, x_scale=(1+0.04/1.2))  # position might be a bit off, it has been measurured in relation 
no_tag4 = 1  # init "no tag" counter

# initiating camera 5
R_cam5 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise
# testcam5 = cam.camera(5, 0.205, 0.89, 0.77, 0, 0, 0, R_cam5,z_scale=1/1) # positions not right yet
testcam5 = cam.camera(5, 0, 0, 0, 0, 0, math.pi, R_cam5,
                      z_scale=1 / 1, x_scale=(1+0.04/1.2))  # positions not right yet
no_tag5 = 1  # init "no tag" counter

# initiating camera 6
R_cam6 = np.diag(np.array([(0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2,
                           (0.1) ** 2, (0.1) ** 2, (0.5) ** 2]))  # measurement noise

testcam6 = cam.camera(6, 3.985, 1.51, 0.81, 0, -0.11, (math.pi + 0.015), R_cam6, z_scale=1 / 1.09, x_scale=(1+0.04/1.2))
no_tag6 = 1  # init "no tag" counter


def main():
    rospy.init_node('ext_auv_localization')
    # rospy.Subscriber("/usb_cam1/camera_info", CameraInfo, callback1_timer, queue_size=1)
    # rospy.Subscriber("/usb_cam2/camera_info", CameraInfo, callback2_timer, queue_size=1)
    #rospy.Subscriber("/tag_detections1", AprilTagDetectionArray, callback1, queue_size=1)
    rospy.Subscriber("/tag_detections2", AprilTagDetectionArray, callback2, queue_size=1)
    # rospy.Subscriber("/tag_detections3", AprilTagDetectionArray, callback3, queue_size=1)
    #rospy.Subscriber("/tag_detections4", AprilTagDetectionArray, callback4, queue_size=1)
    # rospy.Subscriber("/tag_detections5", AprilTagDetectionArray, callback5, queue_size=1)
    #rospy.Subscriber("/tag_detections6", AprilTagDetectionArray, callback6, queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
