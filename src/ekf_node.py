#!/usr/bin/env python

import numpy as np #working with vectors and matrices
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
from tf.transformations import quaternion_from_euler # defined q = (x, y, z, w)
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from ekf.msg import callback_data
from pyquaternion import Quaternion  # defined q = (w, x, y, z)
#testing
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


def callback1(msg):  # the purpose of this function is to recieve the position data of all seen tags from camera 1, create the h, H and R Matrices according to the seen tags and give them to ekf for further processing while publishing the tf-frame of it's camera
	global testcam1, ekf, pub_tf_cam1, no_tag1
	
	testcam1.callcam(ekf, pub_tf_cam1, no_tag1, msg)


def callback2(msg):  # sole purpose of this function is to recieve the position data of all seen tags from camera 2
	
	global testcam2, ekf, pub_tf_cam2, no_tag2
	
	testcam2.callcam(ekf, pub_tf_cam2, no_tag2, msg)

def callback3(msg):

	global testcam3, ekf, pub_tf_cam3, no_tag3

	testcam3.callcam(ekf, pub_tf_cam3, no_tag3, msg)

def callback4(msg):

	global testcam4, ekf, pub_tf_cam4, no_tag4
	
	testcam4.callcam(ekf, pub_tf_cam4, no_tag4, msg)
	
def callback5(msg):

	global testcam5, ekf, pub_tf_cam5, no_tag5
	
	testcam5.callcam(ekf, pub_tf_cam5, no_tag5, msg)
	
def callback6(msg):

	global testcam6, ekf, pub_tf_cam6, no_tag6
	
	testcam6.callcam(ekf, pub_tf_cam6, no_tag6, msg)
	

# initiating publishers
pub_ekf_output = rospy.Publisher('/ekf_position', PoseWithCovarianceStamped, queue_size = 1)
pub_tf_object = tf2_ros.TransformBroadcaster()
pub_tf_cam1 = tf2_ros.TransformBroadcaster()
pub_tf_cam2 = tf2_ros.TransformBroadcaster()
pub_tf_cam3 = tf2_ros.TransformBroadcaster()
pub_tf_cam4 = tf2_ros.TransformBroadcaster()
pub_tf_cam5 = tf2_ros.TransformBroadcaster()
pub_tf_cam6 = tf2_ros.TransformBroadcaster()

# initiating globals for all cameras
x_hat_0 = np.array([0.25, 0.45, -0.8, 0, 0, 0]).reshape(6, 1)
skalar = 10
P_mat_0 = np.diag([10*skalar, 10*skalar, 10*skalar, 0.5, 0.5, 0.5])
process_noise_pos = 0.03
process_noise_angle = 0.1
Q_mat = np.diag([process_noise_pos ** 2,
 		 process_noise_pos ** 2, 
		 process_noise_pos ** 2, 
		 process_noise_angle ** 2, 
		 process_noise_angle ** 2, 
		 process_noise_angle ** 2])
ekf = ekf_model.EkfLocalization(x_hat_0, P_mat_0, Q_mat, pub_ekf_output, pub_tf_object)

# initiating camera 1
R_cam1 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4)]))*1000 # measurement noise
testcam1 = cam.camera(1, 0.062, 1.59, -0.76, 0, 0, 0, R_cam1) # new pos: x = 0.062, y = 1.59, z = -0.76
# old position = -0.32, 1.525, -0.895
no_tag1 = 1 # init "no tag" counter

# initiating camera 2
R_cam2 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*1000 # measurement noise
testcam2 = cam.camera(2, 0.062, 0.41, -0.76, 0, 0, 0, R_cam2) # new pos: x = 0.062, y = 0.41, z = -0.76
# old position = -0.32, 0.326, -0.895
no_tag2 = 1 # init "no tag" counter

#initiating camera 3
R_cam3 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*10000 # measurement noise
testcam3 = cam.camera(3, 4.205, 0.12, -0.57, 0, 0, 3*math.pi/4, R_cam3)
no_tag3 = 1 # init "no tag" counter

#initiating camera 4
R_cam4 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*10000 # measurement noise
testcam4 = cam.camera(4, 0.11, 1.965, -0.57, 0, 0, -math.pi/4, R_cam4) # position might be a bit off, it has been measurured in relation to cam 1\\ new pos: x = 0.11, y = 1.965, z = -0.57
# old position = -0.26, 1.9, -0.71
no_tag4 = 1 # init "no tag" counter

#initiating camera 5
R_cam5 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*1000 # measurement noise
testcam5 = cam.camera(5, 4.160, 1.595, -0.795, 0, 0, math.pi, R_cam5) # positions not right yet
no_tag5 = 1 # init "no tag" counter

#initiating camera 6
R_cam6 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*1000 # measurement noise
testcam6 = cam.camera(6, 4.16, 0.405, -0.76, 0, 0, math.pi, R_cam6)
no_tag6 = 1 # init "no tag" counter

def main():
	rospy.init_node('ext_auv_localization')
	rospy.Subscriber("/tag_detections1", AprilTagDetectionArray, callback1, queue_size=1)
	rospy.Subscriber("/tag_detections2", AprilTagDetectionArray, callback2, queue_size=1)
	rospy.Subscriber("/tag_detections3", AprilTagDetectionArray, callback3, queue_size=1)
	rospy.Subscriber("/tag_detections4", AprilTagDetectionArray, callback4, queue_size=1)
	rospy.Subscriber("/tag_detections5", AprilTagDetectionArray, callback5, queue_size=1)
	rospy.Subscriber("/tag_detections6", AprilTagDetectionArray, callback6, queue_size=1)
	rospy.spin()


if __name__ == '__main__':
	main()
