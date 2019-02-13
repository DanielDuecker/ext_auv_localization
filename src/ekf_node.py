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


def callback1(msg):  # the purpose of this function is to recieve the position data of all seen tags from camera 1, create the h, H and R Matrices according to the seen tags and give them to ekf for further processing while publishing the tf-frame of it's camera
	global testcam1, ekf, pub_tf_cam1, no_tag1
	
	testcam1.callcam(ekf, pub_tf_cam1, no_tag1, msg)


def callback2(msg):  # sole purpose of this function is to recieve the position data of all seen tags from camera 2
	
	global testcam2, ekf, pub_tf_cam2, no_tag2
	
	testcam2.callcam(ekf, pub_tf_cam2, no_tag2, msg)
	


# initiating publishers
pub_ekf_output = rospy.Publisher('/ekf_position', PoseWithCovarianceStamped, queue_size = 1)
pub_tf_object = tf2_ros.TransformBroadcaster()
pub_tf_cam1 = tf2_ros.TransformBroadcaster()
pub_tf_cam2 = tf2_ros.TransformBroadcaster()

# initiating globals for all cameras
x_hat_0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
skalar = 1000000
P_mat_0 = np.diag([10*skalar, 10*skalar, 10*skalar, 0.5, 0.5, 0.5])
process_noise_pos = 0.05
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
testcam1 = cam.camera(1, -0.32, 1.525, -0.895, 0, 0, 0, R_cam1)
no_tag1 = 1 # init "no tag" counter

# initiating camera 2
R_cam2 = np.diag(np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4),
			   1.5 * (10 ** -6), 1.5 * (10 ** -6), 1.5 * (10 ** -4)]))*1000 # measurement noise
testcam2 = cam.camera(2, -0.32, 0.326, -0.895, 0, 0, 0, R_cam2)
no_tag2 = 1 # init "no tag" counter

def main():
	rospy.init_node('ext_auv_localization')
	rospy.Subscriber("/tag_detections1", AprilTagDetectionArray, callback1, queue_size=1)
	rospy.Subscriber("/tag_detections2", AprilTagDetectionArray, callback2, queue_size=1)
	rospy.spin()


if __name__ == '__main__':
	main()
