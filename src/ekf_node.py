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
	"""
	# testing new h-functions
	testcam1.h_with_quaternions()
	# testing end
	"""
	time_stamp = msg.header.stamp
	R_cam1 = testcam1.get_R_mat()
	h_cam1 = testcam1.h_cam(ekf.get_x_hat())
	H_cam1 = testcam1.get_H_Jac_sym()
	H_cam1_at_x = testcam1.H_jac_at_x(ekf.get_x_hat(), H_cam1)
	z = np.zeros((15, 1))
	h_sub = np.zeros((15, 1))
	R_sub = np.zeros((15, 15))
	H_jac_at_x_sub = np.zeros((15, 6))
	i = 0
	while i < len(msg.detections):
		if msg.detections[i].id == (40,):  # paranthesis and comma necessary since only then does "msg.detections[i].id" truely equal the subscriber output (query about tag-id)
			h_sub[0:3, 0:1] += h_cam1[0:3, 0:1]		
			R_sub[0:3, 0:3] += R_cam1[0:3, 0:3]
			H_jac_at_x_sub[0:3, 0:6] += H_cam1_at_x[0:3, 0:6]
			z[0] += msg.detections[i].pose.pose.pose.position.x
			z[1] += msg.detections[i].pose.pose.pose.position.y
			z[2] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (41,):
			h_sub[3:6, 0:1] += h_cam1[3:6, 0:1]
			R_sub[3:6, 3:6] += R_cam1[3:6, 3:6]
			H_jac_at_x_sub[3:6, 0:6] += H_cam1_at_x[3:6, 0:6]
			z[3] += msg.detections[i].pose.pose.pose.position.x
			z[4] += msg.detections[i].pose.pose.pose.position.y
			z[5] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (42,):
			h_sub[6:9, 0:1] += h_cam1[6:9, 0:1]
			R_sub[6:9, 6:9] += R_cam1[6:9, 6:9]
			H_jac_at_x_sub[6:9, 0:6] += H_cam1_at_x[6:9, 0:6]
			z[6] += msg.detections[i].pose.pose.pose.position.x
			z[7] += msg.detections[i].pose.pose.pose.position.y
			z[8] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (43,):
			h_sub[9:12, 0:1] += h_cam1[9:12, 0:1]
			R_sub[9:12, 9:12] += R_cam1[9:12, 9:12]
			H_jac_at_x_sub[9:12, 0:6] += H_cam1_at_x[9:12, 0:6]
			z[9] += msg.detections[i].pose.pose.pose.position.x
			z[10] += msg.detections[i].pose.pose.pose.position.y
			z[11] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (44,):
			h_sub[12:15, 0:1] += h_cam1[12:15, 0:1]
			R_sub[12:15, 12:15] += R_cam1[12:15, 12:15]
			H_jac_at_x_sub[12:15, 0:6] += H_cam1_at_x[12:15, 0:6]
			z[12] += msg.detections[i].pose.pose.pose.position.x
			z[13] += msg.detections[i].pose.pose.pose.position.y
			z[14] += msg.detections[i].pose.pose.pose.position.z
		else: 
			pass
		i += 1	

	R_sub = R_sub[R_sub.nonzero()]  # one dimensional vector containing all non zero values of the 15x15 R matrix
	z = z[z.nonzero()]
	# making a diagonal matrix R out of nonzero elements of R_sub ONLY IF WE SEE A TAG AT ALL(ultimate goal: scaling R according to how many and which tags we see)
	if len(msg.detections) == 0:
		print('camera1: no tag detected for {} steps'.format(no_tag1))
		no_tag1 += 1
	else:
		R_start = np.diag([R_sub[0], R_sub[1], R_sub[2]])
		no_tag1 = 1
		i, k = 0, 0
		while k < (len(msg.detections)-1):
			if len(msg.detections) == 1:
				break
			else:
				R_start = np.block([[R_start, np.zeros((i+3, 3))], [np.zeros((3, i+3)), np.diag([R_sub[i+3], R_sub[i+4], R_sub[i+5]])]])
				i += 3
				k += 1

		# building H_at_x and h_sub out of the tags we see (by getting ignorung the sub-arrays that are '0')
		sH = 0
		lH = 0
		H_start = np.zeros((15, 6))
		while sH/3 < 15:
			if np.any(H_jac_at_x_sub[sH:sH+3, 0:6]) == True:           # Scan the first 3 lines of H_jac_at_x_sub, if any of the elements are not zero, fill H_start with these lines. 
				H_start[lH:lH+3, 0:6] = H_cam1_at_x[sH:sH+3, 0:6]
				h_sub[lH:lH+3, 0:1] = h_cam1[sH:sH+3, 0:1]
				lH += 3
			sH += 3							   # If all elements from these lines are equal to zero, move on the the next 3 lines as long as sH/3 < 15
		H_start, garbage = np.vsplit(H_start, [3*len(msg.detections)])
		h_sub, trash = np.vsplit(h_sub, [3*len(msg.detections)])
		# print('R_start', R_start)					# new R_matrix with variances and covariances of the tags that are measured right now (max. 15x15)
		# print('H_start', H_start)  					# new H_matrix only considering the position of the tags that are currently measured (max. 15x6)
	# turning h_sub (row-vector by default) into a column-vector
	h_sub = h_sub.reshape(len(h_sub), 1)  					# new h_function only considering the position of the tags that are currently measured (max. 15x1)
	# print('h_sub', h_sub)
	# turning z (row-vector by default) into column-vactor
	z = z.reshape(len(z), 1) 						# new measurement-vector considering only the position of the tags that are currently measured
	ekf.predict()
	if not len(z) == 0:
		ekf.ekf_get_measurement(z, h_sub, H_start, R_start, time_stamp)	# passing the matrices to "ekf", a class instance of EkfLocalization
		ekf.update(z, h_sub, H_start, R_start)				# ekf.update is only called when a measurement 
		"""
		z       : measurement-vector according to the amount of seen tags (3-dimensions x, y, z per seen tag)
		h_sub   : measurement function scaled to amount of tags seen
		H_start : scaled Jacobian at the position x_hat (also according to amount of seen tags)
		R_start : Covariance matrix scaled according to seen tags
		"""
	# publishing Position, Orientation and Covariance as output of the Extended-Kalman-Filter
	ekf.ekf_publish(time_stamp, ekf.get_x_hat(), ekf.get_P_mat())
	# publishing the tf-transformation of the camera1-frame
	transforms_cam1 = []
	ekf_tf_cam1 = TransformStamped()
	quat_cam1 = quaternion_from_euler(-testcam1._a, -testcam1._b, -testcam1._g)
	ekf_tf_cam1 = TransformStamped()
	ekf_tf_cam1.header.stamp = time_stamp
	ekf_tf_cam1.header.frame_id = 'map'
	ekf_tf_cam1.child_frame_id = 'camera1'
	# changing world frame to NWU which is rviz's standart coordinate frame convention
	cam_location_NED = np.array([testcam1._t_x, testcam1._t_y, testcam1._t_z]).reshape(3, 1)
	NED2NWU = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
	cam_location_NWU = NED2NWU.dot(cam_location_NED)
	quat_cam1_temp = Quaternion(np.asarray([quat_cam1[3], quat_cam1[0], quat_cam1[1], quat_cam1[2]]))
	R_transform1 = quat_cam1_temp.rotation_matrix
	R_transformed_NWU = NED2NWU.dot(R_transform1)
	quat_NWU = Quaternion(matrix=R_transformed_NWU)
	# continue publishing with applied changes
	ekf_tf_cam1.transform.translation.x = cam_location_NWU[0]
	ekf_tf_cam1.transform.translation.y = cam_location_NWU[1]
	ekf_tf_cam1.transform.translation.z = cam_location_NWU[2]
	ekf_tf_cam1.transform.rotation.x = quat_NWU[1]
	ekf_tf_cam1.transform.rotation.y = quat_NWU[2]
	ekf_tf_cam1.transform.rotation.z = quat_NWU[3]
	ekf_tf_cam1.transform.rotation.w = quat_NWU[0]
	transforms_cam1.append(ekf_tf_cam1)
	pub_tf_cam1.sendTransform(transforms_cam1)  # rename into name of the publisher



def callback2(msg):  # sole purpose of this function is to recieve the position data of all seen tags from camera 2
	global testcam2, ekf, pub_tf_cam2, no_tag2
	time_stamp = msg.header.stamp
	R_cam2 = testcam2.get_R_mat()
	h_cam2 = testcam2.h_cam(ekf.get_x_hat())
	H_cam2 = testcam2.get_H_Jac_sym()
	H_cam2_at_x = testcam2.H_jac_at_x(ekf.get_x_hat(), H_cam2)
	z = np.zeros((15, 1))
	h_sub = np.zeros((15, 1))
	R_sub = np.zeros((15, 15))
	H_jac_at_x_sub = np.zeros((15, 6))
	i = 0
	while i < len(msg.detections):
		if msg.detections[i].id == (40,):  # paranthesis and comma necessary since only then does "msg.detections[i].id" truely equal the subscriber output (query about tag-id)
			h_sub[0:3, 0:1] += h_cam2[0:3, 0:1]		
			R_sub[0:3, 0:3] += R_cam2[0:3, 0:3]
			H_jac_at_x_sub[0:3, 0:6] += H_cam2_at_x[0:3, 0:6]
			z[0] += msg.detections[i].pose.pose.pose.position.x
			z[1] += msg.detections[i].pose.pose.pose.position.y
			z[2] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (41,):
			h_sub[3:6, 0:1] += h_cam2[3:6, 0:1]
			R_sub[3:6, 3:6] += R_cam2[3:6, 3:6]
			H_jac_at_x_sub[3:6, 0:6] += H_cam2_at_x[3:6, 0:6]
			z[3] += msg.detections[i].pose.pose.pose.position.x
			z[4] += msg.detections[i].pose.pose.pose.position.y
			z[5] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (42,):
			h_sub[6:9, 0:1] += h_cam2[6:9, 0:1]
			R_sub[6:9, 6:9] += R_cam2[6:9, 6:9]
			H_jac_at_x_sub[6:9, 0:6] += H_cam2_at_x[6:9, 0:6]
			z[6] += msg.detections[i].pose.pose.pose.position.x
			z[7] += msg.detections[i].pose.pose.pose.position.y
			z[8] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (43,):
			h_sub[9:12, 0:1] += h_cam2[9:12, 0:1]
			R_sub[9:12, 9:12] += R_cam2[9:12, 9:12]
			H_jac_at_x_sub[9:12, 0:6] += H_cam2_at_x[9:12, 0:6]
			z[9] += msg.detections[i].pose.pose.pose.position.x
			z[10] += msg.detections[i].pose.pose.pose.position.y
			z[11] += msg.detections[i].pose.pose.pose.position.z
		elif msg.detections[i].id == (44,):
			h_sub[12:15, 0:1] += h_cam2[12:15, 0:1]
			R_sub[12:15, 12:15] += R_cam2[12:15, 12:15]
			H_jac_at_x_sub[12:15, 0:6] += H_cam2_at_x[12:15, 0:6]
			z[12] += msg.detections[i].pose.pose.pose.position.x
			z[13] += msg.detections[i].pose.pose.pose.position.y
			z[14] += msg.detections[i].pose.pose.pose.position.z
		else: 
			pass
		i += 1	

	R_sub = R_sub[R_sub.nonzero()]  # one dimensional vector containing all non zero values of the 15x15 R matrix
	z = z[z.nonzero()]
	# making a diagonal matrix R out of nonzero elements of R_sub ONLY IF WE SEE A TAG AT ALL(ultimate goal: scaling R according to how many and which tags we see)
	if len(msg.detections) == 0:
		print('camera2: no tag detected for {} steps'.format(no_tag2))
		no_tag2 += 1
	else:
		R_start = np.diag([R_sub[0], R_sub[1], R_sub[2]])
		no_tag2 = 1
		i, k = 0, 0
		while k < (len(msg.detections)-1):
			if len(msg.detections) == 1:
				break
			else:
				R_start = np.block([[R_start, np.zeros((i+3, 3))], [np.zeros((3, i+3)), np.diag([R_sub[i+3], R_sub[i+4], R_sub[i+5]])]])
				i += 3
				k += 1

		# building H_at_x and h_sub out of the tags we see (by getting ignorung the sub-arrays that are '0')
		sH = 0
		lH = 0
		H_start = np.zeros((15, 6))
		while sH/3 < 15:
			if np.any(H_jac_at_x_sub[sH:sH+3, 0:6]) == True:           # Scan the first 3 lines of H_jac_at_x_sub, if any of the elements are not zero, fill H_start with these lines. 
				H_start[lH:lH+3, 0:6] = H_cam2_at_x[sH:sH+3, 0:6]
				h_sub[lH:lH+3, 0:1] = h_cam2[sH:sH+3, 0:1]
				lH += 3
			sH += 3							   # If all elements from these lines are equal to zero, move on the the next 3 lines as long as sH/3 < 15
		H_start, garbage = np.vsplit(H_start, [3*len(msg.detections)])
		h_sub, trash = np.vsplit(h_sub, [3*len(msg.detections)])
		# print('R_start', R_start)					# new R_matrix with variances and covariances of the tags that are measured right now (max. 15x15)
		# print('H_start', H_start)  					# new H_matrix only considering the position of the tags that are currently measured (max. 15x6)
	# turning h_sub (row-vector by default) into a column-vector
	h_sub = h_sub.reshape(len(h_sub), 1)  					# new h_function only considering the position of the tags that are currently measured (max. 15x1)
	# print('h_sub', h_sub)
	# turning z (row-vector by default) into column-vactor
	z = z.reshape(len(z), 1) 						# new measurement-vector considering only the position of the tags that are currently measured
	ekf.predict()
	if not len(z) == 0:
		ekf.ekf_get_measurement(z, h_sub, H_start, R_start, time_stamp)	# passing the matrices to "ekf", a class instance of EkfLocalization
		ekf.update(z, h_sub, H_start, R_start)				# ekf.update is only called when a measurement 
		"""
		z       : measurement-vector according to the amount of seen tags (3-dimensions x, y, z per seen tag)
		h_sub   : measurement function scaled to amount of tags seen
		H_start : scaled Jacobian at the position x_hat (also according to amount of seen tags)
		R_start : Covariance matrix scaled according to seen tags
		"""
	# # publishing Position, Orientation and Covariance as output of the Extended-Kalman-Filter
	ekf.ekf_publish(time_stamp, ekf.get_x_hat(), ekf.get_P_mat())
	# publishing the tf-transformation of the camera1-frame
	transforms_cam2 = []
	ekf_tf_cam2 = TransformStamped()
	quat_cam2 = quaternion_from_euler(testcam2._a, testcam2._b, testcam2._g)
	ekf_tf_cam2 = TransformStamped()
	ekf_tf_cam2.header.stamp = time_stamp
	ekf_tf_cam2.header.frame_id = 'map'
	ekf_tf_cam2.child_frame_id = 'camera2'
	# changing world frame to NWU which is rviz standart coordinate frame convention
	cam_location_NED = np.array([testcam2._t_x, testcam2._t_y, testcam2._t_z]).reshape(3, 1)
	NED2NWU = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
	cam_location_NWU = NED2NWU.dot(cam_location_NED)
	quat_cam2_temp = Quaternion(np.asarray([quat_cam2[1], quat_cam2[2], quat_cam2[3], quat_cam2[0]]))
	R_transform2 = quat_cam2_temp.rotation_matrix
	R_transformed_NWU = NED2NWU.dot(R_transform2)
	quat_NWU_temp = Quaternion(matrix=R_transformed_NWU)
	quat_NWU = Quaternion(quat_NWU_temp[3], quat_NWU_temp[0], quat_NWU_temp[1], quat_NWU_temp[2])
	# continue publishing
	ekf_tf_cam2.transform.translation.x = cam_location_NWU[0]
	ekf_tf_cam2.transform.translation.y = cam_location_NWU[1]
	ekf_tf_cam2.transform.translation.z = cam_location_NWU[2]
	ekf_tf_cam2.transform.rotation.x = quat_NWU[0]
	ekf_tf_cam2.transform.rotation.y = quat_NWU[1]
	ekf_tf_cam2.transform.rotation.z = quat_NWU[2]
	ekf_tf_cam2.transform.rotation.w = quat_NWU[3]
	transforms_cam2.append(ekf_tf_cam2)
	pub_tf_cam2.sendTransform(transforms_cam2)  # rename into name of the publisher



# initiating publishers
pub_ekf_output = rospy.Publisher('/ekf_position', PoseWithCovarianceStamped, queue_size = 1)
pub_tf_object = tf2_ros.TransformBroadcaster()
pub_tf_cam1 = tf2_ros.TransformBroadcaster()
pub_tf_cam2 = tf2_ros.TransformBroadcaster()

# initiating globals for all cameras
x_hat_0 = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
P_mat_0 = np.diag([100, 100, 100, 0.5, 0.5, 0.5])
process_noise = 0.5
Q_mat = np.diag([process_noise ** 2, process_noise ** 2, process_noise ** 2, process_noise ** 2, process_noise ** 2, process_noise ** 2])
ekf = ekf_model.EkfLocalization(x_hat_0, P_mat_0, Q_mat, pub_ekf_output, pub_tf_object)

# initiating camera 1
R_cam1 = np.eye(15) * (np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4)]).reshape(15, 1)) # measurement noise
testcam1 = cam.camera(1, -0.13027812, 0.77786706, -0.85595675, -0.4727079075435697, 0.015231312520850101, 0.03427884792178946, R_cam1)
no_tag1 = 1
#old orientation data (a = -1.0472, b = 0, c = -math.pi/2)0.4727079075435697, 0.015231312520850101, 0.03427884792178946,

# initiating camera 2
R_cam2 = np.eye(15) * (np.array([1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4), 1.5 * (10 ** -6), 1.5 * (10 ** -6), 3 * (10 ** -4)]).reshape(15, 1)) # measurement noise
testcam2 = cam.camera(2, -0.17621339, 0.09021926, -0.85382418, 0.4172191040475571, 0.014481942780375505, 3.1249935327318172, R_cam2)
no_tag2 = 1

def main():
	rospy.init_node('ekf')
	rospy.Subscriber("/tag_detections1", AprilTagDetectionArray, callback1, queue_size=1)
	rospy.Subscriber("/tag_detections2", AprilTagDetectionArray, callback2, queue_size=1)
	rospy.spin()


if __name__ == '__main__':
	main()
