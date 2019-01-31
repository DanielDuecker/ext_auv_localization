#!/usr/bin/env python

from pyquaternion import Quaternion
import rospy
import math
import numpy as np
from apriltags2_ros.msg import AprilTagDetectionArray
from numpy.linalg import inv
from math import atan2, asin
from tf.transformations import euler_from_quaternion



"""
to configurate this programm:
step1: Change to subscribed topic in the "main" function to "/tag_detectioni", where "i" is the number corresponding to the camera that needs to be calibrated.
step2: Change the position of the tag (pos_tag_w) in "globals" according to where it is located in the world frame.
step3: do the same for the angle of the tag in the world frame (phi).
note: I strongly recommend to keep the tag in the horizontal middle axis of the camera (keep y as close to zero as possible) and to have the green tag-border (seen through /tag_detections_image) face downwards in the image, since this programm gives the best results this way.
"""



# globals
pos_tag_w = np.array([1, 1.63, 0]).reshape(3, 1)		# position of the calibration-tag in world coordinates (change everytime the tag gets moved somewhere else)
phi = 0								# angle of which the tag is rotated around z-axis (in relation to the world system)
Rot_tag2w_temp = inv(np.array([[math.cos(phi), -math.sin(phi), 0],
			[math.sin(phi), math.cos(phi), 0],
			[0, 0, 1]]).reshape(3, 3))
Rot_wtemp2w = np.array([0, 1, 0, 1, 0, 0, 0, 0, -1]).reshape(3, 3)
R_tag2w = Rot_wtemp2w.dot(Rot_tag2w_temp)


# trying to get euler angles from quaternion directly (R_cam2tag will later have to be computed from euler angles)
def quaternion2euler_new(quat):
	sqx = quat[1]*quat[1]
	sqy = quat[2]*quat[2]
	sqz = quat[3]*quat[3]
	# x-axis rotation
	roll = atan2(2*(quat[0]*quat[1]+quat[2]*quat[3]),1-2*(sqx+sqy))
	# y-axis rotation
	test = asin(2*(quat[0]*quat[2]-quat[3]*quat[1]))
	if test >= 1:
		pitch = sin(math.pi/2)
	else:
		pitch = asin(test)
	# z-axis rotation
	yaw = atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(sqy+sqz))
	return np.array([roll, pitch, yaw])


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def callback(msg):
	global pos_tag_w, R_tag2w
	my_quaternion = Quaternion(np.asarray([
	msg.detections[0].pose.pose.pose.orientation.w,
	msg.detections[0].pose.pose.pose.orientation.x,
        msg.detections[0].pose.pose.pose.orientation.y,
        msg.detections[0].pose.pose.pose.orientation.z]))				# Quaternion transforming cam-frame into tag-frame

	# euler angles (hopefully) and their seperate rotation matrices
	euler_vec = quaternion2euler_new(my_quaternion)
	roll = euler_vec[0]
	pitch = euler_vec[1]
	yaw = euler_vec[2]
	R_roll = np.array([[1, 0, 0],
			[0, np.cos(roll), -np.sin(roll)],
			[0, np.sin(roll), np.cos(roll)]])
	R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
			[0, 1, 0],
			[-np.sin(pitch), 0, np.cos(pitch)]])
	R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
			[np.sin(yaw), np.cos(yaw), 0],
			[0, 0, 1]])

    	R_cam2tag = inv(R_yaw.dot(R_pitch).dot(R_roll))					# Rotationmatrix that is equivalent to the Quaternion

	vec_meas_cam2tag_c = np.array([msg.detections[0].pose.pose.pose.position.x,	# position measurement from the tag (in camera system)
				       msg.detections[0].pose.pose.pose.position.y,
				       msg.detections[0].pose.pose.pose.position.z]).reshape(3, 1)
	vec_tag2cam_t = R_cam2tag.dot(-vec_meas_cam2tag_c)				# vector cam-tag in tag system

	pos_cam_w = pos_tag_w + R_tag2w.dot(vec_tag2cam_t)				# position vector of the camera in the world system
	R_cam_w = inv(R_tag2w.dot(R_cam2tag))						# orientation of the camera in the world system (rotation Matrix)
	ori_cam_w = rotationMatrixToEulerAngles(R_cam_w)
	print 'position cam:'
	print pos_cam_w
	print 'euler angles:'
	print ori_cam_w[0]
	print ori_cam_w[1]
	print ori_cam_w[2]


def main():
	rospy.init_node('get_camera_pos_ori')
	rospy.Subscriber("/tag_detections1", AprilTagDetectionArray, callback, queue_size=1)  # subscribe to "/tag_detectionsi" to calibrate "camera i"
	rospy.spin()


if __name__ == '__main__':
	main()
