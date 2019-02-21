# class for the ekf-operation
import numpy as np
import math
import time
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from numpy.linalg import inv
from pyquaternion import Quaternion
# testing: publishing z_tilde
import rospy
from ekf.msg import callback_data
# testing end


class EkfLocalization:

	def __init__(self, x_hat_0, P_mat_0, Q_mat, pub_ekf_output, pub_tf_object):
		self._x_hat = x_hat_0
		self._P_mat = P_mat_0
		self._Q_mat = Q_mat
		self._pub_ekf_output = pub_ekf_output
		self._pub_tf_object = pub_tf_object
		# class variables
		self._z = 0
		self._h_at_x = 0
		self._H_cam_at_x = 0
		self._R_mat = 0
		self._F = np.eye(6)
		#self._time_stamp = 0.0
		self._cam_id = 0
		# testing: publishing z_tilde
		self._pub = rospy.Publisher('/z_tilde', callback_data, queue_size = 1)

	def get_x_hat(self):
		return self._x_hat

	def get_P_mat(self):
		return self._P_mat

	def get_time_stamp(self):
		return self._time_stamp

	def predict(self):
		x_hat_temp = self._x_hat.reshape(6, 1)
		self._x_hat = self._F.dot(x_hat_temp)
		self._P_mat = self._F.dot(self._P_mat.dot(self._F)) + self._Q_mat

	def update(self, z, h_at_x, H_cam_at_x, R_mat):
		L = inv(H_cam_at_x.dot(self._P_mat.dot(H_cam_at_x.T)) + R_mat)
		K = self._P_mat.dot((H_cam_at_x.T).dot(L))
		z_tilde = z - h_at_x
		# testing: publishing z_tilde
		z_tilde_msg = callback_data()
		z_tilde_msg.h = z_tilde
		self._pub.publish(z_tilde_msg)
		# testing end
		# limit size of angles by +/- pi
		self._x_hat = self._x_hat + K.dot(z_tilde)
		self._P_mat = (np.eye(6) - K.dot(H_cam_at_x)).dot(self._P_mat)
		if self._x_hat[3] > 0.0:
			self._x_hat[3] = np.mod(self._x_hat[3], np.pi)
		elif self._x_hat[3] < 0.0:
			self._x_hat[3] = -np.mod(-self._x_hat[3], np.pi)
		
		if self._x_hat[4] > 0.0:
			self._x_hat[4] = np.mod(self._x_hat[4], np.pi)
		elif self._x_hat[4] < 0.0:
			self._x_hat[4] = -np.mod(-self._x_hat[4], np.pi)
			
		if self._x_hat[5] > 0.0:
			self._x_hat[5] = np.mod(self._x_hat[5], np.pi)
		elif self._x_hat[5] < 0.0:
			self._x_hat[5] = -np.mod(-self._x_hat[5], np.pi)
		

		"""
		L = auxiliary variable
		K = Kalman gain
		z_tilde = difference of actual measurement and expectation
		"""

   	def ekf_get_measurement(self, z, h_at_x, H_cam_at_x, R_mat, time_stamp, cam_id):
		self._z = z
		self._h_at_x = h_at_x
		self._H_cam_at_x = H_cam_at_x
		self._R_mat = R_mat
		self._time_stamp = time_stamp
		self._cam_id = cam_id
	
        	"""
		z       : measurement-vector according to the amount of seen tags (3-dimensions x, y, z per seen tag)
		h_sub   : measurement function scaled to amount of tags seen
		H_start : scaled Jacobian at the position x_hat (also according to amount of seen tags)
		R_start : Covariance matrix scaled according to seen tags
		"""

	def ekf_publish(self, time_stamp, x_hat, P):
		# converting the object's orientation from euler-angles to a quaternion, since tf and nav_msgs both describe orientation through quaternions
		quat = quaternion_from_euler(x_hat[3], x_hat[4], x_hat[5])

		# publishing Position, Orientation and Covariance as output of the Extended-Kalman-Filter
		ekfOutput_msg = PoseWithCovarianceStamped()
		ekfOutput_msg.header.stamp = time_stamp
		ekfOutput_msg.header.frame_id = 'cube'
		ekfOutput_msg.pose.pose.position.x = x_hat[0]
		ekfOutput_msg.pose.pose.position.y = x_hat[1]
		ekfOutput_msg.pose.pose.position.z = x_hat[2]
		ekfOutput_msg.pose.pose.orientation.x = quat[0]  # quaternion (must be calculated from euler angles)
		ekfOutput_msg.pose.pose.orientation.y = quat[1]
		ekfOutput_msg.pose.pose.orientation.z = quat[2]
		ekfOutput_msg.pose.pose.orientation.w = quat[3]
		covar = P.flatten()  # published covariance matrix must be of type 'array' while P is still of type 'numpy.ndarray'. flatten() converts type ndarray to type array
		
		ekfOutput_msg.pose.covariance = covar.tolist()
		self._pub_ekf_output.publish(ekfOutput_msg)


		# publishing the tf-transformation of the cube/boat/object-frame
		transforms = []
		ekf_tf_msg = TransformStamped()
		ekf_tf_msg.header.stamp  = time_stamp
		ekf_tf_msg.header.frame_id = 'map'
		ekf_tf_msg.child_frame_id = 'cube'
		# changing world frame to NWU which is rviz standart coordinate frame convention (doing so by converting quat into a rotation matrix and transforming it into rviz standart)
		cube_location_NED = np.array([x_hat[0], x_hat[1], x_hat[2]]).reshape(3, 1)
		NED2NWU = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
		cube_location_NWU = NED2NWU.dot(cube_location_NED)
		quat_cube_temp = Quaternion(np.asarray([quat[1], quat[2], quat[3], quat[0]]))
		R_transform = quat_cube_temp.rotation_matrix
		R_transformed_NWU = NED2NWU.dot(R_transform)
		quat_NWU_temp = Quaternion(matrix=R_transformed_NWU)
		quat_NWU = Quaternion(quat_NWU_temp[3], quat_NWU_temp[0], quat_NWU_temp[1], quat_NWU_temp[2])
		# continue publishing
		ekf_tf_msg.transform.translation.x = cube_location_NWU[0]
		ekf_tf_msg.transform.translation.y = cube_location_NWU[1]
		ekf_tf_msg.transform.translation.z = cube_location_NWU[2]
		ekf_tf_msg.transform.rotation.x = quat_NWU[0]
		ekf_tf_msg.transform.rotation.y = quat_NWU[1]
		ekf_tf_msg.transform.rotation.z = quat_NWU[2]
		ekf_tf_msg.transform.rotation.w = quat_NWU[3]
		transforms.append(ekf_tf_msg)
		self._pub_tf_object.sendTransform(transforms)
