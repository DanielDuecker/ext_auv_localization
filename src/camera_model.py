# class for easy camera addition
import numpy as np
import math
import sympy
import time
import rospy
import symengine

from sympy import *
from sympy.abc import x, y, z, p, t, s

from numpy.linalg import inv
from tf.transformations import quaternion_from_euler # defined q = (x, y, z, w)
from tf.transformations import quaternion_inverse
from tf.transformations import quaternion_multiply
from pyquaternion import Quaternion  # defined q = (w, x, y, z)
from geometry_msgs.msg import TransformStamped


class camera:

	def __init__(self, cam_id, t_x, t_y, t_z, a, b, g, R_mat,z_scale, x_scale):
		self._cam_id = cam_id
		self._t_x = t_x
		self._t_y = t_y
		self._t_z = t_z
		self._a = a
		self._b = b
		self._g = g
		self._R_mat = R_mat
		self._z_scale = z_scale
		self._x_scale = x_scale
		self._is_time_offset_init = False
		self._time_offset_secs = 0
		self._time_offset_nsecs = 0
		# class variables
		print("Please give me some time to preprocess the H(x) jacobian of Cam#" +str(cam_id))
		self._H_jacobi_sym = self.compute_jacobian()		# symbolic Jacobian of h
		self._H_jac_func = self.H_jac_to_numpy()  # converts symbolic jacobian to numpy-function handle
		print("Done!")
		
	def set_time_offset(self, secs, nsecs):
		if not self._is_time_offset_init:
			self._time_offset_secs = secs
			self._time_offset_nsecs = nsecs
			self._is_time_offset_init = True
		else:
			pass
			
	def compensate_cam_time(self, new_secs, new_nsecs):
		out_nsecs = new_nsecs - self._time_offset_nsecs
		if out_nsecs < 0:
			out_nsecs = abs(out_nsecs)
			out_secs = new_secs - self._time_offset_secs-1
		else:
			out_secs = new_secs - self._time_offset_secs
		
		return out_secs, out_nsecs
	
	
	
	def get_H_Jac_sym(self):
		return	self._H_jacobi_sym
	
	def H_jac_to_numpy(self):  
	# lambify sympy matrix -> speed up x100
	
		H_jac = self.get_H_Jac_sym()	
    		H_jac_func_handle = lambdify([x, y, z, p, t, s], H_jac, "numpy")
    		
    		return H_jac_func_handle

	def RotationMatrixToEulerAngles(self, R):
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
		
	def get_R_mat(self):
		return self._R_mat
		

	def H_jac_at_x(self, x_w_hat):  
	# evaluate the jacobian H from a camera at estimate x_w_hat
	        
	        H_jac_at_x = self._H_jac_func(x_w_hat[0,0],x_w_hat[1,0],x_w_hat[2,0],x_w_hat[3,0],x_w_hat[4,0],x_w_hat[5,0])
  		    		
    		return H_jac_at_x



	def h_cam(self, x_cube_w):
	        
		# defining parameters
		x_cube_w = x_cube_w.reshape(1, 6)
    		x_cube_w = x_cube_w[0]
    		x_w = x_cube_w[0]
    		y_w = x_cube_w[1]
   		z_w = x_cube_w[2]
    		p = x_cube_w[3]*0  # phi
    		t = x_cube_w[4]*0  # theta
    		s = x_cube_w[5]    # psi

		# vectors from cube-center to tag-center (cube-system)
		"""
		old offsets
		"""
		#offset1 = np.array([0, -0.075, 0, 1]).reshape(4, 1)
		#offset2 = np.array([-0.075, 0, 0, 1]).reshape(4, 1)
		#offset3 = np.array([0, 0.075, 0, 1]).reshape(4, 1)
		#offset4 = np.array([0, 0, -0.075, 1]).reshape(4, 1)
		#offset5 = np.array([0, 0, 0.075, 1]).reshape(4, 1)
		"""
		new offsets
		"""
		offset1 = np.array([0.075, 0, 0, 1]).reshape(4, 1)  #45
		offset2 = np.array([0, -0.075, 0, 1]).reshape(4, 1) #46
		offset3 = np.array([-0.075, 0, 0, 1]).reshape(4, 1) #47
		offset4 = np.array([0, 0, -0.075, 1]).reshape(4, 1) #48
		offset5 = np.array([0, 0.085, 0, 1]).reshape(4, 1)  #49
		"""
		boat offsets
		"""		
		#offset1 = np.array([0, -0.082, 0, 1]).reshape(4, 1)  #45
		#offset2 = np.array([0, 0.082, 0, 1]).reshape(4, 1) #46
		#offset3 = np.array([0, 0, 0.082, 1]).reshape(4, 1) #47
		#offset4 = np.array([0.173, 0, 0, 1]).reshape(4, 1) #48
		#offset5 = np.array([-0.24, 0, 0, 1]).reshape(4, 1)  #49
		
		offset = np.array([offset1, offset2, offset3, offset4, offset5]).reshape(20, 1)
		

		# Tansformation matrix from cube to world
		R_x_cube = np.array([[1, 0, 0],					# Rotation around the x-axis
				[0, np.cos(p), -np.sin(p)],
				[0, np.sin(p), np.cos(p)]])
		R_y_cube = np.array([[np.cos(t), 0, np.sin(t)],			# Rotation around the y-axis
				[0, 1, 0],
				[-np.sin(t), 0, np.cos(t)]])
		R_z_cube = np.array([[np.cos(s), -np.sin(s), 0],		# Rotation around the z-axis
				[np.sin(s), np.cos(s), 0],
				[0, 0, 1]])
		R_cube2world = np.transpose(R_z_cube.dot(R_y_cube.dot(R_x_cube)))	# Rotation matrix consisting of all 3 axis rotations
		T_cube2world = np.block([					# Transformation matrix from cube to world system
					[(R_cube2world), np.array([x_w, y_w, z_w]).reshape(3, 1)],
					[np.zeros(3), 1]
					])

		# Transformation matrix from world to cam
		R_x = np.array([[1, 0, 0],					# Rotation around the x-axis
				[0, np.cos(self._a), -np.sin(self._a)],
				[0, np.sin(self._a), np.cos(self._a)]])
		R_y = np.array([[np.cos(self._b), 0, np.sin(self._b)],		# Rotation around the y-axis
				[0, 1, 0],
				[-np.sin(self._b), 0, np.cos(self._b)]])
		R_z = np.array([[np.cos(self._g), -np.sin(self._g), 0],		# Rotation around the z-axis
				[np.sin(self._g), np.cos(self._g), 0],
				[0, 0, 1]])
		#R_permut = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])		# needed to adjust world koordinate system to cam koordinate system
		#R_world2cam = R_permut.dot(R_z).dot(R_y).dot(R_x)		# rotation matrix from world to camera
		R_world2cam = np.transpose(R_z.dot(R_y).dot(R_x))				# rotation matrix from world to camera
		
		camera_position_w = np.array([self._t_x, self._t_y, self._t_z]).reshape(3, 1)
		
		camera_position_cam = R_world2cam.dot(-camera_position_w) # old positions self._t_x, self._t_y, self._t_z
		T_world2cam = np.block([					# Transformation matrix from world to camera
					[R_world2cam, camera_position_cam],
					[np.zeros(3), 1]
					])

		# Axis-transformation from world-system (NED) to camera-system (EDN)
		#R_umrechnung = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)

		# computing the h-function for each tag and stacking the 3x1 vectors on top of eachother afterwards (h_i = Rumrechnung*T_world2cam*T_cube2world*offset_i)
		h = np.zeros((15, 1))
		j_counter = 0
		i = 0
		while i < 15:
			#h_temp = R_umrechnung.dot(T_world2cam).dot(T_cube2world).dot(offset[j_counter:j_counter+4, 0:1]).reshape(4, 1)
			h_temp1 = T_world2cam.dot(T_cube2world.dot((offset[j_counter:j_counter+4, 0:1]).reshape(4, 1)))
			h_temp = [[h_temp1[1,0:1]],
     			          [h_temp1[2,0:1]],
				  [h_temp1[0,0:1]]]
			h[i:i+3, 0:1] = h_temp[0:3]
			i += 3
			j_counter +=4
		"""
		# this section is only needed to compute h for the jacobian-function (computes h with variables -> not h_at_x)
		
		# creating h_but_not_at_x (for H_jac_at_x) analog to above, only with variables instead of concrete numbers
		x, y, z, p, t, s, t_x, t_y, t_z, a, b, g = sympy.symbols('x y z p t s t_x t_y t_z a b g')
		R_x_cube_sympy = sympy.Matrix([[1, 0, 0],
					[0, sympy.cos(p), sympy.sin(p)],
					[0, -sympy.sin(p), sympy.cos(p)]])
		R_y_cube_sympy = sympy.Matrix([[sympy.cos(t), 0, -sympy.sin(t)],
					[0, 1, 0],
					[sympy.sin(t), 0, sympy.cos(t)]])
		R_z_cube_sympy = sympy.Matrix([[sympy.cos(s), sympy.sin(s), 0],
					[-sympy.sin(s), sympy.cos(s), 0],
					[0, 0, 1]])
		R_world2cube_sympy = R_z_cube_sympy*(R_y_cube_sympy*(R_x_cube_sympy))
		R_cube2world_sympy = R_world2cube_sympy.T
		#print 'R_cube2world_sympy'
		#print R_cube2world_sympy
		T_cube2world_sympy = sympy.Matrix([[sympy.cos(s)*sympy.cos(t), -sympy.sin(s)*sympy.cos(t), sympy.sin(t), x],
						[sympy.sin(p)*sympy.sin(t)*sympy.cos(s) + sympy.sin(s)*sympy.cos(p), -sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + sympy.cos(p)*sympy.cos(s), -sympy.sin(p)*sympy.cos(t), y],
						[sympy.sin(p)*sympy.sin(s) - sympy.sin(t)*sympy.cos(p)*sympy.cos(s), sympy.sin(p)*sympy.cos(s) + sympy.sin(s)*sympy.sin(t)*sympy.cos(p), sympy.cos(p)*sympy.cos(t), z],
						[0, 0, 0, 1]])
		R_x_sympy = sympy.Matrix([[1, 0, 0],
					[0, sympy.cos(a), sympy.sin(a)],
					[0, -sympy.sin(a), sympy.cos(a)]])
		R_y_sympy = sympy.Matrix([[sympy.cos(b), 0, -sympy.sin(b)],
					[0, 1, 0],
					[sympy.sin(b), 0, sympy.cos(b)]])
		R_z_sympy = sympy.Matrix([[sympy.cos(g), sympy.sin(g), 0],
					[-sympy.sin(g), sympy.cos(g), 0],
					[0, 0, 1]])
		R_world2cam_sympy = (R_z_sympy*(R_y_sympy*(R_x_sympy)))
		
		#print 'R_world2cam_sympy'
		#print R_world2cam_sympy
		hilfsvec = sympy.Matrix([[t_x], [t_y], [t_z]])
		hilfsvec2 = R_world2cam_sympy*hilfsvec
		T_world2cam_sympy = sympy.Matrix([[sympy.cos(b)*sympy.cos(g), sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a), sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g), hilfsvec2[0]],
						[-sympy.sin(g)*sympy.cos(b), -sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g), sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a), hilfsvec2[1]],
						[sympy.sin(b), -sympy.sin(a)*sympy.cos(b), sympy.cos(a)*sympy.cos(b), hilfsvec2[2]],
						[0, 0, 0, 1]])
		R_umrechnung_sympy = sympy.Matrix([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
		offset_sympy = sympy.Matrix(offset)
		h_sympy = sympy.Matrix([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		j_counter = 0
		i = 0
		while i < 15:
			h_temp_symp1 = (T_world2cam_sympy*(T_cube2world_sympy*(offset[j_counter:j_counter+4, 0:1]))).reshape(4, 1)
			h_temp_sympy = [[h_temp_symp1[1,0:1]],
     			          	[h_temp_symp1[2,0:1]],
				  	[h_temp_symp1[0,0:1]]]
			h_sympy[i:i+3, 0:1] = h_temp_sympy[0:3]
			i += 3
			j_counter +=4
		print 'h_sympy:'
		print h_sympy
		"""

		
		#elapse_h = time.time()-t_h
		#print("elapse_h = " + str(elapse_h)) # time_counter
		
		# returning h_at_x
		return h # 15x1 vector showing the position of all tags depending on the cube's position
		
	
	
	
	
        

	

    		
    	def callcam(self, ekf, pub_tf_cam, no_tag, msg):
	    	#print('ekfcallcam: get x5 = ' + str(ekf.get_x_hat()[5]))
	    	#print('seconds from msg:')
	    	#print (str(msg.header.stamp.secs-rospy.get_rostime().secs) + ' ' + str(msg.header.stamp.nsecs-rospy.get_rostime().nsecs))
	    	#print('actual current time')
	    	#print(str(rospy.get_rostime().secs)+' '+str(rospy.get_rostime().nsecs))
    		ekf.predict()
		time_stamp = msg.header.stamp
		if len(msg.detections) == 0:
				#print('camera'+'{}: no tag detected'.format(self._cam_id))
				pass
		else:# len(msg.detections) == 0:
			R_cam =self._R_mat
			h_cam = self.h_cam(ekf.get_x_hat())
			H_cam_at_x = self.H_jac_at_x(ekf.get_x_hat())
			z = np.zeros((15, 1))
			h_sub = np.zeros((15, 1))
			R_sub = np.zeros((15, 15))
			H_jac_at_x_sub = np.zeros((15, 6))
			R_scale = 0
			i = 0
			#z_scale = 1/1.35
			while i < len(msg.detections):
				if msg.detections[i].id == (45,):  # paranthesis and comma necessary since only then does "msg.detections[i].id" truely equal the subscriber output (query about tag-id)
					z[0] += msg.detections[i].pose.pose.pose.position.x*self._x_scale  			#z
					z[1] += msg.detections[i].pose.pose.pose.position.y             	#x
					z[2] += msg.detections[i].pose.pose.pose.position.z*self._z_scale	#y
					h_sub[0:3, 0:1] += h_cam[0:3, 0:1]		
					H_jac_at_x_sub[0:3, 0:6] += H_cam_at_x[0:3, 0:6]
					R_sub[0:3, 0:3] += R_cam[0:3, 0:3]
					#R_sub[1,1] += R_scale*(z[1]*100)**2
				elif msg.detections[i].id == (46,):
					z[3] += msg.detections[i].pose.pose.pose.position.x*self._x_scale  			#z
					z[4] += msg.detections[i].pose.pose.pose.position.y  			#x
					z[5] += msg.detections[i].pose.pose.pose.position.z*self._z_scale	#y
					h_sub[3:6, 0:1] += h_cam[3:6, 0:1]		
					H_jac_at_x_sub[3:6, 0:6] += H_cam_at_x[3:6, 0:6]
					R_sub[3:6, 3:6] += R_cam[3:6, 3:6]
					#R_sub[4,4] += R_scale*(z[4]*100)**2
				elif msg.detections[i].id == (47,):
					z[6] += msg.detections[i].pose.pose.pose.position.x*self._x_scale
					z[7] += msg.detections[i].pose.pose.pose.position.y
					z[8] += msg.detections[i].pose.pose.pose.position.z*self._z_scale
					h_sub[6:9, 0:1] += h_cam[6:9, 0:1]
					H_jac_at_x_sub[6:9, 0:6] += H_cam_at_x[6:9, 0:6]
					R_sub[6:9, 6:9] += R_cam[6:9, 6:9]
					#R_sub[7,7] += R_scale*(z[7]*100)**2
				elif msg.detections[i].id == (48,):
					z[9] += msg.detections[i].pose.pose.pose.position.x*self._x_scale
					z[10] += msg.detections[i].pose.pose.pose.position.y
					z[11] += msg.detections[i].pose.pose.pose.position.z*self._z_scale
					h_sub[9:12, 0:1] += h_cam[9:12, 0:1]	
					H_jac_at_x_sub[9:12, 0:6] += H_cam_at_x[9:12, 0:6]
					R_sub[9:12, 9:12] += R_cam[9:12, 9:12]
					#R_sub[10,10] += R_scale*(z[10]*100)**2
				elif msg.detections[i].id == (49,):
					z[12] += msg.detections[i].pose.pose.pose.position.x*self._x_scale
					z[13] += msg.detections[i].pose.pose.pose.position.y
					z[14] += msg.detections[i].pose.pose.pose.position.z*self._z_scale
					h_sub[12:15, 0:1] += h_cam[12:15, 0:1]
					H_jac_at_x_sub[12:15, 0:6] += H_cam_at_x[12:15, 0:6]
					R_sub[12:15, 12:15] += R_cam[12:15, 12:15]
					#R_sub[13,13] += R_scale*(z[13]*100)**2
				else: 
					pass
				i += 1	
			
			R_sub = R_sub[R_sub.nonzero()]  # one dimensional vector containing all non zero values of the 15x15 R matrix
			z = z[z.nonzero()]
			# making a diagonal matrix R out of nonzero elements of R_sub ONLY IF WE SEE A TAG AT ALL(ultimate goal: scaling R according to how many tags we see)
			if len(msg.detections) == 0:
				print('camera'+'{}: no tag detected'.format(self._cam_id))
				#print('no tag detected for {} steps'.format(no_tag))
				no_tag += 1
			else:
				R_start = np.diag([R_sub[0], R_sub[1], R_sub[2]])
				no_tag = 1 # reset no tag counter
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
						H_start[lH:lH+3, 0:6] = H_cam_at_x[sH:sH+3, 0:6]
						h_sub[lH:lH+3, 0:1] = h_cam[sH:sH+3, 0:1]
						lH += 3
					sH += 3							   # If all elements from these lines are equal to zero, move on the the next 3 lines as long as sH/3 < 15
				H_start, garbage = np.vsplit(H_start, [3*len(msg.detections)])
				h_sub, trash = np.vsplit(h_sub, [3*len(msg.detections)])
				# print('R_start', R_start)					# new R_matrix with variances and covariances of the tags that are measured right now (max. 15x15)
				# print('H_start', H_start)  					# new H_matrix only considering the position of the tags that are currently measured (max. 15x6)
			# turning h_sub (row-vector by default) into a column-vector
			# new h_function only considering the position of the tags that are currently measured (max. 15x1)
			h_sub = h_sub.reshape(len(h_sub), 1)  					
			# turning z (row-vector by default) into column-vactor
			# new measurement-vector considering only the position of the tags that are currently measured
			z = z.reshape(len(z), 1)						
			if not len(z) == 0:
				# passing the matrices to "ekf", a class instance of EkfLocalization
				ekf.ekf_get_measurement(z, h_sub, H_start, R_start, time_stamp, self._cam_id)	
				# ekf.update is only called when a measurement 
				ekf.set_xhat345_to_zero()
				ekf.update(z, h_sub, H_start, R_start)
				"""
				z       : measurement-vector according to the amount of seen tags (3-dimensions x, y, z per seen tag)
				h_sub   : measurement function scaled to amount of tags seen
				H_start : scaled Jacobian at the position x_hat (also according to amount of seen tags)
				R_start : Covariance matrix scaled according to seen tags
				"""
				#print('ekf: get x5 = ' + str(ekf.get_x_hat()[5]))
		# publishing Position, Orientation and Covariance as output of the Extended-Kalman-Filter
		#print('ekfpublish: get x5 = ' + str(ekf.get_x_hat()[5]))
		ekf.ekf_publish(time_stamp, ekf.get_x_hat(), ekf.get_P_mat())
		#print('ekfdebugger: get x5 = ' + str(ekf.get_x_hat()[5]))
		# publishing the tf-transformation of the camera1-frame
		transforms_cam = []
		ekf_tf_cam = TransformStamped()
		camera_pose_quat_w = quaternion_from_euler(self._a, self._b, self._g)
		ekf_tf_cam.header.stamp  = time_stamp
		ekf_tf_cam.header.frame_id = 'map'
		ekf_tf_cam.child_frame_id = 'camera'+'{}'.format(self._cam_id)
		
		
		# changing world frame to NWU which is rviz's standart coordinate frame convention
		cam_location_NED = np.array([self._t_x, self._t_y, self._t_z]).reshape(3, 1)
		
		#EDN2NED = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape(3, 3)
		NED2CAM = np.array([[0, 1, 0],
				    [0, 0, 1],
				    [1, 0, 0]])
		NED2NWU = np.array([[1, 0, 0],
				    [0, -1, 0],
				    [0, 0, -1]])
		cam_location_NWU = NED2NWU.dot(cam_location_NED)
		
		# transform to py-quaternion notation
		camera_pose_quat_w_temp = Quaternion(np.asarray([camera_pose_quat_w[3], 
								 camera_pose_quat_w[0], 
								 camera_pose_quat_w[1], 
								 camera_pose_quat_w[2]]))
		# get rotation matrix from quaternion (py-quaternion notation)
		R_w2cam = camera_pose_quat_w_temp.rotation_matrix
		
		R_transformed_NWU = NED2NWU.dot(R_w2cam.dot(np.transpose(NED2CAM)))
		#R_transformed_NWU = NED2NWU.dot(R_transform)
		quat_NWU = Quaternion(matrix=R_transformed_NWU)
		# continue publishing with applied changes
		
		ekf_tf_cam.transform.translation.x = cam_location_NWU[0]
		ekf_tf_cam.transform.translation.y = cam_location_NWU[1]
		ekf_tf_cam.transform.translation.z = cam_location_NWU[2]
		ekf_tf_cam.transform.rotation.x = quat_NWU[1]
		ekf_tf_cam.transform.rotation.y = quat_NWU[2]
		ekf_tf_cam.transform.rotation.z = quat_NWU[3]
		ekf_tf_cam.transform.rotation.w = quat_NWU[0]
		

		transforms_cam.append(ekf_tf_cam)
		pub_tf_cam.sendTransform(transforms_cam)  # rename into name of the publisher
			
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		


	def compute_jacobian(self):
		x, y, z, p, t, s = sympy.symbols('x y z p t s')
		a = self._a
		b = self._b
		g = self._g
		t_x = self._t_x
		t_y = self._t_y
		t_z = self._t_z
		
		h = sympy.Matrix([[-t_x*sympy.sin(g)*sympy.cos(b) + t_y*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + t_z*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a)) - (x + 0.082*sympy.sin(s)*sympy.cos(t))*sympy.sin(g)*sympy.cos(b) + (sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a))*(z - 0.082*sympy.sin(p)*sympy.cos(s) - 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g))*(y + 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.082*sympy.cos(p)*sympy.cos(s))], [t_x*sympy.sin(b) - t_y*sympy.sin(a)*sympy.cos(b) + t_z*sympy.cos(a)*sympy.cos(b) + (x + 0.082*sympy.sin(s)*sympy.cos(t))*sympy.sin(b) - (y + 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.082*sympy.cos(p)*sympy.cos(s))*sympy.sin(a)*sympy.cos(b) + (z - 0.082*sympy.sin(p)*sympy.cos(s) - 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p))*sympy.cos(a)*sympy.cos(b)], [t_x*sympy.cos(b)*sympy.cos(g) + t_y*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + t_z*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g)) + (x + 0.082*sympy.sin(s)*sympy.cos(t))*sympy.cos(b)*sympy.cos(g) + (sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g))*(z - 0.082*sympy.sin(p)*sympy.cos(s) - 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a))*(y + 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.082*sympy.cos(p)*sympy.cos(s))], [-t_x*sympy.sin(g)*sympy.cos(b) + t_y*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + t_z*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a)) - (x - 0.082*sympy.sin(s)*sympy.cos(t))*sympy.sin(g)*sympy.cos(b) + (sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a))*(z + 0.082*sympy.sin(p)*sympy.cos(s) + 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g))*(y - 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.082*sympy.cos(p)*sympy.cos(s))], [t_x*sympy.sin(b) - t_y*sympy.sin(a)*sympy.cos(b) + t_z*sympy.cos(a)*sympy.cos(b) + (x - 0.082*sympy.sin(s)*sympy.cos(t))*sympy.sin(b) - (y - 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.082*sympy.cos(p)*sympy.cos(s))*sympy.sin(a)*sympy.cos(b) + (z + 0.082*sympy.sin(p)*sympy.cos(s) + 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p))*sympy.cos(a)*sympy.cos(b)], [t_x*sympy.cos(b)*sympy.cos(g) + t_y*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + t_z*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g)) + (x - 0.082*sympy.sin(s)*sympy.cos(t))*sympy.cos(b)*sympy.cos(g) + (sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g))*(z + 0.082*sympy.sin(p)*sympy.cos(s) + 0.082*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a))*(y - 0.082*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.082*sympy.cos(p)*sympy.cos(s))], [-t_x*sympy.sin(g)*sympy.cos(b) + t_y*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + t_z*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a)) - (x + 0.082*sympy.sin(t))*sympy.sin(g)*sympy.cos(b) + (y - 0.082*sympy.sin(p)*sympy.cos(t))*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + (z + 0.082*sympy.cos(p)*sympy.cos(t))*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a))], [t_x*sympy.sin(b) - t_y*sympy.sin(a)*sympy.cos(b) + t_z*sympy.cos(a)*sympy.cos(b) + (x + 0.082*sympy.sin(t))*sympy.sin(b) - (y - 0.082*sympy.sin(p)*sympy.cos(t))*sympy.sin(a)*sympy.cos(b) + (z + 0.082*sympy.cos(p)*sympy.cos(t))*sympy.cos(a)*sympy.cos(b)], [t_x*sympy.cos(b)*sympy.cos(g) + t_y*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + t_z*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g)) + (x + 0.082*sympy.sin(t))*sympy.cos(b)*sympy.cos(g) + (y - 0.082*sympy.sin(p)*sympy.cos(t))*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + (z + 0.082*sympy.cos(p)*sympy.cos(t))*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g))], [-t_x*sympy.sin(g)*sympy.cos(b) + t_y*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + t_z*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a)) - (x + 0.173*sympy.cos(s)*sympy.cos(t))*sympy.sin(g)*sympy.cos(b) + (sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a))*(z + 0.173*sympy.sin(p)*sympy.sin(s) - 0.173*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g))*(y + 0.173*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) + 0.173*sympy.sin(s)*sympy.cos(p))], [t_x*sympy.sin(b) - t_y*sympy.sin(a)*sympy.cos(b) + t_z*sympy.cos(a)*sympy.cos(b) + (x + 0.173*sympy.cos(s)*sympy.cos(t))*sympy.sin(b) - (y + 0.173*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) + 0.173*sympy.sin(s)*sympy.cos(p))*sympy.sin(a)*sympy.cos(b) + (z + 0.173*sympy.sin(p)*sympy.sin(s) - 0.173*sympy.sin(t)*sympy.cos(p)*sympy.cos(s))*sympy.cos(a)*sympy.cos(b)], [t_x*sympy.cos(b)*sympy.cos(g) + t_y*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + t_z*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g)) + (x + 0.173*sympy.cos(s)*sympy.cos(t))*sympy.cos(b)*sympy.cos(g) + (sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g))*(z + 0.173*sympy.sin(p)*sympy.sin(s) - 0.173*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a))*(y + 0.173*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) + 0.173*sympy.sin(s)*sympy.cos(p))], [-t_x*sympy.sin(g)*sympy.cos(b) + t_y*(-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g)) + t_z*(sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a)) - (x - 0.24*sympy.cos(s)*sympy.cos(t))*sympy.sin(g)*sympy.cos(b) + (sympy.sin(a)*sympy.cos(g) + sympy.sin(b)*sympy.sin(g)*sympy.cos(a))*(z - 0.24*sympy.sin(p)*sympy.sin(s) + 0.24*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (-sympy.sin(a)*sympy.sin(b)*sympy.sin(g) + sympy.cos(a)*sympy.cos(g))*(y - 0.24*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.24*sympy.sin(s)*sympy.cos(p))], [t_x*sympy.sin(b) - t_y*sympy.sin(a)*sympy.cos(b) + t_z*sympy.cos(a)*sympy.cos(b) + (x - 0.24*sympy.cos(s)*sympy.cos(t))*sympy.sin(b) - (y - 0.24*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.24*sympy.sin(s)*sympy.cos(p))*sympy.sin(a)*sympy.cos(b) + (z - 0.24*sympy.sin(p)*sympy.sin(s) + 0.24*sympy.sin(t)*sympy.cos(p)*sympy.cos(s))*sympy.cos(a)*sympy.cos(b)], [t_x*sympy.cos(b)*sympy.cos(g) + t_y*(sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a)) + t_z*(sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g)) + (x - 0.24*sympy.cos(s)*sympy.cos(t))*sympy.cos(b)*sympy.cos(g) + (sympy.sin(a)*sympy.sin(g) - sympy.sin(b)*sympy.cos(a)*sympy.cos(g))*(z - 0.24*sympy.sin(p)*sympy.sin(s) + 0.24*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (sympy.sin(a)*sympy.sin(b)*sympy.cos(g) + sympy.sin(g)*sympy.cos(a))*(y - 0.24*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.24*sympy.sin(s)*sympy.cos(p))]])


		h_c = sympy.Matrix(h)
		variables = sympy.Matrix([x, y, z, p, t, s])

                # H(x) Jacobian of h(x)
		return h_c.jacobian(variables)  


	
