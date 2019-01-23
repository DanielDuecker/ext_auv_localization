# class for easy camera addition
import numpy as np
import math
import sympy
import time
import symengine
from numpy.linalg import inv
from tf.transformations import quaternion_from_euler # defined q = (x, y, z, w)
from tf.transformations import quaternion_inverse
from tf.transformations import quaternion_multiply
from pyquaternion import Quaternion  # defined q = (w, x, y, z)


class camera:

	def __init__(self, cam_id, t_x, t_y, t_z, a, b, g, R_mat):
		self._cam_id = cam_id
		self._t_x = t_x
		self._t_y = t_y
		self._t_z = t_z
		self._a = a
		self._b = b
		self._g = g
		self._R_mat = R_mat
		# class variables
		self._H_jacobi_sym = self.compute_jacobian()		# symbolic Jacobian of h (variables have not been subsidied)

	def get_H_Jac_sym(self):
		return	self._H_jacobi_sym

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



	def h_cam(self, x_cube):
		# defining parameters
		x_cube = x_cube.reshape(1, 6)
    		x_cube = x_cube[0]
    		x = x_cube[0]
    		y = x_cube[1]
   		z = x_cube[2]
    		p = x_cube[3]  # phi
    		t = x_cube[4]  # theta
    		s = x_cube[5]  # psi

		# vectors from cube-center to tag-center (cube-system)
		offset1 = np.array([0, -0.075, 0, 1]).reshape(4, 1)
		offset2 = np.array([-0.075, 0, 0, 1]).reshape(4, 1)
		offset3 = np.array([0, 0.075, 0, 1]).reshape(4, 1)
		offset4 = np.array([0, 0, -0.075, 1]).reshape(4, 1)
		offset5 = np.array([0, 0, 0.075, 1]).reshape(4, 1)
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
		R_cube2world = inv(R_z_cube.dot(R_y_cube.dot(R_x_cube)))	# Rotation matrix consisting of all 3 axis rotations
		T_cube2world = np.block([					# Transformation matrix from cube to world system
					[(R_cube2world), np.array([x, y, z]).reshape(3, 1)],
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
		R_world2cam = R_z.dot(R_y).dot(R_x)		# rotation matrix from world to camera
		offsetcam = R_world2cam.dot(-np.array([self._t_x, self._t_y, self._t_z]).reshape(3, 1)) # old positions self._t_x, self._t_y, self._t_z
		T_world2cam = np.block([					# Transformation matrix from world to camera
					[R_world2cam, offsetcam],
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
			h_temp = T_world2cam.dot(T_cube2world.dot((offset[j_counter:j_counter+4, 0:1]).reshape(4, 1)))
			h[i:i+3, 0:1] = h_temp[0:3, 0:1]
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
			h_temp_sympy = (T_world2cam_sympy*(T_cube2world_sympy*(offset[j_counter:j_counter+4, 0:1]))).reshape(4, 1)
			h_sympy[i:i+3, 0:1] = h_temp_sympy[0:3, 0:1]
			i += 3
			j_counter +=4
		print 'h_sympy:'
		print h_sympy
		"""

		# returning h_at_x
		return h							# 15x1 vector showing the position of all tags depending on the cube's position
		
		


	def compute_jacobian(self):
		x, y, z, p, t, s = sympy.symbols('x y z p t s')
		h = sympy.Matrix([[self._t_x*sympy.cos(self._b)*sympy.cos(self._g) + self._t_y*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + self._t_z*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g)) + (x + 0.075*sympy.sin(s)*sympy.cos(t))*sympy.cos(self._b)*sympy.cos(self._g) + (sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g))*(z - 0.075*sympy.sin(p)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a))*(y + 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.075*sympy.cos(p)*sympy.cos(s))], [-self._t_x*sympy.sin(self._g)*sympy.cos(self._b) + self._t_y*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + self._t_z*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a)) - (x + 0.075*sympy.sin(s)*sympy.cos(t))*sympy.sin(self._g)*sympy.cos(self._b) + (sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a))*(z - 0.075*sympy.sin(p)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g))*(y + 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.075*sympy.cos(p)*sympy.cos(s))], [self._t_x*sympy.sin(self._b) - self._t_y*sympy.sin(self._a)*sympy.cos(self._b) + self._t_z*sympy.cos(self._a)*sympy.cos(self._b) + (x + 0.075*sympy.sin(s)*sympy.cos(t))*sympy.sin(self._b) - (y + 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) - 0.075*sympy.cos(p)*sympy.cos(s))*sympy.sin(self._a)*sympy.cos(self._b) + (z - 0.075*sympy.sin(p)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p))*sympy.cos(self._a)*sympy.cos(self._b)], [self._t_x*sympy.cos(self._b)*sympy.cos(self._g) + self._t_y*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + self._t_z*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g)) + (x - 0.075*sympy.cos(s)*sympy.cos(t))*sympy.cos(self._b)*sympy.cos(self._g) + (sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g))*(z - 0.075*sympy.sin(p)*sympy.sin(s) + 0.075*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a))*(y - 0.075*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.cos(p))], [-self._t_x*sympy.sin(self._g)*sympy.cos(self._b) + self._t_y*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + self._t_z*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a)) - (x - 0.075*sympy.cos(s)*sympy.cos(t))*sympy.sin(self._g)*sympy.cos(self._b) + (sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a))*(z - 0.075*sympy.sin(p)*sympy.sin(s) + 0.075*sympy.sin(t)*sympy.cos(p)*sympy.cos(s)) + (-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g))*(y - 0.075*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.cos(p))], [self._t_x*sympy.sin(self._b) - self._t_y*sympy.sin(self._a)*sympy.cos(self._b) + self._t_z*sympy.cos(self._a)*sympy.cos(self._b) + (x - 0.075*sympy.cos(s)*sympy.cos(t))*sympy.sin(self._b) - (y - 0.075*sympy.sin(p)*sympy.sin(t)*sympy.cos(s) - 0.075*sympy.sin(s)*sympy.cos(p))*sympy.sin(self._a)*sympy.cos(self._b) + (z - 0.075*sympy.sin(p)*sympy.sin(s) + 0.075*sympy.sin(t)*sympy.cos(p)*sympy.cos(s))*sympy.cos(self._a)*sympy.cos(self._b)], [self._t_x*sympy.cos(self._b)*sympy.cos(self._g) + self._t_y*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + self._t_z*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g)) + (x - 0.075*sympy.sin(s)*sympy.cos(t))*sympy.cos(self._b)*sympy.cos(self._g) + (sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g))*(z + 0.075*sympy.sin(p)*sympy.cos(s) + 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a))*(y - 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.075*sympy.cos(p)*sympy.cos(s))], [-self._t_x*sympy.sin(self._g)*sympy.cos(self._b) + self._t_y*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + self._t_z*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a)) - (x - 0.075*sympy.sin(s)*sympy.cos(t))*sympy.sin(self._g)*sympy.cos(self._b) + (sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a))*(z + 0.075*sympy.sin(p)*sympy.cos(s) + 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p)) + (-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g))*(y - 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.075*sympy.cos(p)*sympy.cos(s))], [self._t_x*sympy.sin(self._b) - self._t_y*sympy.sin(self._a)*sympy.cos(self._b) + self._t_z*sympy.cos(self._a)*sympy.cos(self._b) + (x - 0.075*sympy.sin(s)*sympy.cos(t))*sympy.sin(self._b) - (y - 0.075*sympy.sin(p)*sympy.sin(s)*sympy.sin(t) + 0.075*sympy.cos(p)*sympy.cos(s))*sympy.sin(self._a)*sympy.cos(self._b) + (z + 0.075*sympy.sin(p)*sympy.cos(s) + 0.075*sympy.sin(s)*sympy.sin(t)*sympy.cos(p))*sympy.cos(self._a)*sympy.cos(self._b)], [self._t_x*sympy.cos(self._b)*sympy.cos(self._g) + self._t_y*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + self._t_z*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g)) + (x - 0.075*sympy.sin(t))*sympy.cos(self._b)*sympy.cos(self._g) + (y + 0.075*sympy.sin(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + (z - 0.075*sympy.cos(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g))], [-self._t_x*sympy.sin(self._g)*sympy.cos(self._b) + self._t_y*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + self._t_z*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a)) - (x - 0.075*sympy.sin(t))*sympy.sin(self._g)*sympy.cos(self._b) + (y + 0.075*sympy.sin(p)*sympy.cos(t))*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + (z - 0.075*sympy.cos(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a))], [self._t_x*sympy.sin(self._b) - self._t_y*sympy.sin(self._a)*sympy.cos(self._b) + self._t_z*sympy.cos(self._a)*sympy.cos(self._b) + (x - 0.075*sympy.sin(t))*sympy.sin(self._b) - (y + 0.075*sympy.sin(p)*sympy.cos(t))*sympy.sin(self._a)*sympy.cos(self._b) + (z - 0.075*sympy.cos(p)*sympy.cos(t))*sympy.cos(self._a)*sympy.cos(self._b)], [self._t_x*sympy.cos(self._b)*sympy.cos(self._g) + self._t_y*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + self._t_z*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g)) + (x + 0.075*sympy.sin(t))*sympy.cos(self._b)*sympy.cos(self._g) + (y - 0.075*sympy.sin(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.sin(self._b)*sympy.cos(self._g) + sympy.sin(self._g)*sympy.cos(self._a)) + (z + 0.075*sympy.cos(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.sin(self._g) - sympy.sin(self._b)*sympy.cos(self._a)*sympy.cos(self._g))], [-self._t_x*sympy.sin(self._g)*sympy.cos(self._b) + self._t_y*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + self._t_z*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a)) - (x + 0.075*sympy.sin(t))*sympy.sin(self._g)*sympy.cos(self._b) + (y - 0.075*sympy.sin(p)*sympy.cos(t))*(-sympy.sin(self._a)*sympy.sin(self._b)*sympy.sin(self._g) + sympy.cos(self._a)*sympy.cos(self._g)) + (z + 0.075*sympy.cos(p)*sympy.cos(t))*(sympy.sin(self._a)*sympy.cos(self._g) + sympy.sin(self._b)*sympy.sin(self._g)*sympy.cos(self._a))], [self._t_x*sympy.sin(self._b) - self._t_y*sympy.sin(self._a)*sympy.cos(self._b) + self._t_z*sympy.cos(self._a)*sympy.cos(self._b) + (x + 0.075*sympy.sin(t))*sympy.sin(self._b) - (y - 0.075*sympy.sin(p)*sympy.cos(t))*sympy.sin(self._a)*sympy.cos(self._b) + (z + 0.075*sympy.cos(p)*sympy.cos(t))*sympy.cos(self._a)*sympy.cos(self._b)]])
		h_c = sympy.Matrix(h)
		variables = sympy.Matrix([x, y, z, p, t, s])
		H_jac = h_c.jacobian(variables)  #Jacobian of h(x)
		return H_jac


	def get_R_mat(self):
		return self._R_mat

	
	def H_jac_at_x(self, x_w_hat, H_jac):  # evaluate the jacobian H from a camera at estimate x_w_hat
		# in: current position estimate x_w_hat
    		# out: jacobian evaluated at x_w_hat
    		x_w_hat = x_w_hat.reshape(1, 6)
    		x_w_hat = x_w_hat[0]
    		x, y, z, p, t, s = sympy.symbols('x y z p t s')
    		x_w_hat_help = sympy.Matrix([x_w_hat[0], x_w_hat[1], x_w_hat[2], x_w_hat[3], x_w_hat[4], x_w_hat[5]])  # converting the incoming numpy array into a sympy array
    		H_jac_at_x_sympy = H_jac.subs(zip([x, y, z, p, t, s], [x_w_hat_help[0], x_w_hat_help[1], x_w_hat_help[2], x_w_hat_help[3], x_w_hat_help[4], x_w_hat_help[5]]))
    		H_jac_at_x = np.array(H_jac_at_x_sympy).astype(np.float64)
    		return H_jac_at_x
