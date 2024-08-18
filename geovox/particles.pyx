# contains logic for various particle types to build assemblies with
from libc.math cimport sin
from libc.math cimport acos
from libc.math cimport fabs
from libc.math cimport sqrt

cdef class Vector:
	

cdef class Quaternion:
	cdef double q0 #scalar part of quaternion
	cdef double[3] qv #vector part of quaternion

	def __cinit__(self, double q0, double[3] qv):
		self.q0 = q0
		self.qv = qv

	def __repr__(self):
		return "Quaternion({0}, [{1}, {2}, {3}])".format(self.q0, self.qv[0], self.qv[1], self.qv[2])

	def __str__(self):
		return "Quaternion:\t({}, {}, {}, {})\n".format(self.q0, self.qv[0], self.qv[1], self.qv[2])

	def checkRotation(self):
		if fabs(self.q0) > 1.0:
			return "NOT A VALID ROTATION"

		# get rotation theta/2
		double[3] u
		double half_theta = acos(self.q0)
		if half_theta == 0.0:
			u[0]=0.0
			u[1]=0.0
			u[2]=0.0
		else:
			double C = 1.0/sin(half_theta)
			u[0] = self.qv[0]*C
			u[1] = self.qv[1]*C
			u[2] = self.qv[2]*C

		double normu = sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2])

		return "Theta= {}\tU= [{}, {}, {}]\t|U|= {}".format(half_theta*2, u[0], u[1], u[2], normu)



	#Define standard arithmetic operations for quaternions: these return a new copy of a quaternion
	def __mul__(self, other):
		#return self*other quaternion product
		q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		qv = self.q0*other.qv + other.q0*self.qv + crossProd(self.qv, other.qv)
		return Quaternion(q0, qv, self.dtype)