################# import basic math from C ####################
from libc.math cimport sqrt, acos, cos, sin #double to double



######################## class for quaternions ######################
cdef class Quaternion:
	# cdef double q0 #scalar component
	# cdef Vector qv #vector component

	def __init__(self, double q0, Vector qv):
		self.q0 = q0
		self.qv = qv

	cpdef Quaternion conj(self):
		return Quaternion(self.q0, -self.qv)

	cpdef Quaternion inv(self):
		cdef double C = 1.0/self.abs2
		return Quaternion(C*self.q0, (-C)*self.qv)


	# rotations
	@property
	def angle(self):
		if abs(self.abs2 - 1) > 1e-8:
			print(f"WARNING: Q= {repr(self)} is not a rotation quaternion")
		return 2.0*acos(self.q0)

	@property
	def axis(self):
		if abs(self.abs2 - 1) > 1e-8:
			print(f"WARNING: Q= {repr(self)} is not a rotation quaternion")
		return (1.0/abs(self.qv))*self.qv

	cpdef Quaternion normalize(self):
		cdef double C = 1.0/abs(self)
		self.q0 = C*self.q0
		self.qv = C*self.qv
		return self

	cpdef Quaternion setrotation(self, double angle, Vector axis):
		self.qv = sin(0.5*angle)*axis.normalize()
		self.q0 = cos(0.5*angle)
		return self

	cpdef Vector rotate(self, Vector point):
		cdef Quaternion V = Quaternion(0.0, point)
		V = self*V*self.conj()
		return V.qv


	# magnitude
	@property
	def abs2(self):
		return self.q0*self.q0 + self.qv.x*self.qv.x + self.qv.y*self.qv.y + self.qv.z*self.qv.z

	def __abs__(self):
		return sqrt(self.abs2)


	# arithmetic
	def __mul__(self, Quaternion other): #<Quaternion>*<Quaternion>
		cdef double Q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		cdef Vector QV = self.q0*other.qv + other.q0*self.qv + self.qv.cross(other.qv)
		return Quaternion(Q0, QV)

	def __imul__(self, Quaternion other):
		cdef double Q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		self.qv = self.q0*other.qv + other.q0*self.qv + self.qv.cross(other.qv)
		self.q0 = Q0
		return self


	def __rmul__(self, double c): #scalar multiplication: <double>*<Quaternion>
		return Quaternion(c*self.q0, c*self.qv)

	def __truediv__(self, Quaternion other): #multiplication by the inverse: <Quaternion>/<Quaternion>
		return self*other.inv()

	def __add__(self, Quaternion other):
		return Quaternion(self.q0+other.q0, self.qv+other.qv)

	def __sub__(self, Quaternion other):
		return Quaternion(self.q0-other.q0, self.qv-other.qv)

	def __neg__(self):
		return Quaternion(-self.q0, -self.qv)
	

	# represenation
	def __str__(self):
		string = "Quaternion:\n"
		string+= "\tq0= {0}\n".format(self.q0)
		string+= "\tqv= " + self.qv.__str__()
		return string

	def __repr__(self):
		return "Quaternion({0}, ".format(self.q0)+repr(self.qv)+")"








