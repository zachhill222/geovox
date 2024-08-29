from geovox.utilities.vector cimport Vector

################# class for quaternions ###################################
cdef class Quaternion:
	cdef public double q0 #scalar component
	cdef public Vector qv #vector component
	cpdef Quaternion conj(self)
	cpdef Quaternion inv(self)
	cpdef Vector rotate(self, Vector point)
	cpdef Quaternion normalize(self)
	cpdef Quaternion setrotation(self, double angle, Vector axis)


