from geovox.utilities.vector3 cimport Vector3

################# class for quaternions ###################################
cdef class Quaternion:
	cdef public double q0 #scalar component
	cdef public Vector3 qv #vector component
	cpdef Quaternion conj(self)
	cpdef Quaternion inv(self)
	cpdef Vector3 rotate(self, Vector3 point)
	cpdef Quaternion normalize(self)
	cpdef Quaternion setrotation(self, double angle, Vector3 axis)


