################# constants ################################################
cdef double PI = 3.141592653589793238462643383279

################# class for 3D points ######################################
cdef class Vector3:
	cdef public double x, y, z
	cpdef Vector3 cross(self, Vector3 other) #return self X other (cross product)
	cpdef double dot(self, Vector3 other)
	cpdef double infNorm(self)
	cpdef Vector3 normalize(self)



################# class for 3D bounding boxes #############################
cdef class Box: #closed box
	cdef Vector3 low, high
	cpdef Vector3 vertex(self, int i)
	cpdef bint contains(self, Vector3 point)
	cpdef bint intersects(self, Box other)




################# class for quaternions ###################################
cdef class Quaternion:
	cdef public double q0 #scalar component
	cdef public Vector3 qv #vector component
	cpdef Quaternion conj(self)
	cpdef Quaternion inv(self)
	cpdef Vector3 rotate(self, Vector3 point)
	cpdef Quaternion normalize(self)





