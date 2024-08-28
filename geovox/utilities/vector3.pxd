################# class for 3D points ######################################
cdef class Vector3:
	cdef public double x, y, z
	cpdef Vector3 copy(self)
	cpdef Vector3 cross(self, Vector3 other) #return self X other (cross product)
	cpdef double dot(self, Vector3 other)
	cpdef double infNorm(self)
	cpdef Vector3 normalize(self)