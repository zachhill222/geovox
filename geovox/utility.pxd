################# class for 3D points ####################
cdef class Vector3:
	cdef double x, y, z
	cpdef Vector3 cross(self, Vector3 other) #return self X other (cross product)
	cpdef double dot(self, Vector3 other)
	cpdef double abs2(self)



##################### class for 3D bounding boxes #########################
cdef class Box: #closed box
	cdef Vector3 low, high
	cpdef Vector3 vertex(self, int i)
	cpdef bint contains(self, Vector3 point)
	cpdef bint intersects(self, Box other)




######################## class for quaternions ######################
cdef class Quaternion:
	cdef double q0 #scalar component
	cdef Vector3 qv #vector component
	cpdef Quaternion conj(self)
	cpdef double abs2(self)
	cpdef Quaternion inv(self)
	cpdef Vector3 rotate(self, Vector3 point)
	cpdef normalize(self)





