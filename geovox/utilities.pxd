################# class for 3D points ######################################
cdef class Vector3:
	cdef public double x, y, z
	cpdef Vector3 copy(self)
	cpdef Vector3 cross(self, Vector3 other) #return self X other (cross product)
	cpdef double dot(self, Vector3 other)
	cpdef double infNorm(self)
	cpdef Vector3 normalize(self)



################# class for 3D bounding boxes #############################
cdef class Box: #closed box
	cdef Vector3 low, high
	cpdef Vector3 vertex(self, int n) #get vertex in .vtk VOXEL ordering (different from hexahedron)
	cpdef bint contains(self, Vector3 point)
	cpdef bint strictcontains(self, Vector3 point)
	cpdef bint intersects(self, Box other)
	cpdef Vector3 facecenter(self, int n) #get n-th face center, n=0,1,...,5
	cpdef Vector3 facenormal(self, int n) #get n-th outward unit vector, n=0,1,...,5
	# cpdef Vector3 facesurface(self, int n, double theta1, double theta2)




################# class for quaternions ###################################
cdef class Quaternion:
	cdef public double q0 #scalar component
	cdef public Vector3 qv #vector component
	cpdef Quaternion conj(self)
	cpdef Quaternion inv(self)
	cpdef Vector3 rotate(self, Vector3 point)
	cpdef Quaternion normalize(self)
	cpdef Quaternion setrotation(self, double angle, Vector3 axis)


