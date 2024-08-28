from geovox.utilities.vector3 cimport Vector3

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


