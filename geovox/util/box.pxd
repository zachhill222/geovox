from geovox.linalg.vector cimport Vector

################# class for 3D bounding boxes #############################
cdef class Box: #closed box
	cdef Vector low, high
	cpdef Vector vertex(self, int n) #get vertex in .vtk VOXEL ordering (different from hexahedron)
	cpdef bint contains(self, Vector point)
	cpdef bint strictcontains(self, Vector point)
	cpdef bint intersects(self, Box other)
	cpdef Vector facecenter(self, int n) #get n-th face center, n=0,1,...,5
	cpdef Vector facenormal(self, int n) #get n-th outward unit vector, n=0,1,...,5
	cpdef Box copy(self)

