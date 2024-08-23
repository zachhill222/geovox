from geovox.utilities cimport Box, Vector3


# node class: create a new class from this for specific application
cdef class OctreeNode:
	cdef int depth
	cdef bint isdivided
	cdef Node[8] children
	cdef Box bbox
	cdef int getvertexdata(self, Vector3 point)
	cdef int[8] vertexdata


# main class: create a new class from this for a specific application
cdef class Octree:
	def __cinit__(self, int maxdepth, Box bbox):
		self.maxdepth = maxdepth
		self.bbox     = bbox
