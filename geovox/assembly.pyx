from geovox.particles cimport Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.utilities cimport Box



cdef class Node: #node of octree
	def __cinit__(self, Box box, int level):
		self.box = box
		self.level = level


cdef class Assembly:
	def __init__(self, Box box):
		self.box = box
		self.level = 0
	cpdef 