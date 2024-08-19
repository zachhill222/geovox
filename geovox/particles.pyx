########## import special C functions and tools from utility.pyx ##########
# from libc.math cimport beta
from geovox.utility cimport Vector3, Box, Quaternion

#All particles should be convex

############# base class ################
cdef class Shape3D:
	def __init__(self, Vector3 center, Box bbox):
		self.center = center
		self.bbox = bbox


############ sphere #################
# cdef class Sphere(Shape3D):
# 	pass
