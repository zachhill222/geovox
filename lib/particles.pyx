########## import special C functions and tools from utility.pyx ##########
# from libc.math cimport beta
cimport utility as util

#All particles should be convex

############# base class ################
cdef class Shape3D:
	cdef util.Box bbox #bounding box
	cdef util.Vector3 center #center

	def __cinit__(self, util.Vector3 center, util.Box bbox):
		self.center = center
		self.bbox = bbox


############ sphere #################
cdef class Sphere(Shape3D):
	pass
