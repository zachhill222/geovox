############# import classes from utility ############
from geovox.utility cimport Vector3, Box, Quaternion

############# base class ################
cdef class Shape3D:
	cdef Box bbox #bounding box
	cdef Vector3 center #center


############ sphere #################
# cdef class Sphere(Shape3D):
# 	pass
