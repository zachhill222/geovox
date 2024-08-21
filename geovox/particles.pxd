############# import classes from utility ############
from geovox.utility cimport Vector3, Box, Quaternion


cdef class Shape3D:
	cdef Box bbox #bounding box
	cdef Vector3 center #center
	cdef double volume #volume



cdef class Sphere(Shape3D):
	cdef double r, r2
	cdef bint contains(self, Vector3 point)


cdef class Prism(Shape3D):
	cdef Vector3 R #radii (distance from center to faces)
	cdef Quaternion Q #quaternion for rotation FROM global coordinates TO local coordinates
	cdef Box prism
	cdef bint contains(self, Vector3 point)


cdef class Ellipsoid(Prism):
	cdef bint contains(self, Vector3 point)


cdef class SuperEllipsoid(Prism):
	cdef double[2] eps #shape parameters
	cdef double e0, e1, e2 #powers
	cdef bint contains(self, Vector3 point)