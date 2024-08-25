from geovox.utilities cimport Box
from geovox.particles cimport Sphere, Prism, Ellipsoid, SuperEllipsoid


ctypedef fused particle_t:
	Sphere
	Prism
	Ellipsoid
	SuperEllipsoid



cdef class Node:
	cdef Node parent
	cdef bint isdivided
	cdef list children
	cdef list particle_list
	cdef depth
	cdef Box bbox
	cpdef void divide(self)

# cdef class Assembly:
# 	cdef Box bbox
# 	cdef int maxdepth

# 	cpdef 