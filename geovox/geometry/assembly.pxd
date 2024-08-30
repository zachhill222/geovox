from geovox.linalg.vector cimport Vector
from geovox.util.box cimport Box
from geovox.geometry.particles cimport particle_t
from geovox.util.optimize cimport _closest_point_neldermead, _closest_point_gradient

cdef int NONE_DEPTH = -999

cdef class Node:
	cdef bint isdivided
	cdef readonly list children
	cdef public Node parent
	cdef public Node root
	cdef public set points
	cdef public list particle_list #only used in root
	cdef public list particle_ind_list
	cdef depth
	cdef Box bbox
	cdef public int nvert #number of vertices contained in a single marker. probably shouldn't be public.
	cpdef void divide(self)
	# cpdef void prune(self) #remove self and all children nodes (and their children, etc...)
	# cpdef Node getnode(self, Vector point) #get first leaf that contains the point
	cpdef void insertparticle(self, particle_t P)

	cpdef list leaflist(self) #return a list of all leaves
	cpdef void voxelmesh(self, str filename) #write to a .vtk file


# cdef class Assembly:
# 	cdef public int maxdepth
# 	cdef Node root
