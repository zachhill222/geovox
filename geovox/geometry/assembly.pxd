from geovox.utilities.vector3 cimport Vector3
from geovox.utilities.box cimport Box
from geovox.geometry.particles cimport Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.utilities.optimize cimport NelderMead

cdef int NONE_DEPTH = -999

ctypedef fused particle_t:
	Sphere
	Prism
	Ellipsoid
	SuperEllipsoid



cdef class Node:
	cdef bint isdivided
	cdef list children
	cdef public Node root
	cdef public set points
	cdef public list particle_list
	cdef depth
	cdef Box bbox
	cdef public int nvert #number of vertices contained in a single marker. probably shouldn't be public.
	cpdef void divide(self)
	cpdef Node getnode(self, Vector3 point)
	cpdef void insertparticle(self, particle_t P)
	cpdef bint contains(self, Vector3 point)

	cpdef list leaflist(self) #return a list of all leaves
	cpdef void voxelmesh(self, str filename) #write to a .vtk file


# cdef class Assembly:
# 	cdef public int maxdepth
# 	cdef Node root
