from geovox.linalg.vector cimport Vector
from geovox.util.box cimport Box
from geovox.util.quaternion cimport Quaternion
from geovox.geometry.particles cimport particle_t, Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.util.optimize cimport _closest_point_neldermead, _best_point


cdef int NONE_DEPTH = -9999
# cdef void _divide(Node N)


cdef class Node:
	cdef readonly bint isdivided
	cdef readonly list children
	cdef public Node parent, root
	cdef public set points
	cdef public list particle_list, particle_ind_list
	cdef readonly int depth
	cdef public int totaldepth, maxdepth
	cdef Box bbox
	cdef public int nvert  #number of vertices contained in a single marker. probably shouldn't be public.

	# cpdef void cdivide(self)
	cpdef void divide(self)
	cpdef int sumnvert(self) #set each parent node nvert to be the sum of the children
	cpdef Node getnode(self, Vector point) #get first leaf that contains the point

	cpdef list leaflist(self) #return a list of all leaves
	cpdef void to_vtk(self, str filename) #write to a .vtk file
	cpdef void printall(self)
	cpdef void appendparticle(self, particle_t P)


cdef class Assembly(Node):
	cpdef Box getbbox(self)
	cpdef void process(self)
	cpdef void read(self, str filename, str pType = *)