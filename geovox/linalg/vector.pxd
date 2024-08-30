#cython: language_level=3, boundscheck=False, wraparound=False

cdef double reduce_sum(double* DATA, unsigned int LEN) #sum first LEN elements of DATA


################# generic vector class #####################################
cdef class Vector:
	# attributes
	cdef double* _data
	cdef unsigned int _len

	# functions accessible from Python
	cpdef void fill(self, double scalar)
	cpdef Vector copy(self)
	cpdef double infNorm(self)
	cpdef double sum(self) #return sum of elements
	cpdef double dot(self, Vector other)
	cpdef Vector cross(self, Vector other)
	cpdef double abs2(self)
	
	cpdef void scaleby(self, double scalar)
	cpdef Vector normalize(self) #normalize to a unit vector in 2-norm
	cpdef void i_normalize(self) #in-place normalize

	# data useful for meshing
	cdef public list containedby #list of particles that contain this point. not included in hash


################## useful functions #########################################
cpdef Vector el_min(Vector v, Vector u) #element-wise minimum
cpdef Vector el_max(Vector v, Vector u) #element-wise maximum