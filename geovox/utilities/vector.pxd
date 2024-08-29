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
	cpdef double dot(self, Vector other)
	cpdef Vector cross(self, Vector other)
	cpdef double abs2(self)
	
	cpdef void scaleby(self, double scalar)
	cpdef Vector normalize(self) #normalize to a unit vector in 2-norm
	cpdef void i_normalize(self) #in-place normalize