#cython: language_level=3, boundscheck=False, wraparound=False

from geovox.linalg.vector cimport Vector

cdef class Matrix: #class for dense matrices
	cdef double* _data
	cdef unsigned int _nrows, _ncols

	cpdef bint issquare(self)
	cpdef double trace(self)
	cpdef Vector row(self, unsigned int i)
	cpdef Vector col(self, unsigned int j)
	cpdef Vector dot(self, Vector other)
	cpdef void fill(self, double val)
	cpdef void hilbert(self) #make hilbert matrix for testing
	cpdef void eye(self)