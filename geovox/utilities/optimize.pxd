from geovox.utilities.vector3 cimport Vector3
from geovox.utilities.box cimport Box

# ctypedef double (*f_type)(Vector3)

cdef class NelderMead:
	cdef object F
	cdef Box bounds
	cdef double R2
	cdef public double alpha, gamma, rho, sigma, target, penalty_weight
	cdef public int maxiter
	cdef list simplex
	cdef list fvals

	cpdef double minimize(self)
	cpdef void ordersimplex(self)
	cpdef void takestep(self)
	cpdef double evalF(self, Vector3 point)
	# cpdef Vector3 ensurebounds(self, Vector3 candidate, Vector3 source)
