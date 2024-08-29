from geovox.utilities.vector3 cimport Vector3

ctypedef fused init_row_t:
	Vector3
	list

cdef inline int INDEX(int i, int j): return 3*i+j

cdef class Matrix3:
	cdef public double[9] _data

	cpdef double getval(self, int i, int j)
	cpdef void setval(self, int i, int j, double val)
	cpdef Matrix3 copy(self)
	cpdef Vector3 row(self, int i)
	cpdef Vector3 col(self, int j)

	cpdef Matrix3 transpose(self)
	cpdef double trace(self)
	cpdef double det(self)
	cdef double _cofactor(self, int i, int j)
	cpdef Matrix3 adj(self)
	cpdef Matrix3 inv(self)

	cpdef Vector3 dot(self, Vector3 other) #M*other
	cpdef Vector3 leftdot(self, Vector3 other) #other.transpose * M