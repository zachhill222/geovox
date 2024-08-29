from geovox.utilities.vector cimport Vector

ctypedef fused init_row_t:
	Vector
	list

cdef inline int INDEX(int i, int j): return 3*i+j

cdef class Matrix3:
	cdef public double[9] _data

	cpdef double getval(self, int i, int j)
	cpdef void setval(self, int i, int j, double val)
	cpdef Matrix3 copy(self)
	cpdef Vector row(self, int i)
	cpdef Vector col(self, int j)

	cpdef Matrix3 transpose(self)
	cpdef double trace(self)
	cpdef double det(self)
	cdef double _cofactor(self, int i, int j)
	cpdef Matrix3 adj(self)
	cpdef Matrix3 inv(self)

	cpdef Vector dot(self, Vector other) #M*other
	cpdef Vector leftdot(self, Vector other) #other.transpose * M