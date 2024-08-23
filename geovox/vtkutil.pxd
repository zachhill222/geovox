cdef class IndexIJK:
	cdef int[3] _ijk


cdef class VtkVoxel:
	cdef int vtk_id
	cdef bint[8][3] _ijk
	cpdef IndexIJK ijk(self, int n) #get ijk/xyz index from linear .vtk index
	cpdef int n(self, IndexIJK ijk) #get .vtk linear node from ijk/xyz index (low=0, high=1)