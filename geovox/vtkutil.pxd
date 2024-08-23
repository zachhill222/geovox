cdef class IndexIJK:
	cdef bint[3] _ijk


cdef class VtkVoxel:
	cdef int vtk_id
	cdef bint[8][3] _ijk
	cdef bint* ijk(self, int n) #get ijk/xyz index from linear .vtk index
	cdef int n(self, bint[3] ijk) #get .vtk linear node from ijk/xyz index (low=0, high=1)