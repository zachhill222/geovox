cdef class IndexIJK:
	cdef __cinit__(self, bint i, bint j, bint k):



cdef class VtkVoxel:
	def __cinit__(self):
		self.vtk_id = 11

		self._ijk = [[0,0,0],
					 [1,0,0],
					 [1,1,0],
					 [0,1,0],
					 [0,0,1],
					 [1,0,1],
					 [1,1,1],
					 [0,1,1],
					]


	@property
	def vtk_id(self): return self.vtk_id

	cdef bint* ijk(self, int n): return self._ijk[n] #get ijk/xyz index from linear .vtk index

	cdef int n(self, bint[3] ijk):
		if   (ijk[0]==0 and ijk[1]==0 and ijk[2]==0): return 0
		elif (ijk[0]==1 and ijk[1]==0 and ijk[2]==0): return 1
		elif (ijk[0]==1 and ijk[1]==1 and ijk[2]==0): return 2
		elif (ijk[0]==0 and ijk[1]==1 and ijk[2]==0): return 3
		elif (ijk[0]==0 and ijk[1]==0 and ijk[2]==1): return 4
		elif (ijk[0]==1 and ijk[1]==0 and ijk[2]==1): return 5
		elif (ijk[0]==1 and ijk[1]==1 and ijk[2]==1): return 6
		elif (ijk[0]==0 and ijk[1]==1 and ijk[2]==1): return 7
