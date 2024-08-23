cdef class IndexIJK:
	def __cinit__(self, int i, int j, int k):
		self._ijk = [i,j,k]

	def __getitem__(self, key):
		return self._ijk[key%3]
	def __setitem__(self, key, int value):
		self._ijk[key%3] = value

	@property
	def i(self): return self._ijk[0]
	@i.setter
	def i(self, int value): self._ijk[0] = value

	@property
	def j(self): return self._ijk[1]
	@j.setter
	def j(self, int value): self._ijk[1] = value

	@property
	def k(self): return self._ijk[2]
	@k.setter
	def k(self, int value): self._ijk[2] = value

	def __repr__(self): return f"IndexIJK({self.i}, {self.j}, {self.k})"



cdef class VtkVoxel:
	def __init__(self):
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

	cpdef IndexIJK ijk(self, int n):
		return IndexIJK(self._ijk[n][0], self._ijk[n][1], self._ijk[n][2]) #get ijk/xyz index from linear .vtk index

	cpdef int n(self, IndexIJK ijk):
		if   (ijk[0]==0 and ijk[1]==0 and ijk[2]==0): return 0
		elif (ijk[0]==1 and ijk[1]==0 and ijk[2]==0): return 1
		elif (ijk[0]==1 and ijk[1]==1 and ijk[2]==0): return 2
		elif (ijk[0]==0 and ijk[1]==1 and ijk[2]==0): return 3
		elif (ijk[0]==0 and ijk[1]==0 and ijk[2]==1): return 4
		elif (ijk[0]==1 and ijk[1]==0 and ijk[2]==1): return 5
		elif (ijk[0]==1 and ijk[1]==1 and ijk[2]==1): return 6
		elif (ijk[0]==0 and ijk[1]==1 and ijk[2]==1): return 7

	def __repr__(self): return "VtkVoxel()"

	def __str__(self):
		string = "\nVtkVoxel NODE ORDERING: HIDDEN NODE IS 1\n"
		string+= "\n             3 ---- 2 "
		string+= "\n Y,j         |\      \ "
		string+= "\n |           |  \      \ "
		string+= "\n O --- X,i   0   7 ---- 6"
		string+= "\n \            \  |      |"
		string+= "\n  \            \ |      |"
		string+= "\n   Z,k          \|      |"
		string+= "\n                 4 ---- 5\n"
		return string



