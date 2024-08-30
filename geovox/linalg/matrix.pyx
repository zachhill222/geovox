from libc.stdlib cimport malloc, free

cdef class Matrix:
	def __cinit__(self, *args):
		if len(args) == 1:
			self._nrows = args[0]
			self._ncols = 1
		else:
			self._nrows = args[0]
			self._ncols = args[1]

		self._data = <double*> malloc( self._nrows*self._ncols * sizeof(double))

	def __dealloc__(self):
		if self._data is not NULL:
			free(self._data)

	#get, set, len, etc.
	def __getitem__(self, key): return self._data[key[0]*self._ncols + key[1]]
	def __setitem__(self, key, double value): self._data[key[0]*self._ncols + key[1]] = value
	def __len__(self): return self._nrows*self._ncols

	@property
	def shape(self): return (self._nrows, self._ncols)
	cpdef bint issquare(self): return self._nrows==self._ncols

	cpdef double trace(self):
		if not self.issquare(): raise Exception("Trace called on a non-square Matrix")
		cdef unsigned int i
		val = self[0,0]
		for i in range(1,self._ncols):
			val += self[i,i]
		return val

	cpdef Vector row(self, unsigned int i):
		cdef Vector vec = Vector(self._ncols)
		cdef unsigned int j
		for j in range(self._ncols):
			vec[j] = self[i,j]
		return vec

	cpdef Vector col(self, unsigned int j):
		cdef Vector vec = Vector(self._ncols)
		cdef unsigned int i
		for i in range(self._ncols):
			vec[i] = self[i,j]
		return vec

	def __mul__(self, Matrix other): #self@other
		if not self._ncols==other._nrows: raise Exception("Dimension mismatch")

		cdef Matrix result = Matrix(self._nrows, other._ncols)
		cdef Vector col

		cdef unsigned int i, j
		for j in range(other._ncols):
			col = other.col(j)
			for i in range(self._nrows):
				result[i,j] = self.row(i).dot(col)
		return result

	cpdef Vector dot(self, Vector other): #Matrix*Vector
		if not self._nrows==len(other): raise Exception("Dimension mismatch")

		cdef Vector result = Vector(self._ncols)

		cdef unsigned int i
		for i in range(self._nrows):
			result[i] = self.row(i).dot(other)
		return result

	cpdef void fill(self, double val):
		cdef unsigned int i
		for i in range(self._nrows*self._ncols):
			self._data[i] = val

	cpdef void hilbert(self):
		cdef unsigned int i, j
		for j in range(self._ncols):
			for i in range(self._nrows):
				self[i,j] = 1.0/(i+j+1.0)

	def __str__(self):
		string = ["[\n"]
		if self._nrows <= 4:
			for i in range(self._nrows-1):
				string.append("\t"+str(self.row(i))+"\n")
		else:
			for i in range(3):
				string.append("\t"+str(self.row(i))+"\n")
			string.append("\t...\n")
		string.append("\t"+str(self.row(self._nrows-1))+"\n]")

		return "".join(string)	





