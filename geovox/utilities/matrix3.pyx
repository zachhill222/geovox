cdef class Matrix3:
	def __init__(self, row0, row1, row2):
		cdef int j
		for j from 0 <= j < 3: self._data[INDEX(0,j)] = row0[j]
		for j from 0 <= j < 3: self._data[INDEX(1,j)] = row1[j]
		for j from 0 <= j < 3: self._data[INDEX(2,j)] = row2[j]


	#standard access
	def __getitem__(self, key): #return key-th row vector
		if   key == 0: return Vector3(self._data[0], self._data[1], self._data[2])
		elif key == 1: return Vector3(self._data[3], self._data[4], self._data[5])
		elif key == 2: return Vector3(self._data[6], self._data[7], self._data[8])
		raise Exception("Index is out of bounds in Matrix3.__getitem__")

	cpdef double getval(self, int i, int j): return self._data[INDEX(i,j)]
	cpdef void setval(self, int i, int j, double val): self._data[INDEX(i,j)] = val

	cpdef Matrix3 copy(self): return Matrix3(self[0], self[1], self[2])

	cpdef Vector3 row(self, int i): return self[i]
	cpdef Vector3 col(self, int j):
		if   j == 0: return Vector3(self._data[0], self._data[3], self._data[6])
		elif j == 1: return Vector3(self._data[1], self._data[4], self._data[7])
		elif j == 2: return Vector3(self._data[2], self._data[5], self._data[8])
		raise Exception("Index is out of bounds in Matrix3.col")


	#representation
	def __repr__(self): return f"Matrix3({self[0]}, {self[1]}, {self[2]})"
	def __str__(self):
		string = ""
		for i in range(3):
			for j in range(3):
				string += f"\t{self._data[INDEX(i,j)]}"
			string += "\n"
		return string


	#unary operations
	cpdef Matrix3 transpose(self):
		return Matrix3(self.col(0), self.col(1), self.col(2))

	cpdef double trace(self): return self._data[0]+self._data[4]+self._data[8]

	cpdef double det(self): return self[0].dot(self[1].cross(self[2]))

	cdef double _cofactor(self, int i, int j):
		return self._data[INDEX((i-1)%3, (j-1)%3)]*self._data[INDEX((i+1)%3, (j+1)%3)]\
				   - self._data[INDEX((i+1)%3, (j-1)%3)]*self._data[INDEX((i-1)%3, (j+1)%3)]

	cpdef Matrix3 adj(self): #adjugate
		cdef Matrix3 cofac = Matrix3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
		cdef int i, j
		for i from 0<= i < 3:
			for j from 0 <= j < 3:
				cofac._data[INDEX(i,j)] = self._cofactor(j,i) #need the transpose
		return cofac

	cpdef Matrix3 inv(self):
		cdef double det = self.det()
		if det==0: raise Exception("Singular matrix")
		else: return (1.0/det)*self.adj()



	#binary operations
	def __add__(self, other):
		return Matrix3(self[0]+other[0], self[1]+other[1], self[2]+other[2])
	def __sub__(self, other):
		return Matrix3(self[0]-other[0], self[1]-other[1], self[2]-other[2])
	def __neg__(self):
		return Matrix3(-self[0], -self[1], -self[2])


	def __mul__(self, other):
		cdef prod = Matrix3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
		cdef int i, j
		cdef Vector3 col, row


		for j from 0 <= j < 3:
			col = other.col(j)
			for i from 0 <= i < 3:
				prod.setval(i,j,col.dot(self[i])) #should be able to set prod._data directly?
		return prod

	def __rmul__(self, scalar):
		cdef Matrix3 prod = self.copy()
		cdef int k
		for k from 0 <= k < 9: prod._data[k] *= scalar
		return prod

	def __matmul__(self, other):
		return other[0]*self.col(0) + other[1]*self.col(1) + other[2]*self.col(2)

	cpdef Vector3 dot(self, Vector3 other): #M*other
		cdef Vector3 val 
		val = other[0]*self.col(0) + other[1]*self.col(1) + other[2]*self.col(2)
		return val

	cpdef Vector3 leftdot(self, Vector3 other): #other.transpose * M
		cdef Vector3 val
		val = other[0]*self[0] + other[1]*self[1] + other[2]*self[2]
		return val
		