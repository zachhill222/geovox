################# import basic math from C ####################
from libc.math cimport sqrt, fabs #double to double



################# class for 3D points ####################
cdef class Vector3:
	def __init__(self, double x, double y, double z):
		self._data[0] = x
		self._data[1] = y
		self._data[2] = z

	#access, copying, etc.
	cpdef Vector3 copy(self): return Vector3(self._data[0], self._data[1], self._data[2])

	def __getitem__(self, key):
		if   key==0: return self._data[0]
		elif key==1: return self._data[1]
		elif key==2: return self._data[2]

	def __setitem__(self, key, value):
		if   key==0: self._data[0] = value
		elif key==1: self._data[1] = value
		elif key==2: self._data[2] = value

	def __len__(self): return 3

	@property
	def x(self): return self._data[0]
	@property
	def y(self): return self._data[1]
	@property
	def z(self): return self._data[2]

	@x.setter
	def x(self, double value): self._data[0] = value
	@y.setter
	def y(self, double value): self._data[1] = value
	@z.setter
	def z(self, double value): self._data[2] = value


	#vector operations
	cpdef Vector3 cross(self, Vector3 other): #return self X other (cross product)
		cdef double X, Y, Z
		X = self._data[1]*other._data[2] - self._data[2]*other._data[1]
		Y = self._data[2]*other._data[0] - self._data[0]*other._data[2]
		Z = self._data[0]*other._data[1] - self._data[1]*other._data[0]
		return Vector3(X,Y,Z)

	cpdef double dot(self, Vector3 other): #dot product
		return self._data[0]*other._data[0] + self._data[1]*other._data[1] + self._data[2]*other._data[2]

	#norms
	cpdef double infNorm(self):
		cdef double norm = fabs(self._data[0])
		if fabs(self._data[1]) > norm: norm = fabs(self._data[1])
		if fabs(self._data[2]) > norm: norm = fabs(self._data[2])
		return norm

	@property #call by <Vector3>.abs2
	def abs2(self):
		return self._data[0]*self._data[0] + self._data[1]*self._data[1] + self._data[2]*self._data[2]

	@property
	def prod(self):
		return self._data[0]*self._data[1]*self._data[2]

	def __abs__(self): #compute 2-norm
		return sqrt(self.abs2)

	cpdef Vector3 normalize(self):
		cdef double C = (1.0/abs(self))
		self._data[0] *= C
		self._data[1] *= C 
		self._data[2] *= C
		return self
	
	#arithmetic operations
	def __add__(self, Vector3 other):
		return Vector3(self._data[0]+other._data[0], self._data[1]+other._data[1], self._data[2]+other._data[2])

	def __iadd__(self, Vector3 other):
		self._data[0] += other._data[0]
		self._data[1] += other._data[1]
		self._data[2] += other._data[2]
		return self

	def __sub__(self, Vector3 other):
		return Vector3(self._data[0]-other._data[0], self._data[1]-other._data[1], self._data[2]-other._data[2])

	def __isub__(self, Vector3 other):
		self._data[0]-=other._data[0]
		self._data[1]-=other._data[1]
		self._data[2]-=other._data[2]
		return self

	def __rmul__(self, double c): #scalar product: <double>*<Vector 3>
		return Vector3(c*self._data[0], c*self._data[1], c*self._data[2])

	def __mul__(self, Vector3 other): #component-wise product <Vector3>*<Vector 3>
		return Vector3(self._data[0]*other._data[0], self._data[1]*other._data[1], self._data[2]*other._data[2])

	def __imul__(self, Vector3 other): #component-wise produc <Vector3>*=<Vector 3>
		self._data[0]*=other._data[0]
		self._data[1]*=other._data[1]
		self._data[2]*=other._data[2] 
		return self

	def __truediv__(self, Vector3 other): #component-wise division <Vector3>/<Vector3>
		return Vector3(self._data[0]/other._data[0], self._data[1]/other._data[1], self._data[2]/other._data[2])

	def __itruediv__(self, Vector3 other):
		self._data[0]/=other._data[0]
		self._data[1]/=other._data[1]
		self._data[2]/=other._data[2]
		return self

	def __neg__(self):
		return Vector3(-self._data[0], -self._data[1], -self._data[2])

	#comparisons (cone-based)
	def __lt__(self, Vector3 other): #check if in the open negative cone
		return (self._data[0]<other._data[0] and self._data[1]<other._data[1] and self._data[2]<other._data[2])

	def __le__(self, Vector3 other): #check if in the closed negative cone
		return (self._data[0]<=other._data[0] and self._data[1]<=other._data[1] and self._data[2]<=other._data[2])
	
	def __eq__(self, Vector3 other): #check points are equal
		return (self._data[0]==other._data[0] and self._data[1]==other._data[1] and self._data[2]==other._data[2])

	def __ne__(self, Vector3 other): #check if points are not equal
		return (self._data[0]!=other._data[0] and self._data[1]!=other._data[1] and self._data[2]!=other._data[2])

	def __gt__(self, Vector3 other): #check if in the open positive cone
		return (self._data[0]>other._data[0] and self._data[1]>other._data[1] and self._data[2]>other._data[2])

	def __ge__(self, Vector3 other): #check if in the closed positive cone
		return (self._data[0]>=other._data[0] and self._data[1]>=other._data[1] and self._data[2]>=other._data[2])

	#representations
	def __str__(self):
		return "({0}, {1}, {2})".format(self._data[0], self._data[1], self._data[2])

	def __repr__(self):
		return "Vector3"+self.__str__()

	#hashing (for using sets, faster than lists)
	def __key(self): return (self._data[0], self._data[1], self._data[2])
	def __hash__(self): return hash(self.__key())


