################# import basic math from C ####################
from libc.math cimport sqrt, fabs #double to double



################# class for 3D points ####################
cdef class Vector3:
	def __init__(self, double x, double y, double z):
		self.x = x
		self.y = y
		self.z = z

	cpdef Vector3 copy(self): return Vector3(self.x, self.y, self.z)

	#vector operations
	cpdef Vector3 cross(self, Vector3 other): #return self X other (cross product)
		cdef double X, Y, Z
		X = self.y*other.z - self.z*other.y
		Y = self.z*other.x - self.x*other.z
		Z = self.x*other.y - self.y*other.x
		return Vector3(X,Y,Z)

	cpdef double dot(self, Vector3 other): #dot product
		return self.x*other.x + self.y*other.y + self.z*other.z

	#norms
	cpdef double infNorm(self):
		cdef double norm = fabs(self.x)
		if fabs(self.y) > norm: norm = fabs(self.y)
		if fabs(self.z) > norm: norm = fabs(self.z)
		return norm

	@property #call by <Vector3>.abs2
	def abs2(self):
		return self.x*self.x + self.y*self.y + self.z*self.z

	@property
	def prod(self):
		return self.x*self.y*self.z

	def __abs__(self): #compute 2-norm
		return sqrt(self.abs2)

	cpdef Vector3 normalize(self):
		cdef double C = (1.0/abs(self))
		self.x *= C
		self.y *= C 
		self.z *= C
		return self
	
	#arithmetic operations
	def __add__(self, Vector3 other):
		return Vector3(self.x+other.x, self.y+other.y, self.z+other.z)

	def __iadd__(self, Vector3 other):
		self.x += other.x
		self.y += other.y
		self.z += other.z
		return self

	def __sub__(self, Vector3 other):
		return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)

	def __isub__(self, Vector3 other):
		self.x-=other.x
		self.y-=other.y
		self.z-=other.z
		return self

	def __rmul__(self, double c): #scalar product: <double>*<Vector 3>
		return Vector3(c*self.x, c*self.y, c*self.z)

	def __mul__(self, Vector3 other): #component-wise product <Vector3>*<Vector 3>
		return Vector3(self.x*other.x, self.y*other.y, self.z*other.z)

	def __imul__(self, Vector3 other): #component-wise produc <Vector3>*=<Vector 3>
		self.x*=other.x
		self.y*=other.y
		self.z*=other.z 
		return self

	def __truediv__(self, Vector3 other): #component-wise division <Vector3>/<Vector3>
		return Vector3(self.x/other.x, self.y/other.y, self.z/other.z)

	def __itruediv__(self, Vector3 other):
		self.x/=other.x
		self.y/=other.y
		self.z/=other.z
		return self

	def __neg__(self):
		return Vector3(-self.x, -self.y, -self.z)

	#comparisons (cone-based)
	def __lt__(self, Vector3 other): #check if in the open negative cone
		return (self.x<other.x and self.y<other.y and self.z<other.z)

	def __le__(self, Vector3 other): #check if in the closed negative cone
		return (self.x<=other.x and self.y<=other.y and self.z<=other.z)
	
	def __eq__(self, Vector3 other): #check points are equal
		return (self.x==other.x and self.y==other.y and self.z==other.z)

	def __ne__(self, Vector3 other): #check if points are not equal
		return (self.x!=other.x and self.y!=other.y and self.z!=other.z)

	def __gt__(self, Vector3 other): #check if in the open positive cone
		return (self.x>other.x and self.y>other.y and self.z>other.z)

	def __ge__(self, Vector3 other): #check if in the closed positive cone
		return (self.x>=other.x and self.y>=other.y and self.z>=other.z)

	#representations
	def __str__(self):
		return "({0}, {1}, {2})".format(self.x, self.y, self.z)

	def __repr__(self):
		return "Vector3"+self.__str__()

	#hashing (for using sets, faster than lists)
	def __key(self): return (self.x, self.y, self.z)
	def __hash__(self): return hash(self.__key())


	#getitem
	def __getitem__(self, key):
		if key==0: return self.x
		elif key==1: return self.y
		elif key==2: return self.z

	def __len__(self): return 3


