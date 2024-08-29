################# imports ####################
from libc.math cimport sqrt, fabs #double to double
from libc.stdlib cimport malloc, free


###### Helper functions ################
cdef double reduce_sum(double* DATA, unsigned int LEN): #sum first LEN elements of DATA
	cdef unsigned int HALF_LEN
	cdef double VAL, RIGHT, LEFT


	cdef unsigned int i
	if LEN < 16:
		VAL = DATA[0]
		for i in range(1,LEN):
			VAL += DATA[i]
		return VAL

	if LEN%2: #odd
		VAL = DATA[LEN-1]
		HALF_LEN = (LEN-1)//2
	else: #even
		VAL = 0.0
		HALF_LEN = LEN//2

	LEFT = reduce_sum(&DATA[0], HALF_LEN)
	RIGHT = reduce_sum(&DATA[HALF_LEN], HALF_LEN)
	return VAL + LEFT + RIGHT


##### Vector with fixed, arbitrary length #############
cdef class Vector: #arbitrary dimensions
	# def __cinit__(self, unsigned int _len):
	def __cinit__(self, *args, **kwargs):
		#set length
		if "_len" in kwargs: #length is explicitly set
			self._len = kwargs["_len"]
		elif len(args) == 1: #only 1 value is passed, use it as _len
			self._len = args[0]
		elif len(args) > 1: #lots of values were passed, use them as the _data and set _len=len(_data)
			self._len = len(args)
		else: #no way to get length, use default of 3, as this is a 3D package
			self._len = 3

		self._data = <double*> malloc( self._len * sizeof(double))

	def __dealloc__(self):
		if self._data is not NULL:
			free(self._data)

	def __init__(self, *args, **kwargs):
		cdef unsigned int i
		
		if len(args) > 1: #data to fill
			for i in range(len(args)):
				self._data[i] = args[i]

	#get, set, len, etc.
	def __getitem__(self, unsigned int key): return self._data[key]
	def __setitem__(self, unsigned int key, double value): self._data[key] = value
	def __len__(self): return self._len

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
	
	cpdef Vector copy(self):
		cdef Vector newcopy = Vector(self._len)
		cdef unsigned int i
		for i in range(self._len):
			newcopy[i] = self._data[i]
		return newcopy

	#arithmetic
	def __imul__(self, Vector other): #in-place element-wise mulitplication of two vectors
		if not len(self)==len(other): raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			self._data[i] *= other[i]
		return self

	def __mul__(self, Vector other): #element-wise multiplication of two vectors
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = self._data[i]*other[i]

		return result

	def __truediv__(self, Vector other): #component-wise division <Vector>/<Vector>
		if not len(self)==len(other): raise Exception("Vector dimension mismatch.")

		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = self._data[i]/other[i]

		return result

	def __itruediv__(self, Vector other):
		cdef unsigned int i
		for i in range(self._len):
			self._data[i]/=other[i]
		return self

	def __neg__(self):
		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = -self._data[i]

		return result

	def __isub__(self, Vector other):
		if not len(self)==len(other): raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			self._data[i] -= other[i]
		return self
			
	def __sub__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = self._data[i]-other[i]

		return result

	def __iadd__(self, Vector other):
		if not len(self)==len(other): raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			self._data[i] += other[i]
		return self
			
	def __add__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = self._data[i]+other[i]

		return result

	cpdef void scaleby(self, double scalar): #in-place multiplication by a scalar
		cdef unsigned int i
		for i in range(self._len):
			self._data[i] *= scalar

	def __rmul__(self, double scalar): #multiplication by a scalar
		cdef Vector result = Vector(self._len)

		cdef unsigned int i
		for i in range(self._len):
			result[i] = scalar*self._data[i]

		return result

	#useful functions
	cpdef void fill(self, double scalar): #fill all entries with scalar
		cdef unsigned int i
		for i in range(self._len):
			self._data[i] = scalar

	cpdef double infNorm(self):
		cdef double val = abs(self._data[0])

		cdef unsigned int i
		for i in range(1,self._len):
			val = max(val, fabs(self._data[i]))

		return val

	cpdef Vector cross(self, Vector other): #return self X other (cross product)
		if not (self._len==3 and other._len==3): raise Exception("Cross product is only for Vectors with _len=3")

		cdef Vector result = Vector(3)
		cdef double X, Y, Z
		result._data[0] = self._data[1]*other._data[2] - self._data[2]*other._data[1]
		result._data[1] = self._data[2]*other._data[0] - self._data[0]*other._data[2]
		result._data[2] = self._data[0]*other._data[1] - self._data[1]*other._data[0]
		return result

	cpdef double dot(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")
		cdef Vector product = self*other
		return reduce_sum(&product._data[0], self._len)

	cpdef double abs2(self): return self.dot(self)

	def __abs__(self): return sqrt(self.abs2())

	cpdef Vector normalize(self): #normalize to a unit vector in 2-norm
		cdef double scalar = 1.0/sqrt(self.abs2())
		return self.__rmul__(scalar)

	cpdef void i_normalize(self): #in-place normalize
		cdef double scalar = 1.0/sqrt(self.abs2())
		self.scaleby(scalar)

	#comparisons (element-wise, positive/negative cones)
	def __eq__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			if not self._data[i] == other[i]: return False

		return True


	def __lt__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			if self._data[i] >= other[i]: return False

		return True

	def __gt__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			if self._data[i] <= other[i]: return False

		return True

	def __le__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			if self._data[i] > other[i]: return False

		return True

	def __ge__(self, Vector other):
		if not self._len == other._len: raise Exception("Vector dimension mismatch.")

		cdef unsigned int i
		for i in range(self._len):
			if self._data[i] < other[i]: return False

		return True

	def __str__(self):
		string = ["("]
		if self._len <= 4:
			for i in range(self._len-1):
				string.append(f"{self._data[i]}, ")
		else:
			for i in range(3):
				string.append(f"{self._data[i]}, ")
			string.append("..., ")
		string.append(f"{self._data[self._len-1]})")

		return "".join(string)


	def __repr__(self): return "Vector"+self.__str__()

	def __key(self):
		cdef unsigned int i
		cdef list keylist = []
		for i in range(self._len): keylist.append(self._data[i])
		return (tuple(keylist), self._len)
	
	def __hash__(self): return hash(self.__key())


