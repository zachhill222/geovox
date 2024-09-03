
#create a cdef class as a wrapper for Point
cdef class PyPoint:
	def __dealloc__(self):
		del self.thisptr

	def __cinit__(self, *args):
		cdef int _len
		if len(args) == 1: #only 1 value is passed, use it as _len
			_len = args[0]
		elif len(args) > 1: #lots of values were passed, use them as the _data and set _len=len(_data)
			_len = len(args)
		else: #no way to get length, use default of 3, as this is a 3D package
			_len = 3

		self.thisptr = new Point(_len)

	def __init__(self, *args):
		cdef int i
		if len(args) > 1: #data to fill
			for i in range(len(args)):
				self[i] = args[i]

	cpdef PyPoint copy(self):
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = self[i]
		return result

	def __abs__(self): return self.norm()
	def __len__(self): return self.thisptr._len
	def __getitem__(self, int key): return self.thisptr._data[key]
	def __setitem__(self, int key, double val): self.thisptr._data[key] = val

	@property
	def x(self): return self[0]
	@property
	def y(self): return self[1]
	@property
	def z(self): return self[2]

	@x.setter
	def x(self, double value): self[0] = value
	@y.setter
	def y(self, double value): self[1] = value
	@z.setter
	def z(self, double value): self[2] = value

	def fill(self, val): self.thisptr.fill(val)
	def max(self): return self.thisptr.max()
	def maxind(self): return self.thisptr.maxind()
	def min(self): return self.thisptr.min()
	def minind(self): return self.thisptr.minind()
	def sum(self): return self.thisptr.sum()
	def normalize(self):return cpp2py(self.thisptr.normalize())

	def norm2(self): return self.thisptr.norm2()
	def norm(self): return self.thisptr.norm()
	def infNorm(self): return self.thisptr.infnorm()
	def dist2(self, PyPoint other): return self.thisptr.dist2(py2cpp(other))
	def dist(self, PyPoint other): return self.thisptr.dist(py2cpp(other))
	def dist_inf(self, PyPoint other): return self.thisptr.dist_inf(py2cpp(other))

	def dot(self, PyPoint other): return self.thisptr.dot(py2cpp(other))
	def cross(self, PyPoint other): return cpp2py(self.thisptr.cross(py2cpp(other)))

	def __iadd__(self, PyPoint other):
		cdef int i
		for i in range(self.thisptr._len):
			self[i] += other[i]
		return self

	def __add__(self, PyPoint other):
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = self[i] + other[i]
		return result

	def __isub__(self, PyPoint other):
		cdef int i
		for i in range(self.thisptr._len):
			self[i] -= other[i]
		return self

	def __sub__(self, PyPoint other):
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = self[i] - other[i]
		return result

	def __imul__(self, PyPoint other): #in-place element-wise mulitplication of two vectors
		cdef int i
		for i in range(self.thisptr._len):
			self[i] *= other[i]
		return self

	def __mul__(self, PyPoint other): #element-wise multiplication of two vectors
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = self[i]*other[i]
		return result

	def __rmul__(self, double scalar): #multiplication by a scalar
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = scalar*self[i]
		return result

	def __truediv__(self, PyPoint other): #component-wise division <PyPoint>/<PyPoint>
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = self[i]/other[i]
		return result

	def __itruediv__(self, PyPoint other):
		cdef int i
		for i in range(self.thisptr._len):
			self[i]/=other[i]
		return self

	def __neg__(self):
		cdef PyPoint result = PyPoint(self.thisptr._len)
		cdef int i
		for i in range(self.thisptr._len):
			result[i] = -self[i]
		return result

	#comparisons (element-wise, positive/negative cones)
	def __eq__(self, PyPoint other):
		cdef bint result = True
		cdef int i
		for i in range(self.thisptr._len):
			if not self[i] == other[i]:
				result = False
				break
		return result


	def __lt__(self, PyPoint other):
		cdef bint result = True
		cdef int i
		for i in range(self.thisptr._len):
			if self[i] >= other[i]:
				result = False
				break
		return result

	def __gt__(self, PyPoint other):
		cdef bint result = True
		cdef int i
		for i in range(self.thisptr._len):
			if self[i] <= other[i]:
				result = False
				break
		return result

	def __le__(self, PyPoint other):
		cdef bint result = True
		cdef int i
		for i in range(self.thisptr._len):
			if self[i] > other[i]:
				result = False
				break
		return result

	def __ge__(self, PyPoint other):
		cdef bint result = True
		cdef int i
		for i in range(self.thisptr._len):
			if self[i] < other[i]:
				result = False
				break
		return result


	def __str__(self):
		string = ["("]
		if self.thisptr._len <= 4:
			for i in range(self.thisptr._len-1):
				string.append(f"{self[i]}, ")
		else:
			for i in range(3):
				string.append(f"{self[i]}, ")
			string.append("..., ")
		string.append(f"{self[self.thisptr._len-1]})")

		return "".join(string)


	def __repr__(self): return "PyPoint"+self.__str__()

	def __key(self):
		cdef int i
		cdef list keylist = [self[i] for i in range(self.thisptr._len)]
		keylist.append(self.thisptr._len)
		return tuple(keylist)
	
	def __hash__(self): return hash(self.__key())



@staticmethod
cdef PyPoint cpp2py(Point cpp_obj):
	py_obj = PyPoint(cpp_obj._len)
	cdef int i
	for i in range(cpp_obj._len):
		py_obj[i] = cpp_obj._data[i]
	return py_obj

@staticmethod
cdef Point py2cpp(PyPoint py_obj):
	cdef Point cpp_obj = Point(py_obj.thisptr._len)
	cdef int i
	for i in range(cpp_obj._len):
		cpp_obj._data[i] = py_obj[i]
	return cpp_obj


################## useful functions #########################################
cpdef PyPoint el_min(PyPoint v, PyPoint u):
	cdef PyPoint result = PyPoint(len(v))
	cdef unsigned int i
	for i in range(len(v)):
		result[i] = min(u[i], v[i])
	return result

cpdef PyPoint el_max(PyPoint v, PyPoint u):
	cdef PyPoint result = PyPoint(len(v))
	cdef unsigned int i
	for i in range(len(v)):
		result[i] = max(u[i], v[i])
	return result