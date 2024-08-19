################# import basic math from C ####################
from libc.math cimport sqrt, fabs #double to double


################# class for 3D points ####################
cdef class Vector3:
	cdef double x, y, z

	def __cinit__(self, double x, double y, double z):
		self.x = x
		self.y = y
		self.z = z

	cpdef Vector3 cross(self, Vector3 other): #return self X other (cross product)
		cdef double X, Y, Z
		X = self.y*other.z - self.z*other.y
		Y = self.z*other.x - self.x*other.z
		Z = self.x*other.y - self.y*other.x
		return Vector3(X,Y,Z)

	def __add__(self, Vector3 other):
		return Vector3(self.x+other.x, self.y+other.y, self.z+other.z)

	def __sub__(self, Vector3 other):
		return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)

	cpdef double dot(self, Vector3 other): #dot product
		return self.x*other.x + self.y*other.y + self.z*other.z

	def __mul__(self, double c): #scalar product
		return Vector3(c*self.x, c*self.y, c*self.z)

	def __rmul__(self, double c): #scalar product
		return Vector3(c*self.x, c*self.y, c*self.z)

	def __iadd__(self, Vector3 other):
		self.x += other.x
		self.y += other.y
		self.z += other.z
		return self

	def __imul__(self, double c):
		self.x *= c
		self.y *= c
		self.z *= c
		return self

	def __neg__(self):
		return Vector3(-self.x, -self.y, -self.z)

	cpdef double abs2(self):
		return self.x*self.x + self.y*self.y + self.z*self.z

	def __abs__(self): #compute 2-norm
		return sqrt(self.abs2())

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

	def __str__(self):
		return "({0}, {1}, {2})".format(self.x, self.y, self.z)

	def __repr__(self):
		return "Vector3"+self.__str__()

##################### class for 3D bounding boxes #########################
cdef class Box: #closed box
	cdef Vector3 low, high

	def __cinit__(self, Vector3 low, Vector3 high):
		if not low < high:
			raise Exception("low < high must return True")
		self.low = low
		self.high = high

	cpdef Vector3 vertex(self, int i): #get vertex in .vtk ordering
		if i==0:   return Vector3(self.low.x, self.low.y, self.low.z)
		elif i==1: return Vector3(self.high.x, self.low.y, self.low.z)
		elif i==2: return Vector3(self.high.x, self.high.y, self.low.z)
		elif i==3: return Vector3(self.low.x, self.high.y, self.low.z)
		elif i==4: return Vector3(self.low.x, self.low.y, self.high.z)
		elif i==5: return Vector3(self.high.x, self.low.y, self.high.z)
		elif i==6: return Vector3(self.high.x, self.high.y, self.high.z)
		elif i==7: return Vector3(self.low.x, self.high.y, self.high.z)

		raise Exception("i must be between 0 and 7")

	cpdef bint contains(self, Vector3 point):
		return self.low <= point and point <= self.high

	cpdef bint intersects(self, Box other):
		if other.high < self.low or other.low > self.high: return False
		for i in range(8):
			if self.contains(other.vertex(i)): return True
		return False

	def __str__(self):
		string = "Box:\n"
		string+="\tlow=  "+self.low.__str__()+"\n"
		string+="\thigh= "+self.high.__str__()
		return string

	def __repr__(self):
		return "Box("+repr(self.low)+", "+repr(self.high)+")"




######################## class for quaternions ######################
cdef class Quaternion:
	cdef double q0 #scalar component
	cdef Vector3 qv #vector component

	def __cinit__(self, double q0, Vector3 qv):
		self.q0 = q0
		self.qv = qv

	def __mul__(self, Quaternion other):
		cdef double Q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		cdef Vector3 QV = self.q0*other.qv + self.qv*other.q0 + self.qv.cross(other.qv)
		return Quaternion(Q0, QV)

	def __rmul__(self, Quaternion other):
		cdef double Q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		cdef Vector3 QV = self.q0*other.qv + self.qv*other.q0 - self.qv.cross(other.qv)
		return Quaternion(Q0, QV)

	def __add__(self, Quaternion other):
		return Quaternion(self.q0+other.q0, self.qv+other.qv)

	def __sub__(self, Quaternion other):
		return Quaternion(self.q0-other.q0, self.qv-other.qv)

	cpdef Quaternion conj(self):
		return Quaternion(self.q0, -self.qv)

	cpdef double abs2(self):
		return self.q0*self.q0 + self.qv.x*self.qv.x + self.qv.y*self.qv.y + self.qv.z*self.qv.z

	def __abs__(self):
		return sqrt(self.abs2())

	cpdef Quaternion inv(self):
		cdef double C = 1.0/self.abs2()
		return Quaternion(C*self.q0, (-C)*self.qv)

	cpdef Vector3 rotate(self, Vector3 point):
		cdef Quaternion V = Quaternion(0.0, point)
		V = self*V*self.conj()
		return V.qv

	def __str__(self):
		string = "Quaternion:\n"
		string+= "\tq0= {0}\n".format(self.q0)
		string+= "\tqv= " + self.qv.__str__()
		return string

	def __repr__(self):
		return "Quaternion({0}, ".format(self.q0)+repr(self.qv)+")"

	cpdef normalize(self):
		cdef double C = 1.0/self.__abs__()
		self.q0 *= C
		self.qv *= C






