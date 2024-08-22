################# import basic math from C ####################
from libc.math cimport sqrt, fabs, acos, cos, sin #double to double



################# class for 3D points ####################
cdef class Vector3:
	# cdef double x, y, z

	def __init__(self, double x, double y, double z):
		self.x = x
		self.y = y
		self.z = z

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

##################### class for 3D bounding boxes #########################
cdef class Box: #non-degenerate closed box, faces parallel to coordinate planes
	# cdef Vector3 low, high

	def __init__(self, Vector3 low, Vector3 high):
		if not low < high:
			raise Exception("low < high must return True")
		self.low = low
		self.high = high

	# allow reading/writing of low and high from python
	@property
	def low(self): return self.low
	@property
	def high(self): return self.high
	@property
	def center(self): return 0.5*(self.low+self.high)

	@low.setter
	def low(self, Vector3 value):
		cdef Vector3 templow = Vector3(0,0,0)
		cdef Vector3 temphigh = Vector3(0,0,0)

		templow.x  = min(value.x, self.high.x)
		temphigh.x = max(value.x, self.high.x)

		templow.y  = min(value.y, self.high.y)
		temphigh.y = max(value.y, self.high.y)
		
		templow.z  = min(value.z, self.high.z)
		temphigh.z = max(value.z, self.high.z)
		
		if templow < temphigh:
			self.low  = templow
			self.high = temphigh
		else:
			print("setting low= "+repr(value)+" would result in an invalid Box. low was not set.")

	@high.setter
	def high(self, Vector3 value):
		cdef Vector3 templow = Vector3(0,0,0)
		cdef Vector3 temphigh = Vector3(0,0,0)

		templow.x  = min(value.x, self.low.x)
		temphigh.x = max(value.x, self.low.x)

		templow.y  = min(value.y, self.low.y)
		temphigh.y = max(value.y, self.low.y)

		templow.z  = min(value.z, self.low.z)
		temphigh.z = max(value.z, self.low.z)
		
		if templow < temphigh:
			self.low  = templow
			self.high = temphigh
		else:
			print("setting high= "+repr(value)+" would result in an invalid Box. high was not set.")


	cpdef Vector3 vertex(self, int n): #get vertex in .vtk ordering
		if n==0:   return Vector3(self.low.x, self.low.y, self.low.z)
		elif n==1: return Vector3(self.high.x, self.low.y, self.low.z)
		elif n==2: return Vector3(self.high.x, self.high.y, self.low.z)
		elif n==3: return Vector3(self.low.x, self.high.y, self.low.z)
		elif n==4: return Vector3(self.low.x, self.low.y, self.high.z)
		elif n==5: return Vector3(self.high.x, self.low.y, self.high.z)
		elif n==6: return Vector3(self.high.x, self.high.y, self.high.z)
		elif n==7: return Vector3(self.low.x, self.high.y, self.high.z)

		raise Exception("n must be between 0 and 7")


	# relation to other points/boxes
	cpdef bint contains(self, Vector3 point):
		return self.low <= point and point <= self.high

	cpdef bint intersects(self, Box other):
		if other.high < self.low or other.low > self.high: return False
		for n in range(8):
			if self.contains(other.vertex(n)): return True
		return False


	# shifting
	def __add__(self, Vector3 other): #shift box by other
		return Box(self.low+other, self.high+other)

	def __iadd__(self, Vector3 other):
		self.low+=other
		self.high+=other
		return self

	def __sub__(self, Vector3 other): #shift box by -other
		return Box(self.low-other, self.high-other)

	def __isub__(self, Vector3 other):
		self.low-=other
		self.high-=other
		return self

	# representation
	def __str__(self):
		string = "Box:\n"
		string+="\tlow=  "+self.low.__str__()+"\n"
		string+="\thigh= "+self.high.__str__()
		return string

	def __repr__(self):
		return "Box("+repr(self.low)+", "+repr(self.high)+")"




######################## class for quaternions ######################
cdef class Quaternion:
	# cdef double q0 #scalar component
	# cdef Vector3 qv #vector component

	def __init__(self, double q0, Vector3 qv):
		self.q0 = q0
		self.qv = qv

	cpdef Quaternion conj(self):
		return Quaternion(self.q0, -self.qv)

	cpdef Quaternion inv(self):
		cdef double C = 1.0/self.abs2
		return Quaternion(C*self.q0, (-C)*self.qv)


	# rotations
	@property
	def angle(self):
		if abs(self.abs2 - 1) > 1e-8:
			print(f"WARNING: Q= {repr(self)} is not a rotation quaternion")
		return 2.0*acos(self.q0)

	@property
	def axis(self):
		if abs(self.abs2 - 1) > 1e-8:
			print(f"WARNING: Q= {repr(self)} is not a rotation quaternion")
		return (1.0/abs(self.qv))*self.qv

	cpdef Quaternion normalize(self):
		cdef double C = 1.0/abs(self)
		self.q0 = C*self.q0
		self.qv = C*self.qv
		return self

	cpdef Quaternion setrotation(self, double angle, Vector3 axis):
		self.qv = sin(0.5*angle)*axis.normalize()
		self.q0 = cos(0.5*angle)
		return self

	cpdef Vector3 rotate(self, Vector3 point):
		cdef Quaternion V = Quaternion(0.0, point)
		V = self*V*self.conj()
		return V.qv


	# magnitude
	@property
	def abs2(self):
		return self.q0*self.q0 + self.qv.x*self.qv.x + self.qv.y*self.qv.y + self.qv.z*self.qv.z

	def __abs__(self):
		return sqrt(self.abs2)


	# arithmetic
	def __mul__(self, Quaternion other): #<Quaternion>*<Quaternion>
		cdef double Q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		cdef Vector3 QV = self.q0*other.qv + other.q0*self.qv + self.qv.cross(other.qv)
		return Quaternion(Q0, QV)

	def __rmul__(self, double c): #scalar multiplication: <double>*<Quaternion>
		return Quaternion(c*self.q0, c*self.qv)

	def __truediv__(self, Quaternion other): #multiplication by the inverse: <Quaternion>/<Quaternion>
		return self*other.inv()

	def __add__(self, Quaternion other):
		return Quaternion(self.q0+other.q0, self.qv+other.qv)

	def __sub__(self, Quaternion other):
		return Quaternion(self.q0-other.q0, self.qv-other.qv)

	def __neg__(self):
		return Quaternion(-self.q0, -self.qv)
	

	# represenation
	def __str__(self):
		string = "Quaternion:\n"
		string+= "\tq0= {0}\n".format(self.q0)
		string+= "\tqv= " + self.qv.__str__()
		return string

	def __repr__(self):
		return "Quaternion({0}, ".format(self.q0)+repr(self.qv)+")"








