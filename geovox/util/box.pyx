################# imports ####################
# from geovox.vtkutil cimport VtkVoxel, IndexIJK
from geovox.linalg.vector cimport el_min, el_max


# vtk voxel for ordering vertices consistently
# cdef VtkVoxel VTK_VOXEL = VtkVoxel()

##################### class for 3D bounding boxes (similar to vtk voxel) #########################
cdef class Box: #non-degenerate closed box, faces parallel to coordinate planes
	# cdef Vector low, high

	def __init__(self, Vector low = Vector(-1.0,-1.0,-1.0), Vector high = Vector(1.0,1.0,1.0)): #low and high are any two opposite vertices of the box
		self.low  = el_min(low, high)
		self.high = el_max(low, high)

		if not self.low < self.high:
			raise Exception("Box must have positive volume. Specify two opposite vertices.")

	cpdef Box copy(self): return Box(self.low, self.high)


	# allow reading/writing of low and high from python
	@property
	def low(self): return self.low
	@property
	def high(self): return self.high
	@property
	def center(self): return 0.5*(self.low+self.high)
	@property
	def sidelength(self): return self.high-self.low
	@property
	def volume(self): return self.sidelength[0]*self.sidelength[1]*self.sidelength[2]
	@property
	def diam(self): return abs(self.sidelength)

	def __getitem__(self, key): #get vertex in .vtk VOXEL ordering (different from hexahedron)
		if   key==0: return Vector(self.low[0],  self.low[1],  self.low[2])
		elif key==1: return Vector(self.high[0], self.low[1],  self.low[2])
		elif key==2: return Vector(self.low[0],  self.high[1], self.low[2])
		elif key==3: return Vector(self.high[0], self.high[1], self.low[2])
		elif key==4: return Vector(self.low[0],  self.low[1],  self.high[2])
		elif key==5: return Vector(self.high[0], self.low[1],  self.high[2])
		elif key==6: return Vector(self.low[0],  self.high[1], self.high[2])
		elif key==7: return Vector(self.high[0], self.high[1], self.high[2])
		raise Exception("Index out of bounds when getting vertex of Box")

	@low.setter
	def low(self, Vector value):
		cdef Vector templow = el_min(self.low, value)
		cdef Vector temphigh = el_max(self.high, value)
		
		if templow < temphigh:
			self.low  = templow
			self.high = temphigh
		else:
			print("setting low= "+repr(value)+" would result in an invalid Box. low was not set.")

	@high.setter
	def high(self, Vector value):
		cdef Vector templow = el_min(self.low, value)
		cdef Vector temphigh = el_max(self.high, value)
		
		if templow < temphigh:
			self.low  = templow
			self.high = temphigh
		else:
			print("setting high= "+repr(value)+" would result in an invalid Box. high was not set.")


	cpdef Vector vertex(self, int n): return self[n]

	cpdef Vector facecenter(self, int n): #get n-th face center, n=0,1,...,5
		return self.center + 0.5*self.facenormal(n)*self.sidelength

	cpdef Vector facenormal(self, int n): #get n-th outward unit vector, n=0,1,...,5
		if   n==0: return Vector(-1.0, 0.0, 0.0)
		elif n==1: return Vector( 0.0,-1.0, 0.0)
		elif n==2: return Vector( 0.0, 0.0,-1.0)
		elif n==3: return Vector( 1.0, 0.0, 0.0)
		elif n==4: return Vector( 0.0, 1.0, 0.0)
		elif n==5: return Vector( 0.0, 0.0, 1.0)



	# relation to other points/boxes
	cpdef bint contains(self, Vector point):
		return self.low <= point and point <= self.high

	cpdef bint strictcontains(self, Vector point):
		return self.low < point and point < self.high

	cpdef bint intersects(self, Box other):
		for n in range(8):
			if self.contains(other.vertex(n)): return True
		for n in range(8): #this shouldn't be necessary?
			if other.contains(self.vertex(n)): return True
		return False


	# shifting
	def __add__(self, Vector other): #shift box by other
		return Box(self.low+other, self.high+other)

	def __iadd__(self, Vector other):
		self.low+=other
		self.high+=other
		return self

	def __sub__(self, Vector other): #shift box by -other
		return Box(self.low-other, self.high-other)

	def __isub__(self, Vector other):
		self.low-=other
		self.high-=other
		return self

	# scaling
	def __rmul__(self, double c): #scale from center
		cdef Vector low = self.low - self.center
		cdef Vector high = self.high - self.center
		return Box(c*low+self.center, c*high+self.center)

	# combine bounding boxes
	def __matmul__(self, Box other):
		cdef unsigned int i, dim
		dim = len(self.low)
		cdef Vector low  = Vector(dim)
		cdef Vector high = Vector(dim)
		for i in range(dim):
			low[i] = min(self.low[i], other.low[i])
			high[i] = max(self.high[i], other.high[i])
		return Box(low,high)

	def __imatmul__(self, Box other):
		cdef unsigned int i, dim
		dim = len(self.low)
		for i in range(dim):
			self.low[i] = min(self.low[i], other.low[i])
			self.high[i] = max(self.high[i], other.high[i])
		return self

	def __repr__(self):
		return "Box("+repr(self.low)+", "+repr(self.high)+")"



