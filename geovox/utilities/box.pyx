################# imports ####################
from geovox.vtkutil cimport VtkVoxel, IndexIJK


# vtk voxel for ordering vertices consistently
cdef VtkVoxel VTK_VOXEL = VtkVoxel()

##################### class for 3D bounding boxes (similar to vtk voxel) #########################
cdef class Box: #non-degenerate closed box, faces parallel to coordinate planes
	# cdef Vector3 low, high

	def __init__(self, Vector3 low, Vector3 high): #low and high are any two opposite vertices of the box
		self.low  = Vector3(min(low.x, high.x), min(low.y, high.y), min(low.z, high.z))
		self.high = Vector3(max(low.x, high.x), max(low.y, high.y), max(low.z, high.z))

		if not self.low < self.high:
			raise Exception("Box must have positive volume. Specify two opposite vertices.")

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
	def volume(self): return self.sidelength.prod

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


	cpdef Vector3 vertex(self, int n): #get vertex in .vtk VOXEL ordering (different from hexahedron)
		cdef IndexIJK ijk = VTK_VOXEL.ijk(n)

		cdef Vector3 vert = self.low.copy()
		
		if ijk[0]: vert.x = self.high.x
		if ijk[1]: vert.y = self.high.y
		if ijk[2]: vert.z = self.high.z

		return vert

	cpdef Vector3 facecenter(self, int n): #get n-th face center, n=0,1,...,5
		return self.center + 0.5*self.facenormal(n)*self.sidelength

	cpdef Vector3 facenormal(self, int n): #get n-th outward unit vector, n=0,1,...,5
		if   n==0: return Vector3(-1.0, 0.0, 0.0)
		elif n==1: return Vector3( 0.0,-1.0, 0.0)
		elif n==2: return Vector3( 0.0, 0.0,-1.0)
		elif n==3: return Vector3( 1.0, 0.0, 0.0)
		elif n==4: return Vector3( 0.0, 1.0, 0.0)
		elif n==5: return Vector3( 0.0, 0.0, 1.0)


	# relation to other points/boxes
	cpdef bint contains(self, Vector3 point):
		return self.low <= point and point <= self.high

	cpdef bint strictcontains(self, Vector3 point):
		return self.low < point and point < self.high

	cpdef bint intersects(self, Box other):
		for n in range(8):
			if self.contains(other.vertex(n)): return True
		for n in range(8): #this shouldn't be necessary?
			if other.contains(self.vertex(n)): return True
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

	# scaling
	def __rmul__(self, double c): #scale from center
		cdef Vector3 low = self.low - self.center
		cdef Vector3 high = self.high - self.center
		return Box(c*low+self.center, c*high+self.center)


	# representation
	def __str__(self):
		string = "Box:\n"
		string+="\tlow=  "+self.low.__str__()+"\n"
		string+="\thigh= "+self.high.__str__()
		return string

	def __repr__(self):
		return "Box("+repr(self.low)+", "+repr(self.high)+")"



