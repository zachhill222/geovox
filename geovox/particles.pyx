########## import special C functions and tools from utility.pyx ##########
from geovox.utility cimport Vector3, Box, Quaternion, PI
from scipy.special import beta
from libc.math cimport pow as fpow

# Each shape must contain a volume and a function of the signature: bint contains(self, Vector3 point)
# that determines if the point is contained in the particle. Each particle should be (mathematically) closed and convex.


############# BASE CLASS ################
cdef class Shape3D:
	def __init__(self, Vector3 center, Box bbox):
		self.center = center
		self.bbox = bbox

	def __repr__(self):
		return "Shape3D("+repr(self.center)+", "+ repr(self.bbox)+")"


############ Sphere #################
cdef class Sphere(Shape3D):
	def __init__(self, double r, Vector3 center):
		cdef Vector3 high = Vector3(center.x+r, center.y+r, center.z+r)
		cdef Vector3 low = Vector3(center.x-r, center.y-r, center.z-r)
		super().__init__(center, Box(low, high))
		self.r      = r
		self.r2     = r*r
		self.volume = (4.0*PI/3.0)*(self.r2*self.r)

	cdef bint contains(self, Vector3 point):
		point-=self.center
		return point.abs2 <= self.r2

	def __repr__(self):
		return "Sphere(double "+str(self.r)+", "+repr(self.center)+")"


############ Prism #################
cdef class Prism(Shape3D):
	def __init__(self, Vector3 R, Vector3 center, Quaternion Q):
		# determine bounding box
		cdef double maxR  = R.infNorm()
		cdef Vector3 low  = Vector3(maxR, maxR, maxR)
		cdef Vector3 high = Vector3(-maxR, -maxR, -maxR)
		cdef Box testbox  = Box(low, high)
		cdef Vector3 vertex
		for n in range(8):
			vertex = Q.rotate(testbox.vertex(n))
			low.x = min(low.x, vertex.x)
			low.y = min(low.x, vertex.y)
			low.z = min(low.x, vertex.z)
			high.x = max(high.x, vertex.x)
			high.y = max(high.x, vertex.y)
			high.z = max(high.x, vertex.z)

		# finish initialization
		super().__init__(center, Box(low, high))

		self.prism = Box(-R,R) #prism in local coordinates
		self.R = R
		self.Q = Q #if rotations are opposite to what are expected, your quaternion should probably be passed to __init__ as Q.conj()
		self.volume = 8.0*self.R.x*self.R.y*self.R.z

	cdef bint contains(self, Vector3 point):
		point-= self.center
		point = self.Q.rotate(point)
		return self.prism.contains(point)

	def __repr__(self):
		return "Prism("+repr(self.R)+", "+repr(self.center)+", "+repr(self.Q)+")"


########### Ellipsoid ###########
cdef class Ellipsoid(Prism):
	def __init__(self, Vector3 R, Vector3 center, Quaternion Q):
		super().__init__(R, center, Q)
		self.volume *= (PI/6.0) #volume of the bounding prism times pi/6

	cdef bint contains(self, Vector3 point):
		point-= self.center
		point = self.Q.rotate(point)
		point/= self.R
		return point.abs2() <= 1

	def __repr__(self):
		return "Ellipsoid("+repr(self.R)+", "+repr(self.center)+", "+repr(self.Q)+")"


############ Super Ellipsoid ##########
cdef class SuperEllipsoid(Prism):
	def __init__(self, Vector3 R, double eps1, double eps2, Vector3 center, Quaternion Q):
		super().__init__(R, center, Q)
		self.eps[0] = eps1
		self.eps[1] = eps2
		self.e0 = 1.0/eps1
		self.e1 = 1.0/eps2
		self.e2 = self.eps2/self.eps1
		self.volume *= (0.25*eps1*eps2* beta(0.5*eps1, 1.0+eps1) * beta(0.5*eps2, 1.0+0.5*eps2))

	cdef bint contains(self, Vector3 point):
		point-= self.center
		point = self.Q.rotate(point)
		point/=self.R

		point.x = fpow(point.x, self.e1)
		point.y = fpow(point.y, self.e1)
		point.z = fpow(point.z, self.e0)
		return fpow((point.x+point.y), self.e2) + point.z <= 1.0

	def __repr__(self):
		return "SuperEllipsoid("+repr(self.R)+", "+"double[2] {{0}, {1}}, ".format(self.eps[0], self.eps[1])+repr(self.center)+", "+repr(self.Q)+")"












		





	
