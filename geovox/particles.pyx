########## import special C functions and tools from utility.pyx ##########
from geovox.utilities cimport Vector3, Box, Quaternion
from libc.math cimport tgamma, fabs
from libc.math cimport pow as fpow

# import numpy as np
# cimport numpy as np


# Each shape must contain a volume property and a function of the signature: bint contains(self, Vector3 point)
# that determines if the point is contained in the particle. Each particle should be (mathematically) closed and convex.
# Each shape must have a method to update both the center and rotation quaternion (if it has one)

############# USEFUL STUFF ##############
cpdef beta(double x, double y):
	return tgamma(x)*tgamma(y)/tgamma(x+y)

	
C_SPHERE = 4.188790204786390984616857844373

############# BASE CLASS ################
cdef class Shape3D:
	def __init__(self, Vector3 center, Box bbox):
		self.center = center
		self.bbox = bbox

	def __repr__(self):
		return "Shape3D("+repr(self.center)+", "+ repr(self.bbox)+")"


############ Sphere #################
cdef class Sphere(Shape3D):
	def __init__(self, double R, Vector3 center):
		cdef Vector3 high = Vector3(center.x+R, center.y+R, center.z+R)
		cdef Vector3 low = Vector3(center.x-R, center.y-R, center.z-R)
		super().__init__(center, Box(low, high))
		self.R      = R
		self.R2     = R*R
		self.R1     = 1.0/R

	cpdef double levelval(self, Vector3 point): #evaluate level set squared
		return (self.R1*(point-self.center)).abs2

	cpdef Vector3 levelgrad(self, Vector3 point): #gradient of the level function
		return 2.0*(point - self.center)


	cpdef bint contains(self, Vector3 point): return self.levelval(point) <= 1.0

	@property
	def volume(self): return C_SPHERE*(self.R2*self.R)
	
	@property
	def bbox(self): return self.bbox

	@property
	def R(self): return self.R
	@R.setter
	def R(self, double value):
		self.R = value
		self.R2 = value*value
		self.bbox = self.getbbox()


	@property
	def center(self): return self.center
	@center.setter
	def center(self, Vector3 value):
		self.center = value
		self.bbox = self.getbbox()

	def __repr__(self):
		return f"Sphere(double {str(self.R)}, {repr(self.center)})"

	def __str__(self):
		string = "Sphere:\n"
		string+= f"\tR= {self.R}\n"
		string+= f"\tcenter= {repr(self.center)}\n"
		return string


############ Prism #################
cdef class Prism(Shape3D):
	def __init__(self, Vector3 R, Vector3 center, Quaternion Q):
		self.R = R
		self.localprism = Box(-self.R, self.R)
		self.Q = Q #if rotations are opposite to what are expected, your quaternion should probably be passed to __init__ as Q.conj()
		self.center = center
		self.bbox = self.getbbox()

	cdef Box getbbox(self):
		# determine bounding box
		cdef double maxR  = 2*abs(self.R)
		cdef Vector3 low  = Vector3(maxR, maxR, maxR)+self.center #placeholder to guarentee components from a vertex are used
		cdef Vector3 high = Vector3(-maxR, -maxR, -maxR)+self.center #placeholder to guarentee components from a vertex are used
		cdef Vector3 vert
		for n in range(8):
			vert = self.vertex(n)
			low.x = min(low.x, vert.x)
			low.y = min(low.y, vert.y)
			low.z = min(low.z, vert.z)
			high.x = max(high.x, vert.x)
			high.y = max(high.y, vert.y)
			high.z = max(high.z, vert.z)
		box = Box(low, high)
		return box

	cpdef double levelval(self, Vector3 point): #evaluate level set squared
		point-= self.center
		point = self.Q.rotate(point)
		point/= self.R
		return point.infNorm()

	# cpdef Vector3 levelgrad(self, Vector3 point): #gradient of the level function, not implemented

	cpdef bint contains(self, Vector3 point): return self.levelval(point) <= 1.0

	cpdef Vector3 vertex(self, int n):#get n-th vertex (.vtk ordering in local coordinates and then translated to global)
		return self.Q.conj().rotate(self.localprism.vertex(n))+self.center

	cpdef Vector3 facecenter(self, int n):
		cdef Vector3 point
		if n==0:   point = Vector3( -self.R.x, 0.0, 0.0)
		elif n==1: point = Vector3(  self.R.x, 0.0, 0.0)
		elif n==2: point = Vector3( 0.0, -self.R.y, 0.0)
		elif n==3: point = Vector3( 0.0,  self.R.y, 0.0)
		elif n==4: point = Vector3( 0.0, 0.0, -self.R.z)
		elif n==5: point = Vector3( 0.0, 0.0,  self.R.z)

		return self.Q.conj().rotate(point)+self.center

	@property
	def volume(self): return 8.0*self.R.x*self.R.y*self.R.z
	
	@property
	def bbox(self): return self.bbox
	
	@property
	def center(self): return self.center
	@center.setter
	def center(self, Vector3 value):
		self.center = value
		self.bbox = self.getbbox()

	@property
	def R(self): return self.R
	@R.setter
	def R(self, Vector3 value):
		self.R = value
		self.localprism = Box(-self.R, self.R)
		self.bbox = self.getbbox()

	@property
	def Q(self): return self.Q
	@Q.setter
	def Q(self, Quaternion value):
		self.Q = value
		self.bbox = self.getbbox()

	def __repr__(self):
		return f"Prism({repr(self.R)}, {repr(self.center)}, {repr(self.Q)})"

	def __str__(self):
		string = "Prism:\n"
		string+= f"\tR= {repr(self.R)}\n"
		string+= f"\tcenter= {repr(self.center)}\n"
		string+= f"\tQ= {repr(self.Q)}\n"
		return string


########### Ellipsoid ###########
cdef class Ellipsoid(Prism):
	def __init__(self, Vector3 R, Vector3 center, Quaternion Q):
		super().__init__(R, center, Q)

	@property
	def volume(self):
		return C_SPHERE*self.R.x*self.R.y*self.R.z #volume of the bounding prism times pi/6

	cpdef double levelval(self, Vector3 point): #evaluate level set squared
		point-= self.center
		point = self.Q.rotate(point)
		point/= self.R
		return point.abs2

	cpdef Vector3 levelgrad(self, Vector3 point): #gradient of the level function
		point-= self.center
		point = self.Q.rotate(point) #rotate into reference frame
		point/= self.R #scale
		point = 2.0*point #gradient is the sum of squares
		return self.Q.conj().rotate(point) #rotate back into global reference frame

	cpdef bint contains(self, Vector3 point): return self.levelval(point) <= 1.0

	def __repr__(self):
		return f"Ellipsoid({repr(self.R)}, {repr(self.center)}, {repr(self.Q)})"

	def __str__(self):
		string = "Ellipsoid:\n"
		string+= f"\tR= {repr(self.R)}\n"
		string+= f"\tcenter= {repr(self.center)}\n"
		string+= f"\tQ= {repr(self.Q)}\n"
		return string


############ Super Ellipsoid ##########
cdef class SuperEllipsoid(Prism):
	def __init__(self, Vector3 R, double eps1, double eps2, Vector3 center, Quaternion Q):
		super().__init__(R, center, Q)
		self.eps[0] = eps1
		self.eps[1] = eps2
		self.e0 = 2.0/eps1
		self.e1 = 2.0/eps2
		self.e2 = eps2/eps1

		self.C_SUPERELLIPSOID = self.eps[0]*self.eps[1] * beta(0.5*self.eps[0], 1.0+self.eps[0]) * beta(0.5*self.eps[1], 1.0+0.5*self.eps[1])

	@property
	def volume(self):
		return 2.0 * self.R.x*self.R.y*self.R.z * self.C_SUPERELLIPSOID

	cpdef double levelval(self, Vector3 point): #evaluate level set (squared)
		point-= self.center
		point = self.Q.rotate(point)
		point/= self.R
		point.x = fpow(fabs(point.x), self.e1)
		point.y = fpow(fabs(point.y), self.e1)
		point.z = fpow(fabs(point.z), self.e0)
		return fpow((point.x+point.y), self.e2) + point.z

	cpdef Vector3 levelgrad(self, Vector3 point): #gradient of the level function
		point-= self.center
		point = self.Q.rotate(point) #rotate into reference frame
		point/= self.R #scale
		cdef Vector3 grad = Vector3(0.0,0.0,0.0)
		grad.x = self.e1 * fpow( fpow(point.x, self.e1)+fpow(point.y, self.e1), self.e2 ) * fpow(point.x, self.e1-1.0) / self.R.x
		grad.y = self.e1 * fpow( fpow(point.x, self.e1)+fpow(point.y, self.e1), self.e2 ) * fpow(point.y, self.e1-1.0) / self.R.y
		grad.z = self.e0 * fpow(point.z, self.e0-1.0) / self.R.z

		return self.Q.conj().rotate(grad) #rotate back into global reference frame

	cpdef bint contains(self, Vector3 point): return self.levelval(point) <= 1.0
	# cpdef bint contains(self, Vector3 point): return True

	def __repr__(self):
		return f"SuperEllipsoid({repr(self.R)}, double {self.eps[0]}, double {self.eps[1]}, {repr(self.center)}, {repr(self.Q)})"

	def __str__(self):
		string = "SuperEllipsoid:\n"
		string+= f"\tR= {repr(self.R)}\n"
		string+= f"\tcenter= {repr(self.center)}\n"
		string+= f"\tQ= {repr(self.Q)}\n"
		string+= f"\teps1= {self.eps[0]}\n"
		string+= f"\teps2= {self.eps[1]}\n"
		return string












		





	
