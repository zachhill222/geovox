############# import classes from utility ############
from geovox.utility cimport Vector3, Box, Quaternion


############# BASE CLASS ################
cdef class Shape3D:
	cdef Box bbox #bounding box
	cdef Vector3 center #center


############# PARTICLE CLASSES #########
cdef class Sphere(Shape3D):
	cdef double R, R2
	cpdef bint contains(self, Vector3 point)
	cdef inline Box getbbox(self):
		return Box( Vector3(self.center.x-self.R, self.center.y-self.R, self.center.z-self.R), Vector3(self.center.x+self.R, self.center.y+self.R, self.center.z+self.R))


cdef class Prism(Shape3D):
	cdef Vector3 R #radii (distance from center to faces)
	cdef Quaternion Q #quaternion for rotation FROM global coordinates TO local coordinates
	cdef Box localprism #local coordinates, Box(-R,R)
	cdef Box getbbox(self) #get bbox after Q or center have been updated
	cpdef bint contains(self, Vector3 point)
	cpdef Vector3 vertex(self, int n) #get n-th vertex (.vtk ordering in local coordinates and then translated to global)
	cpdef Vector3 facecenter(self, int n) #get n-th facecenter (xlow, xhigh, ... in local coordinates and then translated to global)



cdef class Ellipsoid(Prism):
	cpdef bint contains(self, Vector3 point)


cdef class SuperEllipsoid(Prism):
	cdef double[2] eps #shape parameters
	cdef double e0, e1, e2 #powers
	cpdef bint contains(self, Vector3 point)