############# imports ############
from geovox.linalg.vector cimport Vector
from geovox.util.box cimport Box
from geovox.util.quaternion cimport Quaternion


#generic particle type
ctypedef fused particle_t:
	Sphere
	Prism
	Ellipsoid
	SuperEllipsoid

# cdef double LEVELVALCUTOFF = 1.0


############# BASE CLASS ################
cdef class Shape3D:
	cdef Box bbox #bounding box
	cdef Vector center #center


############# PARTICLE CLASSES #########
cdef double SphereLevelFun(double R, Vector center, Vector point)

cdef class Sphere(Shape3D):
	cdef double R, R2, R1
	cpdef Sphere copy(self)
	cpdef bint contains(self, Vector point)
	cdef inline Box getbbox(self):
		return Box( Vector(self.center.x-self.R, self.center.y-self.R, self.center.z-self.R), Vector(self.center.x+self.R, self.center.y+self.R, self.center.z+self.R))



cdef double PrismLevelVal(Vector R, Vector center, Quaternion Q, Vector point)

cdef class Prism(Shape3D):
	cdef Vector R #radii (distance from center to faces)
	cdef Quaternion Q #quaternion for rotation FROM global coordinates TO local coordinates
	cdef Box localprism #local coordinates, Box(-R,R)
	cpdef Prism copy(self)
	cdef Box getbbox(self) #get bbox after Q or center have been updated
	cpdef bint contains(self, Vector point)
	cpdef Vector vertex(self, int n) #get n-th vertex (.vtk ordering in local coordinates and then translated to global)
	cpdef Vector facecenter(self, int n) #get n-th facecenter (xlow, xhigh, ... in local coordinates and then translated to global)




cdef double EllipsoidLevelVal(Vector R, Vector center, Quaternion Q, Vector point)

cdef class Ellipsoid(Prism):
	cpdef Ellipsoid copy(self)
	cpdef bint contains(self, Vector point)




cdef SuperEllipsoidLevelVal(Vector R, double e0, double e1, double e2, Vector center, Quaternion Q, Vector point)

cdef class SuperEllipsoid(Prism):
	cdef double C_SUPERELLIPSOID
	cdef double[2] eps #shape parameters
	cdef double e0, e1, e2 #powers
	cpdef SuperEllipsoid copy(self)
	cpdef bint contains(self, Vector point)


