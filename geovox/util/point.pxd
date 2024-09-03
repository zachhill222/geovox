#cython: language_level=3, boundscheck=False, wraparound=False


#import Point from C++
cdef extern from "point.hpp" namespace "GeoVox::util":
	cdef cppclass Point:
		Point() except +

		# constructors
		Point(int) except +

		# attributes
		int _len
		double* _data


		#############################################################################
		############################ CONVENIENCE METHODS ############################
		#############################################################################
		void fill(double& scalar) #set all elements to scalar
		double max() #get maximum element
		int    maxind() #get lowest index of maximum element
		double min() #get minimum element
		int    minind() #get lowest index of minimum element
		double sum() #sum all elements
		Point normalize() #normalize NOT in-place to a Euclidean norm of 1


		#############################################################################
		############################# DISTANCE METHODS ##############################
		#############################################################################
		double norm2() #Euclidean norm squared
		double norm() #Euclidean norm
		double infnorm() #max-norm
		double dist2(const Point& other) #Euclidean distance squared
		double dist(const Point& other) #Euclidean distance
		double dist_inf(const Point& other) #max-norm distance


		#############################################################################
		############################# LINEAR ALGEBRA METHODS ########################
		#############################################################################
		double dot(const Point& other) #dot product
		Point cross(const Point& other) #cross product, only if _len==3, but no error check

#create a cdef class as a wrapper for Point
cdef class PyPoint:
	cdef Point* thisptr
	cpdef PyPoint copy(self)


################## useful functions #########################################
cpdef PyPoint el_min(PyPoint v, PyPoint u)
cpdef PyPoint el_max(PyPoint v, PyPoint u)