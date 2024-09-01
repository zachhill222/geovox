from geovox.linalg.vector cimport Vector
from geovox.linalg.matrix cimport Matrix
from geovox.linalg.solvers cimport gaussSeidel
# from geovox.geometry.particles cimport LEVELVALCUTOFF
from geovox.util.box cimport Box


# cdef double _eval_fun(object F, Vector point, Box box)
cdef double _closest_point_neldermead(object F, Box box, int verbose) #roughly try to find the closest point in particle P to x using nelder mead


# cdef Vector _closest_point_newton(object P, Box box)

cdef Vector _line_search_ternary(object F, Vector start, Vector direction, double left, double right, double tol)
cdef Vector _closest_point_gradient(object P, Box box, int verbose)
cdef Vector _best_point(object P, Box box, int verbose)