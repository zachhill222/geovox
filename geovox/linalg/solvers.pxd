#cython: language_level=3, boundscheck=False, wraparound=False

from geovox.linalg.vector cimport Vector
from geovox.linalg.matrix cimport Matrix

cpdef Vector gaussSeidel(Matrix A, Vector b, Vector x, unsigned int n_iter)