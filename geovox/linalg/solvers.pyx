cpdef Vector gaussSeidel(Matrix A, Vector b, Vector x, unsigned int n_iter):
	if not (A.issquare() and A.shape[0] == len(b) and len(x)==len(b)): return Exception("Dimension mismatch")

	cdef double diag1, summation

	cdef unsigned int i, j, k
	for k in range(n_iter):
		for i in range(x._len):
			diag1 = 1.0/A[i,i]
			summation = b[i]
			for j in range(0,i):
				summation -= A[i,j]*x[j]
			for j in range(i+1,x._len):
				summation -= A[i,j]*x[j]
			x[i] = diag1*summation

	return x





