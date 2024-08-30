cdef double _eval_fun(object P, Vector point, Box box):
	cdef double val, penalty
	val = P.levelval(point)

	if not box.contains(point):
		val += 10*((box.center-point)/box.sidelength).infNorm()

	return val


cdef Vector _closest_point_neldermead(object P, Box box):
	# set iteration parameters
	cdef double alpha, gamma, rho, sigma
	alpha = 1.0
	gamma = 2.0
	rho   = 0.5
	sigma = 0.5

	# set stopping criteria
	cdef unsigned int maxiter
	maxiter = 25
	cdef double tol, R
	R   = 0.5*abs(box.sidelength)
	tol = 0.01*R #tolerance for the distance between best and worst point


	# determine best starting simplex
	cdef int minvertex, n
	cdef double tempfval, minfval
	
	minvertex = 0
	minfval   = _eval_fun(P, box.vertex(0), box)
	for n in range(1,8):
		tempfval = _eval_fun(P, box.vertex(n), box)
		if tempfval < minfval:
			minvertex = n
			minfval   = tempfval

	#initialize simplex
	cdef list simplex
	simplex = [Vector(3)]*4
	if   minvertex == 0: simplex = [box.vertex(0), box.vertex(1), box.vertex(2), box.vertex(4)]
	elif minvertex == 1: simplex = [box.vertex(1), box.vertex(0), box.vertex(3), box.vertex(5)]
	elif minvertex == 2: simplex = [box.vertex(2), box.vertex(0), box.vertex(3), box.vertex(6)]
	elif minvertex == 3: simplex = [box.vertex(3), box.vertex(1), box.vertex(2), box.vertex(7)]
	elif minvertex == 4: simplex = [box.vertex(4), box.vertex(0), box.vertex(5), box.vertex(6)]
	elif minvertex == 5: simplex = [box.vertex(5), box.vertex(1), box.vertex(4), box.vertex(7)]
	elif minvertex == 6: simplex = [box.vertex(6), box.vertex(2), box.vertex(4), box.vertex(7)]
	elif minvertex == 7: simplex = [box.vertex(7), box.vertex(3), box.vertex(5), box.vertex(6)]


	#initialize function values
	cdef Vector fvals
	fvals = Vector(4)
	for n in range(4):
		fvals[n] = _eval_fun(P, simplex[n], box)


	#main loop preparation
	cdef bint unordered
	cdef unsigned int n_sweeps
	cdef Vector tempvert

	cdef Vector centroid, reflected, expanded, contracted
	cdef double f_reflected, f_expanded, f_contracted
	cdef bint shrink_flag

	cdef unsigned int iteration
	for iteration in range(maxiter):
		
		#order simplex
		unordered = True
		n_sweeps = 0 
		while unordered and n_sweeps < 12:
			n_sweeps += 1
			unordered = False
			for n in range(3):
				if fvals[n] > fvals[n+1]:
					unordered = True
					
					#swap fvals
					tempfval = fvals[n]
					fvals[n] = fvals[n+1]
					fvals[n+1] = tempfval

					#swap vertices
					tempvert = simplex[n]
					simplex[n] = simplex[n+1]
					simplex[n+1] = tempvert

		# convergence check
		if abs(simplex[0]-simplex[3]) < tol or fvals[0] <= 1.0:
			# print("NelderMead converged")
			break


		#get new vertex for simplex
		shrink_flag = False

		# compute centroid (always feasible)
		centroid = 0.25*(simplex[0]+simplex[1]+simplex[2]+simplex[3])
		
		# compute refleced and ensure feasible
		reflected = centroid + alpha*(centroid-simplex[3])
		# reflected = ensurebounds(reflected, centroid)
		f_reflected = _eval_fun(P, reflected, box)

		# main logic
		if  fvals[0] <= f_reflected and f_reflected < fvals[2]:
			fvals[3] = f_reflected
			simplex[3] = reflected
			# print("accept reflected")

		elif f_reflected < fvals[0]:
			expanded = centroid + gamma*(reflected - centroid)
			# expanded = ensurebounds(expanded, centroid)
			f_expanded = _eval_fun(P, expanded, box)

			if f_expanded < f_reflected:
				fvals[3] = f_expanded
				simplex[3] = expanded
				# print("accept expanded")
			else:
				fvals[3] = f_reflected
				simplex[3] = reflected
				# print("accept reflected")
		else:
			if f_reflected < fvals[3]:
				contracted = centroid + rho*(reflected - centroid)
				# contracted = ensurebounds(contracted, centroid)
				f_contracted = _eval_fun(P, contracted, box)

				if f_contracted < f_reflected:
					fvals[3] = f_contracted
					simplex[3] = contracted
					# print("accept contracted (outside)")

				else:
					shrink_flag = True
			else:
				contracted = centroid + rho*(simplex[3]-centroid) #always feasible
				f_contracted = _eval_fun(P, contracted, box)
				if f_contracted < fvals[3]:
					fvals[3] = f_contracted
					simplex[3] = contracted
					# print("accept contracted (inside)")
				else:
					shrink_flag = True

		# shrink step
		if shrink_flag:
			# print("shrinking")
			for n in range(1,4):
				simplex[n] = simplex[0] + sigma*(simplex[n] - simplex[0]) #always feasible if 0<sigma<1
				fvals[n] = _eval_fun(P, simplex[n], box)


	return simplex[0]


