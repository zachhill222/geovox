# cdef double _eval_fun(object F, Vector point, Box box):
# 	cdef double val, penalty
# 	val = F(point)

# 	if not box.contains(point):
# 		penalty = 1000.0*((box.center-point)/box.sidelength).infNorm()
# 		# print(f"applying penalty of {penalty} to levelval of {val}")
# 		val += penalty

# 	return val


cdef double _closest_point_neldermead(object F, Box box, int verbose):
	# set iteration parameters
	cdef double alpha, gamma, rho, sigma
	alpha = 1.0
	gamma = 2.0
	rho   = 0.5
	sigma = 0.5

	# set stopping criteria
	cdef unsigned int maxiter = 25

	# determine best starting simplex
	cdef int minvertex, n
	cdef double tempfval, minfval
	
	minvertex = 0
	minfval   = F(box.vertex(0))
	for n in range(1,8):
		tempfval = F(box.vertex(n))
		if tempfval < minfval:
			minvertex = n
			minfval   = tempfval

	if verbose > 0: print(f"NelderMead START: best vertex {minvertex}= {box.vertex(minvertex)} with value {minfval}")

	#initialize simplex
	cdef list simplex
	simplex = [Vector(3)]*4
	cdef Box initbox = box
	if   minvertex == 0: simplex = [initbox.vertex(0), initbox.vertex(1), initbox.vertex(2), initbox.vertex(4)]
	elif minvertex == 1: simplex = [initbox.vertex(1), initbox.vertex(0), initbox.vertex(3), initbox.vertex(5)]
	elif minvertex == 2: simplex = [initbox.vertex(2), initbox.vertex(0), initbox.vertex(3), initbox.vertex(6)]
	elif minvertex == 3: simplex = [initbox.vertex(3), initbox.vertex(1), initbox.vertex(2), initbox.vertex(7)]
	elif minvertex == 4: simplex = [initbox.vertex(4), initbox.vertex(0), initbox.vertex(5), initbox.vertex(6)]
	elif minvertex == 5: simplex = [initbox.vertex(5), initbox.vertex(1), initbox.vertex(4), initbox.vertex(7)]
	elif minvertex == 6: simplex = [initbox.vertex(6), initbox.vertex(2), initbox.vertex(4), initbox.vertex(7)]
	elif minvertex == 7: simplex = [initbox.vertex(7), initbox.vertex(3), initbox.vertex(5), initbox.vertex(6)]


	#setup penalty function
	cdef Box penaltybox = 1.1*box

	#initialize function values
	cdef Vector fvals
	fvals = Vector(4)
	for n in range(4):
		# fvals[n] = _eval_fun(F, simplex[n], penaltybox)
		fvals[n] = F(simplex[n])

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
					tempvert = simplex[n].copy()
					simplex[n] = simplex[n+1].copy()
					simplex[n+1] = tempvert.copy()

		if verbose > 1:
			print(f"({iteration}) levelval= {F(simplex[0])}")
			print(f"\tfvals= {fvals}")

		# convergence check
		if fvals[0] <= 1.0:
			break

		# if not (box.contains(simplex[0]) or box.contains(simplex[1]) or box.contains(simplex[2]) or box.contains(simplex[3])): #entire simplex is outside of the box
		# 	break


		# main logic
		shrink_flag = False

		# compute centroid (always feasible)
		centroid = 0.33333333333333*(simplex[0]+simplex[1]+simplex[2])
		
		# REFLECTED CHECK
		reflected = centroid + alpha*(centroid-simplex[3])
		f_reflected = F(reflected)

		if  fvals[0] <= f_reflected and f_reflected < fvals[2]:
			fvals[3] = f_reflected
			simplex[3] = reflected
			if verbose > 2:  print(f"\t({iteration}) accept reflected")
			continue

		# EXPANSION CHECK
		if f_reflected < fvals[0]:
			expanded = centroid + gamma*(reflected - centroid)
			f_expanded = F(expanded)

			if f_expanded < f_reflected:
				fvals[3] = f_expanded
				simplex[3] = expanded
				if verbose > 2:  print(f"\t({iteration}) accept expanded")
				continue
			else:
				fvals[3] = f_reflected
				simplex[3] = reflected
				if verbose > 2:  print(f"\t({iteration}) accept reflected (checked expanded)")
				continue


		# CONTRACTION CHECK
		if f_reflected < fvals[3]:
			contracted = centroid + rho*(reflected - centroid)
			f_contracted = F(contracted)

			if f_contracted < f_reflected:
				fvals[3] = f_contracted
				simplex[3] = contracted
				if verbose > 2:  print(f"\t({iteration}) accept contracted (outside)")
				continue
			else:
				shrink_flag = True
		else:
			contracted = centroid + rho*(simplex[3]-centroid)
			f_contracted = F(contracted)
			if f_contracted < fvals[3]:
				fvals[3] = f_contracted
				simplex[3] = contracted
				if verbose > 2: print(f"\t({iteration}) accept contracted (inside)")
				continue
			else:
				shrink_flag = True

		# shrink step
		if shrink_flag:
			if verbose > 2: print(f"\t\tshrinking")
			for n in range(1,4):
				simplex[n] = simplex[0] + sigma*(simplex[n] - simplex[0])
				fvals[n] = F(simplex[n])

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

	if verbose>0: print(f"NelderMead STOP: {simplex[0]} (inbox={box.contains(simplex[0])}, distance to box center= {abs(simplex[0]-box.center)}) with value {F(simplex[0])} in {iteration} iterations")

	return F(simplex[0])











cdef Vector _line_search_ternary(object F, Vector start, Vector direction, double left, double right, double tol):
	cdef double fa, fb, b, c
	cdef double fleft  = F(start + left*direction)
	cdef double fright = F(start + right*direction)

	if fleft <= 1.0: return start + left*direction
	elif fright <= 1.0: return start + right*direction

	while right-left > tol:
		a = 0.6666*left + 0.3333*right
		b = 0.3333*left + 0.6666*right

		fa = F(start + a*direction)
		fb = F(start + b*direction)

		if fa < fb:
			right = b
		else:
			left  = a

	if   min(fleft, fa, fb, fright) == fleft: return start + left*direction
	elif min(fleft, fa, fb, fright) == fa: return start + a*direction
	elif min(fleft, fa, fb, fright) == fb: return start + b*direction
	elif min(fleft, fa, fb, fright) == fright: return start + right*direction


cdef Vector _best_point(object P, Box box, int verbose): #coordinate direction line search
	if verbose > 0: print(f"COORDINATE SEARCH START: box= {box}, center= {box.center}, contained= {box.contains(box.center)}")
	cdef int maxiter = 25
	cdef double stopfval = 1.0

	# starting guess
	cdef Vector X = box.center
	cdef double fval = P.levelval(X)

	cdef Vector X_guess, searchDir
	searchDir = Vector(len(X))
	X_guess = Vector(len(X))

	cdef double left, right, fguess
	cdef unsigned int iteration, dim
	for iteration in range(maxiter):
		for dim in range(len(X)):
			searchDir.fill(0.0)
			searchDir[dim] = 1.0

			left  = box.low[dim] - X[dim]
			right = box.high[dim] - X[dim]

			# print(searchDir, left, right, X)
			# print(f"\t\tX+LEFT*DIR= {X+left*searchDir}, containded= {box.contains(X+left*searchDir)}")
			# print(f"\t\tX+RIGHT*DIR= {X+right*searchDir}, containded= {box.contains(X+right*searchDir)}")

			X_guess = _line_search_ternary(P.levelval, X, searchDir, left, right, 0.05*box.diam)
			fguess = P.levelval(X_guess)

			if fguess <= fval:
				X = X_guess
				fval = fguess

		if X_guess == X or fguess <= stopfval: return X_guess

		if verbose>1: print(f"\t({iteration}) X= {X} fval= {fval}, contained in box= {box.contains(X)}")

	if verbose > 0: print(f"COORDINATE SEARCH STOP: X= {X} fval= {fval}, contained in box= {box.contains(X)}")
	return X



cdef Vector _closest_point_gradient(object P, Box box, int verbose):
	# set stopping criteria
	cdef int maxiter = 10
	cdef double stopfval = 1.0

	# starting guess
	cdef Vector X = box.center

	cdef double C = 1.0/((0.5*box.diam)**2)
	cdef object FP = lambda point: P.levelval(point) + C*(point-box.center).abs2()

	cdef Vector gradF, gradP, searchDir

	cdef int iteration, j
	for iteration in range(maxiter):
		if verbose > 0: print(f"{(iteration)}: X= {X} (dist= {abs(X-box.center)}, levelval= {P.levelval(X)}, FPval= {FP(X)}")
		if P.levelval(X) <= stopfval: break

		#set up iteration
		gradF = P.levelgrad(X)
		gradP = (2.0*C*(abs(X-box.center))) * X

		searchDir = -(gradF+gradP).normalize()

		X = _line_search_ternary(FP, X, searchDir, 0.0, box.diam, 0.01*box.diam)

	return X





