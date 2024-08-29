cdef class NelderMead:
	def __init__(self, object F, Box bounds, double target):
		#problem definition
		self.F = F
		self.bounds = bounds
		self.R2 = abs(bounds.center-bounds.vertex(0))
		self.penalty_weight = 1.0

		#standard parameters
		self.alpha = 1.0
		self.gamma = 2.0
		self.rho   = 0.5
		self.sigma = 0.5

		#stopping criteria
		self.maxiter = 50
		self.target  = target

		#initialize simplex
		self.simplex = [Vector(3)]*4

		cdef int minvertex, n
		cdef double tempfval
		
		minvertex = 0
		for n from 1 <= n < 8:
			if self.evalF(bounds.vertex(n)) < self.evalF(bounds.vertex(minvertex)):
				minvertex = n
		if   minvertex == 0: self.simplex = [bounds.vertex(0), bounds.vertex(1), bounds.vertex(2), bounds.vertex(4)]
		elif minvertex == 1: self.simplex = [bounds.vertex(1), bounds.vertex(0), bounds.vertex(3), bounds.vertex(5)]
		elif minvertex == 2: self.simplex = [bounds.vertex(2), bounds.vertex(0), bounds.vertex(3), bounds.vertex(6)]
		elif minvertex == 3: self.simplex = [bounds.vertex(3), bounds.vertex(1), bounds.vertex(2), bounds.vertex(7)]
		elif minvertex == 4: self.simplex = [bounds.vertex(4), bounds.vertex(0), bounds.vertex(5), bounds.vertex(6)]
		elif minvertex == 5: self.simplex = [bounds.vertex(5), bounds.vertex(1), bounds.vertex(4), bounds.vertex(7)]
		elif minvertex == 6: self.simplex = [bounds.vertex(6), bounds.vertex(2), bounds.vertex(4), bounds.vertex(7)]
		elif minvertex == 7: self.simplex = [bounds.vertex(7), bounds.vertex(3), bounds.vertex(5), bounds.vertex(6)]


		#initialize function values
		self.fvals = [0.0]*4
		for n in range(4):
			self.fvals[n] = self.evalF(self.simplex[n])

	cpdef double minimize(self):
		cdef int n_steps = 0
		while self.fvals[0] > self.target and n_steps < self.maxiter:
			n_steps += 1
			self.takestep()

		# print(n_steps, self.fvals[0])
		return self.fvals[0]


	cpdef double evalF(self, Vector point): #add penalty function to F
		cdef double val = self.F(point)

		cdef double penalty = abs(self.bounds.center - point) - self.R2

		if penalty > 0:
			val += self.penalty_weight*penalty
		return val


	cpdef void ordersimplex(self):
		cdef bint unordered = True
		cdef int n_sweeps = 0
		cdef Vector tempvert
		cdef double tempfval


		cdef int n 
		while unordered and n_sweeps < 12:
			n_sweeps += 1
			unordered = False
			for n from 0 <= n < 3:
				if self.fvals[n] > self.fvals[n+1]:
					unordered = True
					
					#swap fvals
					tempfval = self.fvals[n]
					self.fvals[n] = self.fvals[n+1]
					self.fvals[n+1] = tempfval

					#swap vertices
					tempvert = self.simplex[n]
					self.simplex[n] = self.simplex[n+1]
					self.simplex[n+1] = tempvert

		if unordered: print("SIMPLEX METHOD ORDER FAILED")

	cpdef void takestep(self):
		cdef Vector centroid, reflected, expanded, contracted
		cdef double f_reflected, f_expanded, f_contracted
		cdef bint shrink_flag = False

		# compute centroid (always feasible)
		centroid = 0.25*(self.simplex[0]+self.simplex[1]+self.simplex[2]+self.simplex[3])
		
		# compute refleced and ensure feasible
		reflected = centroid + self.alpha*(centroid-self.simplex[3])
		# reflected = self.ensurebounds(reflected, centroid)
		f_reflected = self.evalF(reflected)


		# main logic
		if  self.fvals[0] <= f_reflected and f_reflected < self.fvals[2]:
			self.fvals[3] = f_reflected
			self.simplex[3] = reflected
			# print("accept reflected")

		elif f_reflected < self.fvals[0]:
			expanded = centroid + self.gamma*(reflected - centroid)
			# expanded = self.ensurebounds(expanded, centroid)
			f_expanded = self.evalF(expanded)

			if f_expanded < f_reflected:
				self.fvals[3] = f_expanded
				self.simplex[3] = expanded
				# print("accept expanded")
			else:
				self.fvals[3] = f_reflected
				self.simplex[3] = reflected
				# print("accept reflected")
		else:
			if f_reflected < self.fvals[3]:
				contracted = centroid + self.rho*(reflected - centroid)
				# contracted = self.ensurebounds(contracted, centroid)
				f_contracted = self.evalF(contracted)

				if f_contracted < f_reflected:
					self.fvals[3] = f_contracted
					self.simplex[3] = contracted
					# print("accept contracted (outside)")

				else:
					shrink_flag = True
			else:
				contracted = centroid + self.rho*(self.simplex[3]-centroid) #always feasible
				f_contracted = self.evalF(contracted)
				if f_contracted < self.fvals[3]:
					self.fvals[3] = f_contracted
					self.simplex[3] = contracted
					# print("accept contracted (inside)")
				else:
					shrink_flag = True

		# shrink step
		cdef int n
		if shrink_flag:
			# print("shrinking")
			for n from 1<= n < 4:
				self.simplex[n] = self.simplex[0] + self.sigma*(self.simplex[n] - self.simplex[0]) #always feasible if 0<sigma<1
				self.fvals[n] = self.evalF(self.simplex[n])

		# order function values
		self.ordersimplex()

		# print("\n================")
		# for n in range(4):
		# 	print(f"fvals[{n}]= {self.fvals[n]}, simplex[{n}]= {self.simplex[n]}")

