cimport cython
# from libc.stdlib cimport malloc, free

from libc.stdlib cimport atof, atoi #string (bits) to double
from libc.math cimport log2



@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _divide(Node N):
	cdef Box subbox
	cdef int p_child_nvert
	cdef int n_particles = len(N.particle_ind_list)

	cdef double minlevelval, templevelval
	cdef int n, p_ind, child_ind
	cdef Vector closestpoint

	if N.isdivided:
		for child_ind in range(8):	
			_divide(N.children[child_ind])
	else:
		# first division of root node logic
		if N.root is N: #add bounding box vertices, this should only be called on the very first divide() call on the root node
			N.points.update([N.bbox.vertex(n) for n in range(8)])


		# test if division is necessary
		if len(N.particle_ind_list) == 0: return #no need to divide here
		if N.nvert == 8: return #particles are convex, no need to divide here, all child regions would have nvert=8 as well
		if N.depth+1 > N.root.maxdepth: return #too deep

		# create children
		for child_ind in range(8):
			subbox = Box(N.bbox.vertex(child_ind), N.bbox.center)
			N.children.append(Node(subbox, N.depth+1))

			N.children[child_ind].nvert  = 0
			N.children[child_ind].root   = N.root
			N.children[child_ind].parent = N


		# P_levelevals = tuple([N.root.particle_list[N.particle_ind_list[p_ind]].levelval for p_ind in range(n_particles)])

		# update populate child attributes particle list and nvert, decide which particles should be passed down
		for child_ind in range(8):
			child = N.children[child_ind]

			# update point/vertex list in root
			for n in range(8):
				N.root.points.add(child.bbox.vertex(n))

			# loop through particles to see which should be passed down to the child
			for p_ind in range(n_particles):
				P = N.root.particle_list[N.particle_ind_list[p_ind]]

				if abs(child.bbox.center - N.root.particle_list[N.particle_ind_list[p_ind]].center) <= 0.5*(child.bbox.diam + N.root.particle_list[N.particle_ind_list[p_ind]].diam):
					#compute vertex intersections
					
					for n in range(8):
						templevelval = P.levelval(child.bbox.vertex(n))
						if templevelval <= 1.0:
							p_child_nvert += 1
					child.nvert = max(child.nvert, p_child_nvert)

					if child.nvert > 0:
						child.particle_ind_list.append(N.root.particle_list.index(P))
					elif child.bbox.diam >= P.diam:
						child.particle_ind_list.append(N.root.particle_list.index(P))
					else: #make a closer inspection in case the particle intersects a face but not a vertex
						if _closest_point_neldermead(P.levelval, child.bbox, verbose=0) <= 1.0:
							child.particle_ind_list.append(N.root.particle_list.index(P))
			


		
		#change parameters from N for a non-leaf node
		N.particle_ind_list = []
		N.nvert = -1 #only leaf nodes should have a valid marker
		N.isdivided = True
		N.root.totaldepth = max(N.root.totaldepth, N.depth+1)
























	


cdef class Node: #node of octree
	def __init__(self, Box bbox = Box(), int depth = 0):
		self.isdivided = False
		self.children = []
		self.depth = depth
		self.totaldepth = 0
		self.maxdepth = 99
		self.bbox = bbox
		self.particle_list = []
		self.particle_ind_list = []
		self.nvert = -1

		self.root = self #for storing global tree information
		self.points = set()

	@property
	def bbox(self): return self.bbox

	cpdef void cdivide(self): _divide(self)

	cpdef void divide(self):
		cdef Box subbox
		cdef int p_child_nvert

		cdef double minlevelval, templevelval
		cdef int n, p_ind, child_ind
		cdef Vector closestpoint

		if self.isdivided:
			for child_ind in range(8):	
				self.children[child_ind].divide()
		else:
			# first division of root node logic
			if self.root is self: #add bounding box vertices, this should only be called on the very first divide() call on the root node
				self.points.update([self.bbox.vertex(n) for n in range(8)])


			# test if division is necessary
			if len(self.particle_ind_list) == 0: return #no need to divide here
			if self.nvert == 8: return #particles are convex, no need to divide here, all child regions would have nvert=8 as well
			if self.depth+1 > self.root.maxdepth: return #too deep

			# create children
			for child_ind in range(8):
				subbox = Box(self.bbox.vertex(child_ind), self.bbox.center)
				self.children.append(Node(subbox, self.depth+1))

				self.children[child_ind].nvert  = 0
				self.children[child_ind].root   = self.root
				self.children[child_ind].parent = self

			# update populate child attributes particle list and nvert, decide which particles should be passed down
			for child_ind in range(8):
				child = self.children[child_ind]

				# update point/vertex list in root
				for n in range(8):
					self.root.points.add(child.bbox.vertex(n))

				# loop through particles to see which should be passed down to the child
				for p_ind in range(len(self.particle_ind_list)):
					P = self.root.particle_list[self.particle_ind_list[p_ind]]

					if abs(child.bbox.center - P.center) <= 0.5*(child.bbox.diam + P.diam):

						#compute vertex intersections
						p_child_nvert = 0 #number of vertices of this child contained in this particle
						for n in range(8):
							templevelval = P.levelval(child.bbox.vertex(n))
							minlevelval = min(minlevelval, templevelval)
							if templevelval <= 1.0:
								p_child_nvert += 1
						child.nvert = max(child.nvert, p_child_nvert)
						# print(minlevelval, p_child_nvert)

						if p_child_nvert > 0:
							child.particle_ind_list.append(self.root.particle_list.index(P))
						elif child.bbox.diam >= P.diam:
							child.particle_ind_list.append(self.root.particle_list.index(P))
						else: #make a closer inspection in case the particle intersects a face but not a vertex
							if _closest_point_neldermead(P.levelval, child.bbox, verbose=0) <= 1.0:
								child.particle_ind_list.append(self.root.particle_list.index(P))
				
			
			#change parameters from self for a non-leaf node
			self.particle_ind_list = []
			self.nvert = -1 #only leaf nodes should have a valid marker
			self.isdivided = True
			self.root.totaldepth = max(self.root.totaldepth, self.depth+1)


	# cdef list getleaveswithpoint(self, Vector point): #get a list of leaves with the specified point
	# 	cdef list _list = []
	# 	cdef int child_ind

	# 	if self.isdivided:
	# 		for child_ind in range(8):
	# 			_list += self.children[child_ind].getleaveswithpoint(point)
	# 	elif self.bbox.contains(point):
	# 		_list.append(self)
	# 	return _list

	# cdef list getneighbors(self, int n): #must be called on a leaf node
	# 	cdef Vector facecenter = self.bbox.facecenter(n)


	cpdef Node getnode(self, Vector point): #return first leaf node that contains the point. point should be interior, but not strictly necessary
		cdef Node node = Node(None, NONE_DEPTH)
		if self.bbox.contains(point):
			if self.isdivided:
				for n in range(8):
					node = self.children[n].getnode(point)
					if not (node.depth == NONE_DEPTH):
						return node
			else: return self
		return node

	cpdef bint inparticle(self, Vector point):
		cdef Node N = self.getnode(point)
		cdef int p_ind
		for p_ind in range(len(N.particle_ind_list)):
			P = self.root.particle_list[N.particle_ind_list[p_ind]]
			if P.contains(point): return True

		return False

	cpdef void appendparticle(self, particle_t P):
		if self.root is self:
			self.particle_list.append(P)
			self.particle_ind_list.append(self.particle_list.index(P))

	cpdef int sumnvert(self): #set each parent node nvert to be the sum of the children
		cdef int n
		if self.isdivided:
			self.nvert = 0
			for n in range(8):
				self.nvert += self.children[n].sumnvert()
		else:
			return self.nvert


	cpdef void printall(self):
		print("-"*4*self.depth, f"depth={self.depth}, nvert= {self.nvert}, bbox= {self.bbox}, particle_ind_list= {self.particle_ind_list}")

		cdef int n
		if self.isdivided:
			for n in range(8):
				self.children[n].printall()

		if self.root is self:
			for n in range(len(self.particle_list)):
				repr(self.particle_list[n])


	cpdef list leaflist(self):
		cdef list _list = []
		cdef int child_ind

		if self.isdivided:
			for child_ind in range(8):
				_list += self.children[child_ind].leaflist()
		else:
			_list.append(self)
		return _list


	cpdef void to_vtk(self, str filename):
		cdef int leaf_ind, n

		#get leaves
		leaves = self.leaflist()

		#get list of nodes, sort for better node ordering
		cdef list _nodelist  = sorted(self.root.points)

		#get voxel element node indices using a hash/dictionary
		cdef dict _nodedict = dict(enumerate(_nodelist))
		cdef dict _getnodeindex = dict(zip(_nodedict.values(), _nodedict.keys()))

		#number of leaves/elements and points
		cdef int n_leaves = len(leaves)
		cdef int n_points = len(_nodelist)

		#write .vtk file
		with open(filename, 'w') as _file:
			###############  header ###############
			_file.write("# vtk DataFile Version 2.0\n")
			_file.write("Octree Structure\n")
			_file.write("ASCII\n\n")

			############## points ###############
			_file.write("DATASET UNSTRUCTURED_GRID\n")
			_file.write(f"POINTS {n_points} float\n")
			string = []
			for leaf_ind from 0 <= leaf_ind < n_points: #Print node locations
				string.append(f"{_nodelist[leaf_ind].x} {_nodelist[leaf_ind].y} {_nodelist[leaf_ind].z}\n")
			string.append("\n")
			_file.write(''.join(string))

			############## voxels ###############
			_file.write(f"CELLS {n_leaves} {n_leaves+8*n_leaves}\n")
			string = []
			for leaf_ind from 0 <= leaf_ind < n_leaves:
				string.append("8 ")
				for n from 0 <= n < 8:
					ind = _getnodeindex[leaves[leaf_ind].bbox.vertex(n)]
					string.append(f"{ind} ")
				string.append("\n")
			string.append("\n")
			_file.write(''.join(string))

			_file.write(f"CELL_TYPES {n_leaves}\n")
			string = "11\n"*n_leaves #might need \n
			string+= "\n"
			_file.write(string)

			############## nvert ################
			_file.write(f"CELL_DATA {n_leaves}\n")
			_file.write("SCALARS nvert double\n")
			_file.write("LOOKUP_TABLE default\n")
			string = []
			for leaf_ind from 0<= leaf_ind < n_leaves:
				string.append(f"{leaves[leaf_ind].nvert}\n")
			string.append("\n")
			_file.write(''.join(string))

	#representation
	def __repr__(self): return f"Node({repr(self.bbox)}, {repr(self.depth)})"
	def __str__(self): 
		string  = f"Node:\n\tdepth= {self.depth}"
		string += f"\n\tbbox.low= {self.bbox}"
		string += f"\n\tnvert= {self.nvert}"
		string += f"\n\tleavesbelow= {len(self.leaflist())}"
		string += f"\n\tnparticles= {len(self.particle_list)}"
		
		if self.root is self:
			for p_ind in range(len(self.root.particle_list)):
				string += f"\n\t({p_ind})\t{repr(self.root.particle_list[p_ind])}"
		
		return string



cdef class Assembly(Node):
	def __init__(self, subsampleres = Vector(100, 100, 100), list particle_list = []):
		super().__init__(Box(Vector(-1,-1,-1), Vector(1,1,1)), 0)

		self.maxdepth = 6
		self.particle_list = particle_list
		self.bbox = self.getbbox()

		self.subsamplebox = self.bbox.copy()
		self.subsampleres = subsampleres

	cpdef Box getbbox(self): #get bbox for particle_list
		if len(self.particle_list) == 0: return Box(Vector(-1,-1,-1), Vector(1,1,1))
		cdef int n
		cdef Box bbox = self.particle_list[0].bbox.copy()
		for n in range(1,len(self.particle_list)):
			bbox @= self.particle_list[n].bbox #smallest bounding box containing box and bbox
		return bbox


	cpdef void to_dat(self, str filename):
		#check if resolution is sufficient
		# cdef Vector h = (1.0/2**self.totaldepth)*self.bbox.sidelength
		cdef Vector target_h = self.subsamplebox.sidelength/self.subsampleres
		# if not target_h < h: raise Exception("Mesh is not fine enough. Increase maxdepth and re-process or reduce required resolution.")

		cdef list block
		cdef Vector centroid = Vector(0,0,0)
		cdef int i, j, k
		with open(filename, 'w') as _file:
			###############  header ###############
			_file.write("nx= {:d}\n".format(<int>self.subsampleres[0]))
			_file.write("ny= {:d}\n".format(<int>self.subsampleres[1]))
			_file.write("nz= {:d}\n".format(<int>self.subsampleres[2]))

			for k in range(self.subsampleres[2]):
				centroid[2] = self.subsamplebox.low[2]+(k+0.5)*target_h[2]
				block = ['\n']
				for j in range(self.subsampleres[1]):
					centroid[1] = self.subsamplebox.low[1]+(j+0.5)*target_h[1]
					for i in range(self.subsampleres[0]):
						centroid[0] = self.subsamplebox.low[0]+(i+0.5)*target_h[0]
						block.append(f"{<int> self.inparticle(centroid)} ")
						# print(centroid)
					block.append('\n')
				_file.write(''.join(block))




	cpdef void process(self):
		self.bbox = self.getbbox()

		cdef int n
		for n in range(self.root.maxdepth):
			self.divide()
			print(f"depth= {self.totaldepth}: {len(self.points)} vertices")


	cpdef void read(self, str filename, str pType = 'SuperEllipsoid'):
		cdef Vector radii, center
		cdef double eps1, ep2
		cdef Quaternion quat

		cdef int lineno = 0
		with open(filename, 'r') as _file:
			# skip over comments in header
			line = "#"
			while line[0] == "#":
				line = _file.readline()
				lineno += 1

			# add shapes to list
			while line != "":
				par      = line.split('\t')
				
				radii    = Vector(atof(par[0].encode()), atof(par[1].encode()), atof(par[2].encode()))
				eps1     = atof(par[3].encode())
				eps2     = atof(par[4].encode())
				center   = Vector(atof(par[5].encode()), atof(par[6].encode()), atof(par[7].encode()))
				quat     = Quaternion(atof(par[8].encode()), Vector(atof(par[9].encode()), atof(par[10].encode()), atof(par[11].encode())))
				
				if pType == 'Sphere': self.appendparticle(Sphere(0.33333*sum(radii), center))
				elif pType == 'Prism': self.appendparticle(Prism(radii, center, quat))
				elif pType == 'Ellipsoid': self.appendparticle(Ellipsoid(radii, center, quat))
				elif pType == 'SuperEllipsoid': self.appendparticle(SuperEllipsoid(radii, eps1, eps2, center, quat))
				else: print(f"unkown particle type on line {lineno} of {filename}: {line}")

				line   = _file.readline()
				lineno += 1

		self.bbox = self.getbbox()



cpdef void dat_to_vtk(Box bbox, str infile, str outfile):
	cdef int nx, ny, nz
	cdef Vector H 

	cdef set pointlist = set()
	cdef list elemboxes = []
	cdef list elemvalue = []

	cdef Vector low = Vector(3)
	cdef Vector high = Vector(3)
	cdef Box elem

	cdef int i, j, k, n
	with open(infile,'r') as file:
		# read dimensions
		line = file.readline()
		nx = int(line.strip("nx= "))
		
		line = file.readline()
		ny = int(line.strip("ny= "))

		line = file.readline()
		nz = int(line.strip("nz= "))

		# save cell size
		H = Vector(bbox.sidelength.x/nx, bbox.sidelength.y/ny, bbox.sidelength.z/nz)
		# print(H, bbox)
		# read mask from file
		for k in range(nz):
			low.z = bbox.low.z + k*H.z
			for j in range(ny):
				low.y = bbox.low.y + j*H.y
				line = file.readline()
				vals = line.split(' ')
				
				if len(vals) > 1:
					for i in range(nx):
						low.x = bbox.low.x + i*H.x
						elem = Box(low,low+H)
						elemboxes.append(elem)
						elemvalue.append(atoi(vals[i].encode()))

						# print(low)
						for n in range(8):
							pointlist.add(elem.vertex(n))

			line = file.readline()

	# write to .vtk
	#get list of nodes, sort for better node ordering
	cdef list _nodelist  = sorted(pointlist)

	#get voxel element node indices using a hash/dictionary
	cdef dict _nodedict = dict(enumerate(_nodelist))
	cdef dict _getnodeindex = dict(zip(_nodedict.values(), _nodedict.keys()))

	#number of elements and points
	cdef int n_elem = len(elemboxes)
	cdef int n_points = len(_nodelist)

	#write .vtk file
	cdef list string
	cdef elem_index, ind
	with open(outfile, 'w') as _file:
		###############  header ###############
		_file.write("# vtk DataFile Version 2.0\n")
		_file.write("Octree Structure\n")
		_file.write("ASCII\n\n")

		############## points ###############
		_file.write("DATASET UNSTRUCTURED_GRID\n")
		_file.write(f"POINTS {n_points} float\n")
		string = []
		for elem_index from 0 <= elem_index < n_points: #Print node locations
			string.append(f"{_nodelist[elem_index].x} {_nodelist[elem_index].y} {_nodelist[elem_index].z}\n")
		string.append("\n")
		_file.write(''.join(string))

		############## voxels ###############
		_file.write(f"CELLS {n_elem} {n_elem+8*n_elem}\n")
		string = []
		for elem_index from 0 <= elem_index < n_elem:
			string.append("8 ")
			for n from 0 <= n < 8:
				ind = _getnodeindex[elemboxes[elem_index].vertex(n)]
				string.append(f"{ind} ")
			string.append("\n")
		string.append("\n")
		_file.write(''.join(string))

		_file.write(f"CELL_TYPES {n_elem}\n")
		string = ["11\n"]*n_elem
		string.append("\n")
		_file.write(''.join(string))

		############## nvert ################
		_file.write(f"CELL_DATA {n_elem}\n")
		_file.write("SCALARS centroid double\n")
		_file.write("LOOKUP_TABLE default\n")
		string = []
		for elem_index from 0<= elem_index < n_elem:
			string.append(f"{elemvalue[elem_index]}\n")
		string.append("\n")
		_file.write(''.join(string))