cdef class Node: #node of octree
	def __init__(self, Box bbox, int depth):
		self.isdivided = False
		self.children = []
		self.depth = depth
		self.bbox = bbox
		self.particle_list = []
		self.particle_ind_list = []
		self.nvert = -1

		self.root = self #for storing global tree information
		self.points = set() #if not calling divide(), this may need to be replaced with [self.bbox.vertex(n) for n in range(8)]


	@property
	def depth(self): return self.depth
	@property
	def bbox(self): return self.bbox

	cpdef void divide(self):
		cdef Box subbox
		cdef int p_child_nvert, p_n_child_added

		cdef double minlevelval, templevelval
		cdef int n, m, i, j, p_ind, child_ind
		cdef Vector closestpoint, vertex

		if self.isdivided:			
			for child in self.children: child.divide()
		else:
			# first division of root node logic
			if self.root is self: #add bounding box vertices, this should only be called on the very first divide() call on the root node
				self.points.update([self.bbox.vertex(n) for n in range(8)])


			# test if division is necessary
			if len(self.particle_ind_list) == 0: return #no need to divide here
			if self.nvert == 8: return #particles are convex, no need to divide here, all child regions would have nvert=8 as well


			# create children
			for child_ind in range(8):
				subbox= Box(self.bbox.vertex(child_ind), self.bbox.center)
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
						minlevelval = 999999
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
							# closestpoint = _closest_point_neldermead(P, child.bbox)
							closestpoint = _closest_point_gradient(P, child.bbox)
							if P.levelval(closestpoint) <= 1.01:
								child.particle_ind_list.append(self.root.particle_list.index(P))
						# else: child.particle_ind_list.append(self.root.particle_list.index(P))
				
			
			#change parameters from self for a non-leaf node
			self.particle_ind_list = []
			self.nvert = -1 #only leaf nodes should have a valid marker
			self.isdivided = True


	# cpdef void gradate(self):
	# 	cdef list _pointlist  = sorted(self.root.points)
	# 	cdef list _leaflist
	# 	cdef int point_ind, leaf_ind1, leaf_ind2, max_depth, min_depth
	# 	for point_ind in range(len(_pointlist)):
	# 		_leaflist = self.getleaveswithpoint(_pointlist[point_ind])
	# 		for leaf_ind1 in range(len(_leaflist)-1):
	# 			for leaf_ind2 in range(leaf_ind1+1, len(_leaflist)):
	# 				if _leaflist[]



	cpdef void insertparticle(self, particle_t P):
		if self is self.root:
			self.root.particle_list.append(P)
			self.root.particle_ind_list.append(self.root.particle_list.index(P))

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


	# cpdef Node getnode(self, Vector point): #return first leaf node that contains the point. point should be interior, but not strictly necessary
	# 	cdef Node node = Node(None, None, NONE_DEPTH)
	# 	if self.bbox.contains(point):
	# 		if self.isdivided:
	# 			for n in range(8):
	# 				node = self.children[n].getnode(point)
	# 				if not (node.depth == NONE_DEPTH):
	# 					return node
	# 		else: return self
	# 	return node


	cpdef list leaflist(self):
		cdef list _list = []
		cdef int child_ind

		if self.isdivided:
			for child_ind in range(8):
				_list += self.children[child_ind].leaflist()
		else:
			_list.append(self)
		return _list


	cpdef void voxelmesh(self, str filename):
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
		string += f"\n\tbbox.low= {self.bbox.low}, bbox.high= {self.bbox.high}"
		string += f"\n\tnvert= {self.nvert}"
		string += f"\n\tleavesbelow= {len(self.leaflist())}"
		string += f"\n\tnparticles= {len(self.particle_list)}"
		for p_ind in range(len(self.root.particle_list)):
			string += f"\n\t({p_ind})\t{repr(self.root.particle_list[p_ind])}"
		return string



# cdef class Assembly:
# 	def __init__(self, Box bbox, int maxdepth, list particle_list):
# 		self.maxdepth = maxdepth
# 		self.root = Node(bbox, 0)