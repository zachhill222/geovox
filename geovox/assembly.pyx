# from geovox.utilities cimport Box, Vector3
from geovox.vtkutil cimport VtkVoxel, IndexIJK
from geovox.optimize cimport NelderMead
# from libc.math cimport sqrt


cdef class Node: #node of octree
	def __init__(self, Box bbox, int depth):
		self.isdivided = False
		self.children = []
		self.depth = depth
		self.bbox = bbox
		self.particle_list = []
		self.nvert = -1

		self.root = self #for storing global tree information
		self.points = set() #if not calling divide(), this may need to be replaced with [self.bbox.vertex(n) for n in range(8)]
		# self.pointindex = [] #if not calling divide(), this may need to be replaced with [n for n in range(8)] converted to int[8]


	@property
	def depth(self): return self.depth
	@property
	def bbox(self): return self.bbox

	cpdef void divide(self):
		cdef Box subbox
		cdef int nvert_temp

		#optimization settings
		cdef double minlevelval
		cdef double facelevelval, facelevelval_temp
		cdef int n, m, i, j
		cdef int n_grid
		cdef bint stopevals
		cdef double theta1, theta2
		cdef NelderMead optimizer

		if self.isdivided:			
			for child in self.children: child.divide()
		else:
			if len(self.particle_list) == 0: return #no need to divide here
			if self.nvert == 8: return #particles are convex, no need to divide here, all child regions would have nvert=8 as well
			
			#parameters for checking dubious voxels
			# minlevelval = 1.0
			# n_grid = max(10-self.depth, 2)

			# create children
			for n in range(8):
				subbox= Box(self.bbox.vertex(n), self.bbox.center)
				self.children.append(Node(subbox, self.depth+1))

			# update particle list and nvert
			for child in self.children:
				child.nvert = 0

				for P in self.particle_list:
					if child.bbox.intersects(P.bbox):
						#compute vertex intersections
						nvert_temp = 0
						for n in range(8):
							if P.contains(child.bbox.vertex(n)): nvert_temp += 1
						child.nvert = max(child.nvert, nvert_temp)

						if child.nvert > 0:
							child.particle_list.append(P)
						else:
							minlevelval = 1.01# + 0.001*abs(child.bbox.sidelength)
							optimizer = NelderMead(P.levelval, child.bbox, minlevelval)
							optimizer.maxiter = 100
							optimizer.penalty_weight = 2**self.depth
							# print("NelderMead")
							if optimizer.minimize() <= minlevelval:
								child.particle_list.append(P)

			# update root points and vertex index
			for child in self.children:
				child.root = self.root
				self.root.points.update([child.bbox.vertex(n) for n in range(8)])
			
			#change parameters from self for a non-leaf node
			self.particle_list = []
			self.nvert = -1 #only leaf nodes should have a valid marker
			self.isdivided = True


	cpdef void insertparticle(self, particle_t P):
		cdef int nvert_temp
		
		if self.isdivided:
			for child in self.children: child.insertparticle(P)
		else:
			if self.bbox.intersects(P.bbox):
				self.particle_list.append(P)
				
				nvert_temp = 0
				for n in range(8):
					nvert_temp += P.contains(self.bbox.vertex(n))
				self.nvert = max(self.nvert, nvert_temp)


	cpdef Node getnode(self, Vector3 point): #return first leaf node that contains the point. point should be interior, but not strictly necessary
		cdef Node node = Node(None, None, NONE_DEPTH)
		# print(f"Checking node: {self}\n")
		if self.bbox.contains(point):
			if self.isdivided:
				for n in range(8):
					node = self.children[n].getnode(point)
					if not (node.depth == NONE_DEPTH):
						return node
			else: return self
		return node

	cpdef bint contains(self, Vector3 point): #determine if point is contained in any particle
		cdef node = self.getnode(point)
		if node.depth == NONE_DEPTH: return False #point is out of bounds

		for P in node.particle_list:
			if P.contains(point): return True
		return False



	@property
	def children(self): return self.children

	cpdef list leaflist(self):
		_list = []
		if self.isdivided:
			for child in self.children:
				_list += child.leaflist()
		else:
			_list.append(self)
		return _list


	cpdef void voxelmesh(self, str filename):
		#get leaves
		leaves = self.leaflist()

		#get list of nodes, sort for better node ordering
		cdef list _nodelist  = sorted(self.root.points)

		#get voxel element node indices using a hash/dictionary
		cdef dict _nodedict = dict(enumerate(_nodelist))
		cdef dict _getnodeindex = dict(zip(_nodedict.values(), _nodedict.keys()))
		cdef list _voxellist = [ [_getnodeindex[leaf.bbox.vertex(n)] for n in range(8)] for leaf in leaves]

		#get nvert data from leaves
		cdef list _nvertlist = [leaf.nvert for leaf in leaves]

		#write .vtk file
		with open(filename, 'w') as _file:
			###############  header ###############
			_file.write("# vtk DataFile Version 2.0\n")
			_file.write("Octree Structure\n")
			_file.write("ASCII\n\n")

			############## points ###############
			_file.write("DATASET UNSTRUCTURED_GRID\n")
			_file.write(f"POINTS {len(_nodelist)} float\n")
			string = []
			for i in range(len(_nodelist)): #Print node locations
				string.append(f"{_nodelist[i].x} {_nodelist[i].y} {_nodelist[i].z}\n")
			string.append("\n")
			_file.write(''.join(string))

			############## voxels ###############
			_file.write(f"CELLS {len(_voxellist)} {len(_voxellist)+8*len(_voxellist)}\n")
			string = []
			for i in range(len(_voxellist)):
				string.append("8 ")
				for j in range(8):
					ind = _voxellist[i][j]
					string.append(f"{ind} ")
				string.append("\n")
			string.append("\n")
			_file.write(''.join(string))

			_file.write(f"CELL_TYPES {len(_voxellist)}\n")
			string = "11\n"*len(_voxellist) #might need \n
			string+= "\n"
			_file.write(string)

			############## depth ################
			# _file.write(f"CELL_DATA {len(_depthlist)}\n")
			# _file.write("SCALARS treedepth double\n")
			# _file.write("LOOKUP_TABLE default\n")
			# string = []
			# for i in range(len(_depthlist)):
			# 	string.append(f"{_depthlist[i]}\n")
			# string.append("\n")
			# _file.write(''.join(string))

			############## nvert ################
			_file.write(f"CELL_DATA {len(_nvertlist)}\n")
			_file.write("SCALARS nvert double\n")
			_file.write("LOOKUP_TABLE default\n")
			string = []
			for i in range(len(_nvertlist)):
				string.append(f"{_nvertlist[i]}\n")
			string.append("\n")
			_file.write(''.join(string))

	#representation
	def __repr__(self): return f"Node({repr(self.bbox)}, {repr(self.depth)})"
	def __str__(self): 
		string  = f"Node:\n\tdepth= {self.depth}"
		string += f"\n\tbbox.low= {self.bbox.low}, bbox.high= {self.bbox.high}"
		string += f"\n\tnvert= {self.nvert}"
		string += f"\n\tnparticles= {len(self.particle_list)}"
		string += f"\n\tleavesbelow= {len(self.leaflist())}"
		return string



cdef class Assembly:
	def __init__(self, Box bbox, int maxdepth, list particle_list):
		self.maxdepth = maxdepth
		self.root = Node(bbox, 0)