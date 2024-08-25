from geovox.utilities cimport Box, Vector3
from geovox.vtkutil cimport VtkVoxel, IndexIJK




cdef class Node: #node of octree
	def __init__(self, Node parent, Box bbox, int depth):
		self.parent = parent
		self.isdivided = False
		self.children = []
		self.depth = depth
		self.bbox = bbox
		self.particle_list = []

	cpdef void divide(self):
		self.isdivided = True

		cdef Box subbox
		for n in range(8):
			subbox= Box(self.bbox.vertex(n), self.bbox.center)
			self.children.append(Node(self, subbox, self.depth+1))

		for P in self.particle_list:
			for n in range(8):
				if P.bbox.intersects(self.children[n]):
					self.children.particle_list.append(P)

		self.particle_list = []


	@property
	def children(self): return self.children

	#helpers for writing vtk files
	def nodelist(self):
		_list = []
		if self.isdivided:
			for n in range(8):
				_list += self.children[n].nodelist()
		else:
			for n in range(8):
				_list.append(self.bbox.vertex(n))

		_uniquelist = []
		for pt in _list:
			if not (pt in _uniquelist):
				_uniquelist.append(pt)
		return _uniquelist

	def voxellist(self):
		_list = []
		if self.isdivided:
			for n in range(8):
				_list += self.children[n].voxellist()
		else:
			_list.append([self.bbox.vertex(n) for n in range(8)]) #coordinates, will need to change to indicies
		return _list

	def depthlist(self):
		_list = []
		if self.isdivided:
			for n in range(8):
				_list += self.children[n].depthlist()
		else:
			_list.append(self.depth)
		return _list

	def voxelmesh(self, filename):
		_nodelist = self.nodelist()
		_voxellist = self.voxellist()
		_depthlist = self.depthlist()

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
					ind = _nodelist.index(_voxellist[i][j])
					string.append(f"{ind} ")
				string.append("\n")
			string.append("\n")
			_file.write(''.join(string))

			_file.write(f"CELL_TYPES {len(_voxellist)}\n")
			string = "11\n"*len(_voxellist) #might need \n
			string+= "\n"
			_file.write(string)

			############## depth ################
			_file.write(f"CELL_DATA {len(_depthlist)}\n")
			_file.write("SCALARS treedepth double\n")
			_file.write("LOOKUP_TABLE default\n")
			string = []
			for i in range(len(_depthlist)):
				string.append(f"{_depthlist[i]}\n")
			_file.write(''.join(string))











# cdef class Assembly:
# 	def __init__(self, Box bbox, int maxdepth, particle_list):
# 		self.box = bbox
# 		self.maxdepth

# 	cpdef 