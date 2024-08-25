from geovox.utilities cimport Vector3, Box
from geovox.vtkutil cimport VtkVoxel, IndexIJK

cdef VtkVoxel VTK_VOXEL = VtkVoxel()

# node of octree with integer vertex data. A the getvertexdata(Vector3 point) function should be over-written in the tree constructor class.
cdef class OctreeNode:
	def __init__(self, Node parent, Box bbox, int depth):
		self.parent = parent
		self.isdivided = False
		self.children = None
		self.depth = depth
		self.bbox = bbox

		if self.parent is None:
			for n in range(8):
				self.vertexdata[n] = self.computevertexdata(self.bbox.vertex(n))  #.vtk ordering
		else:
			self.vertexdata = None
		
	cpdef divide(self):
		#compute vertexdata here to avoid visiting the same node multiple times
		cpdef int[3][3][3] localdata
		cpdef Vector3 delta = Vector3(0,0,0)
		cpdef Vector3 h = 0.5*self.bbox.sidelength
		for i in range(3):
			delta.x = self.bbox.low.x + i*h.x
			for j in range(3):
				delta.y = self.bbox.low.y + j*h.y
				for k in range(3):
					delta.z = self.bbox.low.z + i*h.z

					#use existing data
					if   (i==0 and j==0 and k==0): localdata[i][j][k] = self.vertexdata[0]
					elif (i==2 and j==0 and k==0): localdata[i][j][k] = self.vertexdata[1]
					elif (i==2 and j==2 and k==0): localdata[i][j][k] = self.vertexdata[2]
					elif (i==0 and j==2 and k==0): localdata[i][j][k] = self.vertexdata[3]
					elif (i==0 and j==0 and k==2): localdata[i][j][k] = self.vertexdata[4]
					elif (i==2 and j==0 and k==2): localdata[i][j][k] = self.vertexdata[5]
					elif (i==2 and j==2 and k==2): localdata[i][j][k] = self.vertexdata[6]
					elif (i==0 and j==2 and k==2): localdata[i][j][k] = self.vertexdata[7]
					else:
						localdata[i][j][k] = self.computevertexdata(Vector3(self.bbox.low + delta))

		# divide and copy data
		self.isdivided = True
		cdef Box subbox
		cdef IndexIJK lowindex, ijk
		for n in range(8):
			subbox= Box(self.bbox.vertex(n), self.bbox.center)
			self.children[n] = OctreeNode(self, subbox, self.depth+1)

			#get ijk for lower left vertex and copy data
			lowindex = VTK_VOXEL.ijk(n)
			for i in range(2):
				ijk[0] = i
				for j in range(2):
					ijk[1] = j
					for k in range(2):
						ijk[2] = k
						self.children[n].vertexdata[VTK_VOXEL.n(ijk)] = localdata[lowindex[0]+i][lowindex[1]+j][lowindex[2]+k] #copy local data to child node

			

	cpdef int computevertexdata(self, Vector3 point): #placeholder
		print("You are using the all-zeros computevertexdata() placeholder method.")
		return 0

	@property
	def isleaf(self): return not self.isdivided
	@property
	def depth(self): return self.depth
	@property
	def bbox(self): return self.bbox

	def __repr__(self):
		return f"Node({repr(self.parent)}, {repr(self.bbox)}, {repr(self.depth)})"

	def __str__(self):
		return f"OctreeNode: depth= {self.depth}, isdivided= {self.isdivided}, bbox={repr(self.bbox)}"


# main tree
cdef class Octree:
	def __init__(self, int maxdepth, Box bbox):
		self.maxdepth = maxdepth
		self.bbox     = bbox