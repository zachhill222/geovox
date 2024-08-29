from geovox.geometry import Node
from geovox.geometry import Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.utilities import Vector, Box, Quaternion

from time import time
from sys import getsizeof

box = Box(Vector(-2,-2,-2), Vector(2,2,2))


Q   = Quaternion(1, Vector(0,0,0))
# Q.setrotation(1, Vector(1,1,1))

# P   = Sphere(4/3, Vector(0,0,0))
# P   = Prism(Vector(1,0.5,0.25), Vector(0,0,0), Q)
# P   = Ellipsoid(Vector(1,0.5,0.125), Vector(0,0,0), Q)
P   = SuperEllipsoid(Vector(1,0.5,0.25), 1.0, 3.0, Vector(0,0,0), Q)



print(P)

tree = Node(box, 0)
tree.insertparticle(P)

depth = 8
print(f"Constructing octree to depth= {depth}: ", end="")
tic = time()
for n in range(depth):
	tree.divide()
toc = time()
print(f"{toc-tic} seconds, {getsizeof(tree)/(2**20)} MB")

print(tree)


print(f"Creating VTK voxel mesh: ", end="")
tic = time()
tree.voxelmesh("./tests/outfiles/assembly_1.vtk")
toc = time()
print(f"{toc-tic} seconds")