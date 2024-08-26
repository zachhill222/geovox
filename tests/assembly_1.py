from geovox.assembly import Node
from geovox.utilities import Vector3, Box, Quaternion
from geovox.particles import Sphere, Prism, Ellipsoid, SuperEllipsoid


box = Box(Vector3(-1,-1,-1), Vector3(1,1,1))
# P   = Sphere(4/3, Vector3(0,0,0))

Q   = Quaternion(1, Vector3(0,0,0))
# Q.setrotation(1, Vector3(1,1,0))
# P   = Prism(Vector3(1,0.5,0.125), Vector3(0,0,0), Q)

# P   = Ellipsoid(Vector3(1,0.5,0.125), Vector3(0,0,0), Q)

P   = SuperEllipsoid(Vector3(1,0.5,0.125), 0.75, 1.5, Vector3(0,0,0), Q)



print(P)

tree = Node(box, 0)
tree.insertparticle(P)

for n in range(7):
	tree.divide()

print(tree)

tree.voxelmesh("./tests/outfiles/assembly_1.vtk")
