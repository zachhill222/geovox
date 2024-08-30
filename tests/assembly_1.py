from geovox.geometry import Node
from geovox.geometry import Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.util import Box, Quaternion
from geovox.linalg import Vector

from time import time
from sys import getsizeof

box = Box(Vector(-2,-2,-2), Vector(2,2,2))


Q   = Quaternion(1, Vector(0,0,0))
Q.setrotation(1, Vector(1,1,1))

# P   = Sphere(4/3, Vector(0,0,0))
# P   = Prism(Vector(1,0.5,0.25), Vector(0,0,0), Q)
P   = Ellipsoid(Vector(1,0.5,0.125), Vector(0,0,0), Q)
# P   = SuperEllipsoid(Vector(1,0.5,1), 1.0, 2.0, Vector(0,0,0), Q)



tree = Node(box, 0) #add support for adding boxes
tree.insertparticle(P-Vector(1,-0.5,1))
# tree.insertparticle(Sphere(0.7, Vector(-0.75,-0.75,-1)))
# tree.insertparticle(Prism(Vector(1,0.5,0.5), Vector(1.4,1.5,1.5), Q)) #not sure why this dissapears sometimes
tree.insertparticle(P*Q.conj()+Vector(1,-0.5,1))


depth = 6
for n in range(depth):
	print(f"depth= {n}")
	tree.divide()
	tree.voxelmesh(f"./tests/outfiles/assembly_depth_{n}.vtk")
