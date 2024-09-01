from geovox.geometry import Node
from geovox.geometry import Sphere, Prism, Ellipsoid, SuperEllipsoid
from geovox.util import Box, Quaternion
from geovox.linalg import Vector

from time import time
from sys import getsizeof

box = Box(Vector(-2,-2,-2), Vector(2,2,2))


Q   = Quaternion(1, Vector(0,0,0))
Q.setrotation(1, Vector(1,1,1))

P   = SuperEllipsoid(Vector(1,0.5,1), 1.0, 2.0, Vector(0,0,0), Q)



root = Node(box, 0)
root.appendparticle(P-Vector(1,-0.5,1))
root.appendparticle(Sphere(0.7, Vector(-0.75,-0.75,-1)))
root.appendparticle(Prism(Vector(1,0.5,0.5), Vector(1.4,1.5,1.5), Q))
root.appendparticle(P*Q.conj()+Vector(1,-0.5,1))
root.appendparticle(Ellipsoid(Vector(1,0.5,0.125), Vector(-1.5,-1.5,1.5), Q))


for n in range(6):
	tic = time()
	root.divide()
	root.to_vtk(f"./tests/outfiles/nodetest_depth_{root.totaldepth}.vtk")
	print(f"depth= {root.totaldepth}: {len(root.leaflist())} leaves, {len(root.points)} vertices, {time()-tic} seconds, {getsizeof(root)/(2**10)} KB")