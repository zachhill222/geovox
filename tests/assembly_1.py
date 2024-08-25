from geovox.assembly import Node
from geovox.utilities import Vector3, Box


box = Box(Vector3(0,0,0), Vector3(1,1,1))
tree = Node(None, box, 0)
tree.divide()
tree.children[0].divide()
tree.children[0].children[0].divide()

tree.voxelmesh("assembly_1.vtk")