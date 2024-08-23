from geovox.utilities import Vector3, Quaternion
from geovox.particles import Sphere, Prism, Ellipsoid, SuperEllipsoid

from time import time
from sys import getsizeof
from numpy import pi as PI
from numpy import sqrt


# SPHERE
print(f"\n\n=============== TESTING Sphere ===============")
P = Sphere(2, Vector3(1,2,3))
print(f"P= {repr(P)}")
print(P)
print(f"P.volume= {P.volume}")
print(f"P.bbox= {repr(P.bbox)}")

print("\nTESTING CENTER")
P.center = Vector3(0,0,0)
print(f"P.center = Vector3(0,0,0) -> P.bbox= {repr(P.bbox)}")

print("\nTESTING RADIUS")
P.R*=2
print(f"P.R*=2 -> print(P)")
print(P)
print(f"P.volume= {P.volume}")

print("\nTESTING CONTAINS")
point = Vector3(4,0,0)
print(f"P.contains({repr(point)})= {P.contains(point)}")
point = Vector3(4,0,1e-5)
print(f"P.contains({repr(point)})= {P.contains(point)}")

# PRISM
print(f"\n\n=============== TESTING Prism =============== ")
P = Prism(Vector3(1,2,3), Vector3(0,0,0), Quaternion(1, Vector3(0,0,0)))
print(f"P= {repr(P)}")
print(P)
print(f"P.volume= {P.volume}")
print(f"P.bbox= {repr(P.bbox)}")
P.center = Vector3(1,2,3)

print("\nTESTING CENTER")
print(f"P.center = Vector3(1,2,3) -> P.bbox= {repr(P.bbox)}")
print(P)

P.center-=P.center
print(f"P.center-=P.center -> print(P)")
print(P)

print("\nTESTING RADII")
newR = Vector3(1,sqrt(2),3)
P.R = newR
print(f"P.R= {repr(newR)} -> P.bbox= {repr(P.bbox)}")
print(P)

print("\nTESTING QUATERNION")
newQ = Quaternion(1,Vector3(0,0,0)).setrotation(PI/2, Vector3(0,0,1))
P.Q = newQ
print(f"P.Q= {repr(newQ)} -> P.bbox= {repr(P.bbox)}")

print("\nprint(P):")
print(P)
print("\nprint(P.bbox):")
print(P.bbox)

print("\nTESTING vertex()")
for n in range(8):
	print(f"\tP.vertex({n})= {repr(P.vertex(n))}")

print("\nTESTING facecenter()")
for n in range(6):
	print(f"\tP.facecenter({n})= {repr(P.facecenter(n))}")

print("\nTESTING CONTAINS")
point = P.vertex(0)
print(f"P.contains({repr(point)})= {P.contains(point)}")
point += Vector3(1e-5,-1e-5,0)
print(f"P.contains({repr(point)})= {P.contains(point)}")

point = P.facecenter(4)
print(f"P.contains({repr(point)})= {P.contains(point)}")
point += Vector3(0,0,-1e-5)
print(f"P.contains({repr(point)})= {P.contains(point)}")



# ELLIPSOID
print(f"\n\n=============== TESTING Ellipsoid =============== ")
P = Ellipsoid(Vector3(1,2,3), Vector3(0,0,0), Quaternion(1, Vector3(0,0,0)))
print(f"P= {repr(P)}")
print(P)
print(f"P.volume= {P.volume}")
print(f"P.bbox= {repr(P.bbox)}")

print("\nTESTING CENTER")
P.center = Vector3(1,2,3)
print(f"P.center = Vector3(1,2,3) -> P.bbox= {repr(P.bbox)}")
print(P)
P.center-=P.center
print(f"P.center-=P.center -> print(P)")
print(P)

print("\nTESTING RADII")
newR = Vector3(1,sqrt(2),3)
P.R = newR
print(f"P.R= {repr(newR)} -> P.bbox= {repr(P.bbox)}")
print(P)

print("\nTESTING QUATERNION")
newQ = Quaternion(1,Vector3(0,0,0)).setrotation(PI/2, Vector3(0,0,1))
P.Q = newQ
print(f"P.Q= {repr(newQ)} -> P.bbox= {repr(P.bbox)}")

print("\nprint(P):")
print(P)
print("\nprint(P.bbox):")
print(P.bbox)

print("\nTESTINGN CONTAINS")
point = P.facecenter(4)
print(f"P.contains({repr(point)})= {P.contains(point)}")
point += Vector3(0,0,-1e-5)
print(f"P.contains({repr(point)})= {P.contains(point)}")


# SUPERELLIPSOID
print(f"\n\n=============== TESTING SuperEllipsoid =============== ")
P = SuperEllipsoid(Vector3(1,2,3), 1, 1, Vector3(0,0,0), Quaternion(1, Vector3(0,0,0)))
print(f"P= {repr(P)}")
print(P)
print(f"P.volume= {P.volume}")
print(f"P.bbox= {repr(P.bbox)}")

print("\nTESTING CENTER")
P.center = Vector3(1,2,3)
print(f"P.center = Vector3(1,2,3) -> P.bbox= {repr(P.bbox)}")
print(P)

P.center-=P.center
print(f"P.center-=P.center -> print(P)")
print(P)

print("\nTESTING RADII")
newR = Vector3(1,sqrt(2),3)
P.R = newR
print(f"P.R= {repr(newR)} -> P.bbox= {repr(P.bbox)}")
print(P)

print("\nTESTING QUATERNION")
newQ = Quaternion(1,Vector3(0,0,0)).setrotation(PI/2, Vector3(0,0,1))
P.Q = newQ
print(f"P.Q= {repr(newQ)} -> P.bbox= {repr(P.bbox)}")

print("\nprint(P):")
print(P)
print("\nprint(P.bbox):")
print(P.bbox)

print("\nTESTING EPSILON")
P.eps1 = 0.5
print(f"P.eps1 = 0.5 -> P.volume= {P.volume}")
P.eps2/=5
print(f"P.eps2/=5 -> P.volume= {P.volume}")

print("\nTESTINGN CONTAINS")
point = P.facecenter(4)
print(f"P.contains({repr(point)})= {P.contains(point)}")
point += Vector3(0,0,-1e-5)
print(f"P.contains({repr(point)})= {P.contains(point)}")

n = int(1e6)
print(f"\nCREATING A LIST OF {n} SuperEllipsoid OBJECTS:")
tic = time()
SElist = []
for i in range(n):
	SElist.append(SuperEllipsoid(Vector3(1,2,3), 1, 1, Vector3(0,0,0), Quaternion(1, Vector3(0,0,0))))
toc = time()
print(f"TIME= {toc-tic} SECONDS, MEMORY= {getsizeof(SElist)/(2**20)} MB")