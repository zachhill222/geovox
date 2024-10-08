from geovox.util import Quaternion
from geovox.linalg import Vector
from numpy import sin, cos
from numpy import pi as PI


U = Quaternion(1, Vector(2,3,4))
Q = Quaternion(2, Vector(-2,3,-4))

print(f"U= {repr(U)}")
print(f"Q= {repr(Q)}")

print("\nADDITION:")
print(f"U+Q= {repr(U+Q)}")
print(f"Q+U= {repr(Q+U)}")

print("\nSUBTRACTION:")
print(f"U-Q= {repr(U-Q)}")
print(f"Q-U= {repr(Q-U)}")

print("\nMULTIPLICATION")
print(f"U*Q= {repr(U*Q)}")
print(f"Q*U= {repr(Q*U)}")

print("\nSCALAR MULTIPLICATION:")
print(f"-2*U= {repr(-2*U)}")

print("\nINVERSE:")
print(f"U.inv()= {repr(U.inv())}")
print(f"U*U.inv()= {repr(U*U.inv())}")
print(f"U*Q.inv()= {repr(U*Q.inv())}")
print(f"U/Q = {repr(U/Q)}")

print("\nNEGATION:")
print(f"-U= {repr(-U)}")

print("\nSIZE:")
print(f"U.abs2= {U.abs2}")
print(f"abs(U)= {abs(U)}")
U.normalize()
print(f"U.normalize()-> U= {repr(U)}")
print(f"U.abs2= {U.abs2}")

print("\nASSIGNMENT:")
q0 = 3
qv = Vector(1,2,3)

print(f"Q= {repr(Q)}")
Q.q0 = q0
print(f"Q.q0= {q0} -> Q= {repr(Q)}")
Q.qv = qv
print(f"Q.qv= {qv} -> Q= {repr(Q)}")


print("\nROTATION")
theta = PI/4
axis  = Vector(0,0,3)
point = Vector(1,0,1)
Q.setrotation(theta, axis)
print(f"Q.setrotation({theta}, {repr(axis)})= {repr(Q)}")
print(f"Q.angle= {Q.angle}= PI/{PI/Q.angle}")
print(f"Q.axis= {repr(Q.axis)}")
print(f"point= {repr(point)}")
print(f"Q.rotate(point)= {repr(Q.rotate(point))}")

