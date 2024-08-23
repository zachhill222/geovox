from geovox.utilities import Vector3
from time import time
from sys import getsizeof
from numpy import pi as PI

U = Vector3(1,2,-3)
u = Vector3(U.x, U.y, U.z)
print("u= "+repr(u))

V = Vector3(3,-5,-1)
v = Vector3(V.x, V.y, V.z)
print("v= "+repr(v))

print("\nTESTING VECTOR OPERATIONS")
print("|u|^2=", u.abs2)
print("|u|=", abs(u))
print("-u=", -u)
print("2*u=",2*u)

print("\nNEGATION:")
print(f"-u= {repr(-u)}")

print("\nADDITION")
print("u+v=", u+v)
print("v+u=", v+u)

print("\nSUBTRACTION")
print("u-v=", u-v)
print("v-u=", v-u)

print("\nDOT PRODUCT")
print("u.dot(v)=", u.dot(v))
print("v.dot(u)=", v.dot(u))

print("\nCROSS PRODUCT")
print("uXv=",u.cross(v))
print("vXu=",v.cross(u))

print("\nCOMPONENT-WISE PRODUCT")
print("u*v=", u*v)
print("v*u=", v*u)

print("\nCOMPONENT-WISE DIVISION")
print("u/v=", u/v)
print("v/u=", v/u)

print("\nTESTING IN-PLACE VECTOR OPERATIONS")
u*=v
print("u*=v: u <-", u)
u=Vector3(U.x, U.y, U.z)

u/=v
print("u/=v: u <-", u)
u=Vector3(U.x, U.y, U.z)

print("\nTESTING COMPARISONS (POS/NEG CONES, AKA COMPONENT-WISE)")
O = Vector3(0,0,0)
print("O="+repr(O))
x = Vector3(1,0,0)
print("O<"+repr(x)+":", O<x)
print("O<="+repr(x)+":", O<=x)
print("O>"+repr(x)+":", O>x)
print("O>="+repr(x)+":", O>=x)
print("O=="+repr(x)+":", O==x)
print("O!="+repr(x)+":", O!=x)
print("O==O:", O==O)
print("O!=O:", O!=O)


print("\nTESTING @property SET AND GET")
print("u=",repr(u))
u.x+=10
print("u.x+=10 -> u=",repr(u))
print("u.x=",u.x)
u.y-=10
print("u.y+=10 -> u=",repr(u))
print("u.y=",u.y)
u.z=u.x-u.y
print("u.z=u.x-u.y -> u=",repr(u))
print("u.z=",u.z)


n = int(64e6)
print(f"\nCREATING A LIST OF {n} Vector3 OBJECTS:")
tic = time()
vectorlist = []
for i in range(n):
	vectorlist.append(Vector3(PI,PI,PI))
toc = time()
print(f"TIME= {toc-tic} SECONDS, MEMORY= {getsizeof(vectorlist)/(2**20)} MB")
