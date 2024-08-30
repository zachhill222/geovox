from geovox.util import Box
from geovox.linalg import Vector


low1 = Vector(-1,-2,-3)
high1 = Vector(1,2,3)
box1 = Box(low1, high1)
print(f"box1= {repr(box1)}")
print(f"box1.sidelength= {repr(box1.sidelength)}")
print(f"box1.volume= {box1.volume}")

low2 = Vector(0,0,3)
high2 = Vector(3,3,0)
box2 = Box(low2, high2)
print(f"box2= Box({repr(low2)}), {repr(high2)}) -> box2= {repr(box2)}")

low3 = Vector(3,3,3)
high3 = Vector(6,6,6)
box3 = Box(low3, high3)
print("box3=",repr(box3))

print("\nTESTING INTERSECTION")
print("box1.intersects(box2)=", box1.intersects(box2))
print("box2.intersects(box1)=", box2.intersects(box1))

print("box2.intersects(box3)=", box2.intersects(box3))
print("box3.intersects(box2)=", box3.intersects(box2))

print("box3.intersects(box1)=", box3.intersects(box1))
print("box1.intersects(box3)=", box1.intersects(box3))


print("\nTESTING CONTAINS AND VERTEX: PATTERN SHOULD BE True, False NEAR EACH VERTEX")
box = Box(Vector(-2,-2,-2), Vector(2,2,2))
print(f"box= {repr(box)}")
print(f"box.center= {box.center}")
for n in range(8):
	print(f"box.vertex({n})=",repr(box.vertex(n)))
	x = box.center + 0.9*(box.vertex(n)-box.center)
	y = box.center + 1.1*(box.vertex(n)-box.center)

	print(f"\tbox.contains({x})={box.contains(x)}")
	print(f"\tbox.contains({y})={box.contains(y)}")


print("\nTESTING ASSINGMENT OF low: (FIX high, ASSIGN NEW OPPOSITE VERTEX)")
box = Box(Vector(1,1,1), Vector(2,2,2))
print(f"box= {repr(box)}")

newlow = Vector(0,0,0)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")

newlow = Vector(0,1,3)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")

newlow = Vector(3,3,3)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")


print("\nTESTING ASSINGMENT OF high: (FIX low, ASSIGN NEW OPPOSITE VERTEX)")
box = Box(Vector(1,1,1), Vector(2,2,2))
print(f"box= {repr(box)}")

newhigh = Vector(3,3,3)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")

newhigh = Vector(0,4,-1)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")

newhigh = Vector(0,3,3)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")


print("\nTESTING SHIFTING BY A VECTOR:")
box = Box(Vector(0,1,2), Vector(3,4,5))
print(f"box= {repr(box)}")
shift = Vector(1,2,-3)
print(f"shift= {repr(shift)}")

print(f"box+shift= {repr(box+shift)}")
print(f"box-shift= {repr(box-shift)}")

box+=shift
print(f"box+=shift -> box= {repr(box)}")
box-=shift
print(f"box-=shift -> box= {repr(box)}")


print("\nTESTING FACE AND FACENORMAL:")
box = Box(Vector(0,0,0), Vector(1,1,1))
print(f"box= {repr(box)}")
print(f"box.sidelength= {repr(box.sidelength)}")

for n in range(6):
	print(f"box.facenormal(n)= {repr(box.facenormal(n))},\t box.facecenter(n)= {repr(box.facecenter(n))}")


print("\nTESTING VERTICES:")
box = Box(Vector(0,0,0), Vector(1,1,1))
print(f"box= {repr(box)}")
for n in range(8):
	print(f"box.vertex(n)= {repr(box.vertex(n))}")


print("\nTESTING SCALING:")
box = Box(Vector(1,2,3), Vector(4,5,6))
print(f"box= {repr(box)}")
box = 1.1*box
print(f"box= 1.1*box -> box= {repr(box)}")
print(f"box.center = {repr(box.center)}")