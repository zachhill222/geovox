from geovox.utility import Box, Vector3


low1 = Vector3(-1,-2,-3)
high1 = Vector3(1,2,3)
box1 = Box(low1, high1)
print("box1=",repr(box1))


low2 = Vector3(0,0,0)
high2 = Vector3(3,3,3)
box2 = Box(low2, high2)
print("box2=",repr(box2))

low3 = Vector3(3,3,3)
high3 = Vector3(6,6,6)
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
box = Box(Vector3(-2,-2,-2), Vector3(2,2,2))
print(f"box= {repr(box)}")
print(f"box.center= {box.center}")
for n in range(8):
	print(f"box.vertex({n})=",repr(box.vertex(n)))
	x = box.center + 0.9*(box.vertex(n)-box.center)
	y = box.center + 1.1*(box.vertex(n)-box.center)

	print(f"\tbox.contains({x})={box.contains(x)}")
	print(f"\tbox.contains({y})={box.contains(y)}")


print("\nTESTING ASSINGMENT OF low: (FIX high, ASSIGN NEW OPPOSITE VERTEX)")
box = Box(Vector3(1,1,1), Vector3(2,2,2))
print(f"box= {repr(box)}")

newlow = Vector3(0,0,0)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")

newlow = Vector3(0,1,3)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")

newlow = Vector3(3,3,3)
box.low = newlow
print(f"box.low= {repr(newlow)}: box= {repr(box)}")


print("\nTESTING ASSINGMENT OF high: (FIX low, ASSIGN NEW OPPOSITE VERTEX)")
box = Box(Vector3(1,1,1), Vector3(2,2,2))
print(f"box= {repr(box)}")

newhigh = Vector3(3,3,3)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")

newhigh = Vector3(0,4,-1)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")

newhigh = Vector3(0,3,3)
box.high = newhigh
print(f"box.high= {repr(newhigh)}: box= {repr(box)}")