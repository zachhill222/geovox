from geovox.geometry import Assembly
from geovox.util import Box
from geovox.linalg import Vector

from time import time
from sys import getsizeof

Assem = Assembly()
Assem.read("./tests/sample_particles/particles_1000.txt")
Assem.maxdepth = 8


for n in range(Assem.maxdepth):
	tic = time()
	Assem.cdivide()

	Assem.to_vtk(f"./tests/outfiles/assemblytest_nparticles_{len(Assem.particle_list)}_depth_{Assem.totaldepth}.vtk")
	print(f"depth= {Assem.totaldepth}: {len(Assem.leaflist())} leaves, {len(Assem.points)} vertices, {time()-tic} seconds, {getsizeof(Assem)/(2**10)} KB")

#passing P to NelderMead
# depth= 1: 8 leaves, 27 vertices, 0.10639023780822754 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.24292564392089844 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.7149744033813477 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 12.211674928665161 seconds, 0.125 KB
# depth= 5: 32054 leaves, 35317 vertices, 28.911300659179688 seconds, 0.125 KB
# depth= 6: 203253 leaves, 224952 vertices, 87.02915501594543 seconds, 0.125 KB
# depth= 7: 1119357 leaves, 1344767 vertices, 267.54829931259155 seconds, 0.125 KB

# passing P.levelval (Lambda property) to NelderMead
# depth= 1: 8 leaves, 27 vertices, 0.10834956169128418 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.24503064155578613 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.6347551345825195 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 7.258740425109863 seconds, 0.125 KB
# depth= 5: 32033 leaves, 35694 vertices, 25.097888231277466 seconds, 0.125 KB
# depth= 6: 156003 leaves, 210905 vertices, 54.882346630096436 seconds, 0.125 KB
# depth= 7: 569066 leaves, 829391 vertices, 118.67314267158508 seconds, 0.125 KB


# passing P.levelval (Lambda property) to NelderMead, returning double, not Vector
# depth= 1: 8 leaves, 27 vertices, 0.10469698905944824 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.23701047897338867 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.631826639175415 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 7.683609485626221 seconds, 0.125 KB
# depth= 5: 32026 leaves, 35690 vertices, 23.520033836364746 seconds, 0.125 KB
# depth= 6: 155933 leaves, 210834 vertices, 55.90721321105957 seconds, 0.125 KB
# depth= 7: 568996 leaves, 829320 vertices, 117.0857765674591 seconds, 0.125 KB


# passing P.levelval (cdef method) to NelderMead, returning double, not vector
# depth= 1: 8 leaves, 27 vertices, 0.09791278839111328 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.2259693145751953 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.6324295997619629 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 9.469032764434814 seconds, 0.125 KB
# depth= 5: 32488 leaves, 35782 vertices, 27.846132040023804 seconds, 0.125 KB
# depth= 6: 194454 leaves, 234788 vertices, 88.97668957710266 seconds, 0.125 KB


# changed python "for child in ... child.divide()" loop into C "for child_ind in range(8)..." loop
# depth= 1: 8 leaves, 27 vertices, 0.09799623489379883 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.22305822372436523 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.6316099166870117 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 9.557361364364624 seconds, 0.125 KB
# depth= 5: 32488 leaves, 35782 vertices, 27.264958381652832 seconds, 0.125 KB
# depth= 6: 194454 leaves, 234788 vertices, 89.96332311630249 seconds, 0.125 KB


#changed back to passing lambda fucntion to NM. changed arguments to reduce the number of double divisions
# depth= 1: 8 leaves, 27 vertices, 0.10902214050292969 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.24796438217163086 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.7346112728118896 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 8.299880027770996 seconds, 0.125 KB
# depth= 5: 32026 leaves, 35690 vertices, 24.75150990486145 seconds, 0.125 KB
# depth= 6: 155933 leaves, 210834 vertices, 58.74738097190857 seconds, 0.125 KB


# removed penalty function from NM. terminate if simplex is outside of box.
# depth= 1: 8 leaves, 27 vertices, 0.1045541763305664 seconds, 0.125 KB
# depth= 2: 64 leaves, 125 vertices, 0.242218017578125 seconds, 0.125 KB
# depth= 3: 512 leaves, 729 vertices, 0.5815310478210449 seconds, 0.125 KB
# depth= 4: 4096 leaves, 4913 vertices, 3.362217426300049 seconds, 0.125 KB
# depth= 5: 32215 leaves, 35834 vertices, 14.997673749923706 seconds, 0.125 KB
# depth= 6: 175078 leaves, 227384 vertices, 55.54987645149231 seconds, 0.125 KB















