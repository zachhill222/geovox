from geovox.util import NelderMead
from geovox.util import Box
from geovox.linalg import Vector

#define bounds
bounds = Box(Vector(-1,-1,-1), Vector(1,1,1))

#define objective function
Y = Vector(10,0,0)
F = lambda X: abs(X-Y)

#solve
neldermead = NelderMead(F, bounds, 0)
neldermead.penalty_weight = 100
neldermead.maxiter = 100
print(neldermead.minimize())