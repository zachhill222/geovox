from geovox.utilities import NelderMead
from geovox.utilities import Box, Vector3

#define bounds
bounds = Box(Vector3(-1,-1,-1), Vector3(1,1,1))

#define objective function
Y = Vector3(10,0,0)
F = lambda X: abs(X-Y)

#solve
neldermead = NelderMead(F, bounds, 0)
neldermead.penalty_weight = 100
neldermead.maxiter = 100
print(neldermead.minimize())