from firedrake import *

mesh = PeriodicSquareMesh(4, 4, 4.0, quadrilateral=True)
p = 1

# scalar version (works fine)
V = FunctionSpace(mesh, "DQ", p)
W = VectorFunctionSpace(mesh, V.ufl_element())
X = interpolate(mesh.coordinates, W)
eval_points = X.dat.data_ro.copy()

# this works
#f = Function(V)
#f.dat.data[:] = function_evals[:, 0]


# Want to "interpolate from external data" onto a vector function space
VV = VectorFunctionSpace(mesh, family='DQ', degree=p, dim=2)

# this errors ValueError: <class 'ufl.finiteelement.mixedelement.VectorElement'> modifier must be outermost
# Can I safely reuse the eval_points from the scalar case (assuming p is the same etc)
WV = VectorFunctionSpace(mesh, VV.ufl_element())


#fv = Function(VV)
#XV = interpolate(mesh.coordinates, WV)
#eval_points_V = XV.dat.data_ro.copy()

