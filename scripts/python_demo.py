#import julia
#julia.install()

from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI


if __name__ == "__main__":

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    N_side = 2000
    p = 1
    extent = 1.0
    nx = 16

    extents = (extent, extent)
    boundary_condition = PPMD.FullyPeroidicBoundary()
    domain = PPMD.StructuredCartesianDomain(boundary_condition, extents)

    rank = MPI.Comm_rank(domain.comm)
    assert rank == pyMPI.COMM_WORLD.rank

    A = PPMD.ParticleGroup(
        domain,
        {
            "P": PPMD.ParticleDat(2, position=True),
            "Q": PPMD.ParticleDat(1),
        },
        target_device
    )
    
    
    JFPI.uniform_grid_gaussian_weights(N_side, A)
    PPMD.write(PPMD.ParticleGroupVTK("particle_positions", A))


    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)
 

    mesh = PeriodicSquareMesh(nx, nx, extent, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", p)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    eval_points = X.dat.data_ro.copy()
    
    JFPI.set_eval_positions(dg_project_2d, eval_points)
    

    JFPI.project_evaluate(dg_project_2d, "Q")

    #for stepx in range(100):
    #    JFPI.project_evaluate(dg_project_2d)
    #Nrun = 1000
    #import time
    #PPMD.reset_profile()
    #t0 = time.time()
    #for stepx in range(Nrun):
    #    JFPI.project_evaluate(dg_project_2d)
    #if rank == 0:
    #    print((time.time() - t0) / Nrun)
    #    PPMD.print_profile()


    PPMD.write(PPMD.ParticleGroupVTK("function_evals", dg_project_2d.particle_group_eval))

    
    function_evals = JFPI.get_function_evaluations(dg_project_2d, "Q")
    f = Function(V)
    f.dat.data[:] = function_evals[:, 0]

    outfile = File("firedrake_output.pvd")
    outfile.write(f)


    g = Function(V)
    x, y = SpatialCoordinate(mesh)
    g.interpolate((2.0 / sqrt(pi)) * exp(-(2.0 * ((x - 0.5)**2 + (y - 0.5)**2))))
    outfile = File("firedrake_correct.pvd")
    outfile.write(g)


    VV = VectorFunctionSpace(mesh, family='DQ', degree=p, dim=2)
    fv = Function(VV)

    #WV = VectorFunctionSpace(mesh, VV.ufl_element())
    #XV = interpolate(mesh.coordinates, WV)
    #eval_points = XV.dat.data_ro.copy()
    #import pdb; pdb.set_trace()


    JFPI.project_evaluate(dg_project_2d, "P")
    function_evals = JFPI.get_function_evaluations(dg_project_2d, "P")
    fv.dat.data[:] = function_evals[:]

    outfile = File("firedrake_output_2.pvd")
    outfile.write(fv)









