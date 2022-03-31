#import julia
#julia.install()

from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI
import numpy as np


if __name__ == "__main__":

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    N_side = 2000
    p = 1
    extent = 1.0
    nx = 32

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
            "J": PPMD.ParticleDat(2),
            "V": PPMD.ParticleDat(2),
        },
        target_device
    )
    
    
    #JFPI.uniform_grid_gaussian_weights(N_side, A)
    
    N_total = N_side ** 2
    positions = np.zeros((N_total, 2))
    positions[:, 0] = np.random.uniform(0.0, extent, (N_total,))
    positions[:, 1] = np.fmod(np.random.normal(0.0, 0.10, (N_total,)), extent) + 0.5 * extent
    assert N_total % 2 == 0
    N_half = N_total // 2
    velocities = np.zeros((N_total, 2))

    velocities[:N_half, 0] = 1.0
    velocities[N_half:, 0] = -1.0
    velocities[:,:] += np.random.uniform(0.0, 0.02, (N_total, 2))

    charges = np.ones((N_total, 1))
    charges[N_half:] *= -1.0
    
    PPMD.add_particles(
        A,
        {
            "P": positions,
            "V": velocities,
            "Q": charges,
        }
    )

    assemble_current = PPMD.ParticleLoop(
        target_device,
        PPMD.Kernel(
            "assemble_current",
            """
            J[ix, 1] = Q[ix, 1] * V[ix, 1]
            J[ix, 2] = Q[ix, 1] * V[ix, 2]
            """
        ),
        {
            "J": (PPMD.getindex(A, "J"), PPMD.WRITE),
            "Q": (PPMD.getindex(A, "Q"), PPMD.READ),
            "V": (PPMD.getindex(A, "V"), PPMD.READ),
        }
    )
    PPMD.execute(assemble_current)


    PPMD.write(PPMD.ParticleGroupVTK("particle_state", A))

    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)
 

    mesh = PeriodicSquareMesh(nx, nx, extent, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", p)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    eval_points = X.dat.data_ro.copy()
    VV = VectorFunctionSpace(mesh, family='DQ', degree=p, dim=2)
    fv = Function(VV)


    JFPI.set_eval_positions(dg_project_2d, eval_points)
    JFPI.project_evaluate(dg_project_2d, "J")
    function_evals = JFPI.get_function_evaluations(dg_project_2d, "J")
    fv.dat.data[:] = function_evals[:]

    PPMD.write(PPMD.ParticleGroupVTK("function_evals", dg_project_2d.particle_group_eval))
    outfile = File("current.pvd")
    outfile.write(fv)









