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
    
    N_side = 100
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
            "P_REFERENCE": PPMD.ParticleDat(2),
            "BASIS_EVAL": PPMD.ParticleDat(domain.ndim * (p+1)),
            "Q": PPMD.ParticleDat(1),
        },
        target_device
    )
    
    
    JFPI.uniform_grid_gaussian_weights(N_side, A)
    PPMD.write(PPMD.ParticleGroupVTK("particle_positions", A))





    dg_project_2d = JFPI.DGProject2D(A, p, 16, 16)
    




    


