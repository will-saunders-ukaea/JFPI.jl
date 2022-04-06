
from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI

import numpy as np
import chaospy

def test_gaussian_eval_1():

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    p = 2
    extent = 1.0
    nx = 8

    extents = (extent, extent)
    boundary_condition = PPMD.FullyPeroidicBoundary()
    domain = PPMD.StructuredCartesianDomain(boundary_condition, extents)

    rank = MPI.Comm_rank(domain.comm)
    assert rank == pyMPI.COMM_WORLD.rank
 
    # create mesh and function space on which to project
    mesh = PeriodicSquareMesh(nx, nx, extent, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", p)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    eval_points = X.dat.data_ro.copy()

    # correct field
    x, y = SpatialCoordinate(mesh)

    #correct = interpolate(abs(x - 0.5 * extent), V)
    #correct = interpolate(abs(y - 0.5 * extent), V)
    correct = interpolate((2.0 / sqrt(pi)) * exp(-(2.0 * ((x - 0.5)**2 + (y - 0.5)**2))), V)

    A = PPMD.ParticleGroup(
        domain,
        {
            "P": PPMD.ParticleDat(2, position=True),
            "Q": PPMD.ParticleDat(1),
        },
        target_device
    )

    positions = eval_points
        
    if rank == 0:
        PPMD.add_particles(
            A,
            {
                "P": positions,
            }
        )
    else:
        PPMD.add_particles(A)

    # create projection object
    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)

    JFPI.set_eval_positions(dg_project_2d, eval_points)
    JFPI.set_eval_values(dg_project_2d, correct.dat.data_ro.copy())   
    JFPI.field_evaluate(dg_project_2d, "Q")

    assert np.linalg.norm(PPMD.getindex(A, "P").data - eval_points, np.inf) < 1E-15
    assert np.linalg.norm(PPMD.getindex(dg_project_2d.particle_group_eval, "P").data - eval_points, np.inf) < 1E-15

    for ix in range(A.npart_local):
        assert (PPMD.getindex(A, "Q").data[ix,0] - correct.dat.data_ro[ix]) < 1E-14

    
    PPMD.remove_particles(A, [ix+1 for ix in range(A.npart_local)])

    N_side = 8
    N_side = int(2**N_side)
    N_total = N_side * N_side
    #positions = chaospy.create_halton_samples(N_total, 2).transpose() * extent
    positions = np.random.uniform(0.0, extent, (N_total, 2))
    if rank == 0:
        PPMD.add_particles(
            A,
            {
                "P": positions,
            }
        )
    else:
        PPMD.add_particles(A)

    JFPI.field_evaluate(dg_project_2d, "Q")
    
    P = PPMD.getindex(A, "P")
    Q = PPMD.getindex(A, "Q")
    P_REF = PPMD.getindex(A, "_P_REFERENCE")
    
    for ix in range(A.npart_local):
        pos = P.data[ix, :]
        val = Q.data[ix, 0]
        cor = correct.at(*pos)
        err = abs(val - cor) / abs(cor)
        assert err < 1E-10

        #if err > 1E-10:
        #    print(err, P_REF.data[ix,:])










