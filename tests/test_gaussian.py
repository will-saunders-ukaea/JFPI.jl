
from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI

import numpy as np
import chaospy

def test_gaussian_halton():

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    p = 1
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
    correct = interpolate((2.0 / sqrt(pi)) * exp(-(2.0 * ((x - 0.5)**2 + (y - 0.5)**2))), V)

    V_CG = FunctionSpace(mesh, "CG", p)
    correct_cg = interpolate((2.0 / sqrt(pi)) * exp(-(2.0 * ((x - 0.5)**2 + (y - 0.5)**2))), V_CG)
    
    list_npart = []
    list_errors = []
    #for N_side in np.logspace(2, 3.6, 10):

    A = PPMD.ParticleGroup(
        domain,
        {
            "P": PPMD.ParticleDat(2, position=True),
            "Q": PPMD.ParticleDat(1),
        },
        target_device
    )

    # create projection object
    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)
    JFPI.set_eval_positions(dg_project_2d, eval_points)

    for N_side in (7,8,9,10):
        N_side = int(2**N_side)
        N_total = N_side * N_side

        #JFPI.uniform_grid_gaussian_weights(N_side, A)
        PPMD.remove_particles(A, [px+1 for px in range(A.npart_local)])
        assert A.npart_local == 0
        
        positions = chaospy.create_halton_samples(N_total, 2).transpose() * extent
        quantity = (2.0 / np.sqrt(np.pi)) * np.exp(-(2.0 * ((positions[:, 0] - 0.5)**2 + (positions[:, 1] - 0.5)**2))).reshape((N_total, 1))

        reweight = extent * extent / (N_total)
        
        if rank == 0:
            PPMD.add_particles(
                A,
                {
                    "P": positions,
                    "Q": quantity * reweight,
                }
            )
        else:
            PPMD.add_particles(A)

        JFPI.project_evaluate(dg_project_2d, "Q")

        # do projection and get result    
        function_evals = JFPI.get_function_evaluations(dg_project_2d, "Q")
        f = Function(V)
        f.dat.data[:] = function_evals[:, 0]

        # compute error

        err = errornorm(correct, f)
        #print(N_side, err)

        list_npart.append(N_total)
        list_errors.append(err)

    lin_ab = np.polyfit(np.log10(list_npart), np.log10(list_errors), 1)

    #print("Gradient:", lin_ab[0])

    assert lin_ab[0] < -0.9



