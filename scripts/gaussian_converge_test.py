
from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI


import numpy as np
import chaospy

import tqdm.contrib.itertools
import json


def main(nx, p, N_side):

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    #N_side = 2000
    #p = 1
    extent = 1.0
    #nx = 16

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

    
    N_total = N_side * N_side
    # create uniform grid of particles with gaussian weights
    A = PPMD.ParticleGroup(
        domain,
        {
            "P": PPMD.ParticleDat(2, position=True),
            "Q": PPMD.ParticleDat(1),
        },
        target_device
    )

    #JFPI.uniform_grid_gaussian_weights(N_side, A)
    
    positions = chaospy.create_halton_samples(N_total, 2).transpose() * extent
    #distribution = chaospy.J(chaospy.Uniform(0, extent), chaospy.Uniform(0, extent))
    #positions = samples = distribution.sample(N_total, rule="halton").reshape((N_total, 2))
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



    # create projection object
    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)
    JFPI.set_eval_positions(dg_project_2d, eval_points)
    JFPI.project_evaluate(dg_project_2d, "Q")


    # do projection and get result    
    function_evals = JFPI.get_function_evaluations(dg_project_2d, "Q")
    f = Function(V)
    f.dat.data[:] = function_evals[:, 0]

    # compute error

    err = errornorm(correct, f)
    
    f_cg = project(f, V_CG)
    
    err_cg = errornorm(correct_cg, f_cg)

    #print(N_side, err, err_cg)

    o = {   
        "N_cells": nx,
        "N_particles": N_total,
        "errornorm": err,
        "errornorm_cg": err_cg,
        "p": p
    }

    with open(f"output.json", "w") as fh:
        fh.write(json.dumps(o, indent=2))


if __name__ == "__main__":

        
    config = json.loads(open("input.json").read())
    nx = int(config["N_cells"])
    p = int(config["p"])
    N_side = int(2 ** int(config["N_particles"]))

    main(nx, p, N_side)











