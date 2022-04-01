#import julia
#julia.install()

from firedrake import *

from julia import Julia
from julia import PPMD
from julia import JFPI
from julia import MPI

from mpi4py import MPI as pyMPI
import numpy as np


def positive_mod(x, n):
    return ((x % n) + n) % n


def test_advect_diagonal():

    target_device = PPMD.KACPU()
    #target_device = PPMD.KACUDADevice(128)
    
    N_side = 1000
    p = 0
    extent = 1.0
    nx = 8

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
            "V": PPMD.ParticleDat(2),
        },
        target_device
    )

    mesh = PeriodicSquareMesh(nx, nx, extent, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", p)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    eval_points = X.dat.data_ro.copy()
    f = Function(V)

    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)


    positions, _ = JFPI.uniform_2d(N_side, extent)
    charges = (extent * extent / (N_side * N_side)) * np.ones((N_side * N_side, 1))
    velocities = np.zeros((N_side * N_side, 2))
    velocities[:, 0] = 1.0

    PPMD.add_particles(
        A,
        {
            "P": positions,
            "V": velocities,
            "Q": charges,
        }
    )
    
    # bin particles into cells
    PPMD.execute(dg_project_2d.cell_bin_loop_A)

    # scale q by cell id (base 1 as Julia)
    cell_id_particle_dat = PPMD.cellid_particle_dat_name(dg_project_2d.mesh)
    PPMD.execute(
        PPMD.ParticleLoop(
            target_device,
            PPMD.Kernel(
                "rescale_kernel",
                """
                Q[ix, 1] *= CELL[ix, 1]
                """
            ),
            {
                "Q": (PPMD.getindex(A, "Q"), PPMD.WRITE),
                "CELL": (PPMD.getindex(A, cell_id_particle_dat), PPMD.READ),
            }
        )
    )


    
    #PPMD.write(PPMD.ParticleGroupVTK("particle_positions", A))

    
    JFPI.set_eval_positions(dg_project_2d, eval_points)
    
    
    def project():
        JFPI.project_evaluate(dg_project_2d, "Q")
        function_evals = JFPI.get_function_evaluations(dg_project_2d, "Q")
        f.dat.data[:] = function_evals[:, 0]

    project()

    #outfile = File("f.pvd")
    
    cell_width = extent / nx
    
    offset_x = 0
    offset_y = 0
    def check_errors():
        for cx in range(nx):
            for cy in range(nx):
                point = (
                    (cx + 0.5) * cell_width,
                    (cy + 0.5) * cell_width,
                )
                f_eval = f.at(point, tolerance=1E-8)
                
                correct = positive_mod(cx - offset_x, nx) + positive_mod((cy - offset_y), nx) * nx + 1.0 
                err = abs(f_eval - correct) / abs(correct)

                #print(cx, cy, correct, f_eval, err)
                assert err < 1E-12



    #outfile.write(f)
    check_errors()

    #advect particles by one cell
    advect_loop = PPMD.ParticleLoop(
        target_device,
        PPMD.Kernel(
            "advect_kernel",
            f"""
            P[ix, 1] += {cell_width}
            P[ix, 2] += {cell_width}
            """
        ),
        {
            "P": (PPMD.getindex(A, "P"), PPMD.WRITE),
        }
    )

    for stepx in range(nx * 8):
        PPMD.execute(advect_loop)
        PPMD.global_move(A)
        project()
        #outfile.write(f)

        offset_x += 1
        offset_y += 1
        check_errors()












