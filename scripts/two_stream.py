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
    
    N_side = 1000
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
            "V": PPMD.ParticleDat(2),
        },
        target_device
    )

    positions, _ = JFPI.uniform_2d(N_side, extent)
    charges = 0.5 * (extent * extent / (N_side * N_side)) * np.ones((N_side * N_side, 1))
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

    velocities[:, 0] = -1.0

    PPMD.add_particles(
        A,
        {
            "P": positions,
            "V": velocities,
            "Q": charges,
        }
    )

    
    PPMD.write(PPMD.ParticleGroupVTK("particle_positions", A))


    dg_project_2d = JFPI.DGProject2D(A, p, nx, nx)
 

    mesh = PeriodicSquareMesh(nx, nx, extent, quadrilateral=True)
    V = FunctionSpace(mesh, "DQ", p)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = interpolate(mesh.coordinates, W)
    eval_points = X.dat.data_ro.copy()
    
    JFPI.set_eval_positions(dg_project_2d, eval_points)
    

    JFPI.project_evaluate(dg_project_2d, "Q")

    
    function_evals = JFPI.get_function_evaluations(dg_project_2d, "Q")
    charge_density = Function(V)
    charge_density.dat.data[:] = function_evals[:, 0]

    outfile = File("charge_density_dg.pvd")
    outfile.write(charge_density)


    # V_solve = FunctionSpace(mesh, "CG", p)
    # u = TrialFunction(V_solve)
    # v = TestFunction(V_solve)
    # a = inner(grad(u), grad(v)) * dx


    # x,y = SpatialCoordinate(mesh)
    # F = project(charge_density, V_solve) - Constant(1.0)
    # # F = Constant(0.0)
    # L = F*v*dx

    # potential = Function(V_solve)
    # 
    # sp = {
    #     "snes_type": "newtonls",
    #     "snes_monitor": None,
    #     "ksp_type": "preonly",
    #     "pc_type": "lu",
    #     "pc_factor_mat_solver_type": "mumps",
    # }

    #     
    # sp = {"snes_monitor": None,}

    # #sp = {"ksp_type": "preonly", "pc_type": "lu"}

    # solve(a == L, potential, solver_parameters=sp)

    # outfile = File("phi.pvd")
    # outfile.write(potential)

    BDM = FunctionSpace(mesh, "RTCF", p+1)
    CG = FunctionSpace(mesh, "CG", p)
    W = BDM * CG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    
    f_particle = project(charge_density, CG)
    f = f_particle - Constant(1.0)
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = f*v*dx
    
    w = Function(W)
    solve(a == L, w)
    electric_field, electric_potential = w.split()
    
    File("charge_density_cg.pvd").write(f_particle)
    File("electric_field.pvd").write(electric_field)
    File("electric_potential.pvd").write(electric_potential)





