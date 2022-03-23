module JFPI

using PPMD
using MPI

export DGProject2D
export set_eval_positions
export project
export evaluate
export project_evaluate
export free


"""
Generate kernel code to eval Legendre polynomials using

(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - n P_{n-1}(x)
P_0 = 1.0
P_1 = x

Stability issues for high order, e.g. 30+?
"""
function legendre_eval(p, x, outputs)
    
    eval_code = """
    x = $x
    P0 = 1.0;
    P1 = x;
    """
    for px in 1:p-1
        eval_code *= "P$(px+1) = ($(2.0*px + 1.0) * x * P$px - $px * P$(px-1)) * ($(1.0 / (px + 1.0)))\n"
    end

    for (ix, px) in enumerate(0:p)
        eval_code *= "$(outputs[ix]) = P$px\n"
    end

    return eval_code
end


"""
Create a PPMD Kernel to evaluate the legendre polynomials at the reference positions.
"""
function get_legendre_product_eval_kernel(p, ndim, symbol_positions, symbol_evals) 
    offset = 1
    ndof = p+1
    eval_code = ""
    for dimx in 1:ndim
        eval_code *= legendre_eval(
            p, "$(symbol_positions)[ix, $dimx]", ["$(symbol_evals)[ix, $dx]" for dx in offset:offset+ndof-1]
        )
        offset += ndof
    end
    return eval_code
end


"""
Compute the volume transformation from cell to reference cell
"""
function get_to_reference_scaling(mesh)
    ndim = mesh.domain.ndim
    volume = 1.0
    for dx in 1:ndim
        volume *= mesh.cell_widths[dx]
    end
    return (2.0 ^ ndim) / volume
end


"""
Create a uniform 2D grid
"""
function uniform_2d(N_side, E)
    eps = 1E-15

    positions_1d = LinRange(0.0 + eps, E - eps, N_side)
    positions = zeros(Float64, (N_side*N_side, 2))
    index = 1

    for iy in 1:N_side
        for ix in 1:N_side
            positions[index, 1] = positions_1d[ix]
            positions[index, 2] = positions_1d[iy]
            index += 1
        end
    end
    
    weights = E * E / (N_side * N_side)
    return positions, weights
end

mutable struct DGProject2D
    
    p::Int64
    particle_group_source::ParticleGroup
    particle_group_eval::ParticleGroup
    nx::Int64
    ny::Int64
    
    mesh
    reference_position_loop_A
    reference_position_loop_E
    basis_eval_loop_A
    basis_eval_loop_E

    cell_bin_loop_A
    cell_bin_loop_E
    cell_origins
    cell_widths
    cell_dofs
    cell_basis_indices
    cell_basis_invmass
    assemble_pairloop
    eval_pairloop


    function DGProject2D(particle_group_source::ParticleGroup, p::Int64, nx::Int64, ny::Int64)
        
        target_device = particle_group_source.compute_target
        domain = particle_group_source.domain
        particle_group_eval = ParticleGroup(
            domain,
            Dict(
                 "P" => ParticleDat(domain.ndim, position=true),
                 "P_REFERENCE" => ParticleDat(domain.ndim),
                 "BASIS_EVAL" => ParticleDat(domain.ndim * (p+1)),
                 "Q" => ParticleDat(1),
            ),
            target_device
        )
        
        # TODO remove square assumption
        @assert nx == ny
        extent = domain.extent[1]
        mesh = MinimalWidthCartesianMesh(domain, extent / nx)
        
        A = particle_group_source
        E = particle_group_eval

        # get the loop that bins particles into cells
        cell_bin_loop_A, cell_bin_loop_E = [get_map_particle_to_cell_task(mesh, PG) for PG in [A, E]]
        # get cell offsets and widths
        cell_origins, cell_widths = get_cell_origins_widths(mesh, target_device)

        # loop over particles and compute the reference positions
        reference_position_loop_A, reference_position_loop_E = [PairLoop(
            target_device,
            Kernel(
                "compute_reference_positions",
                """
                for dimx in 1:$(domain.ndim)
                    origin = ORIGINS[dimx]
                    width = WIDTHS[dimx]
                    shifted_position = P[ix, dimx] - origin
                    # map onto [-1, 1]
                    scaled_position = 2.0 * (shifted_position / width) - 1.0
                    P_REFERENCE[ix, dimx] = scaled_position
                end
                """
            ),
            Dict(
                "ORIGINS" => (cell_origins, READ),
                "WIDTHS" => (cell_widths, READ),
            ),
            Dict(
                "P" => (PG["P"], READ),
                "P_REFERENCE" => (PG["P_REFERENCE"], READ),
            )
           ) for PG in [A, E]]

        # evaluate the basis functions along each dimension
        basis_eval_loop_A, basis_eval_loop_E = [ParticleLoop(
            target_device,
            Kernel(
                "basis_func_eval",
                get_legendre_product_eval_kernel(p, domain.ndim, "P_REFERENCE", "BASIS_EVAL")
            ),
            Dict(
                 "P_REFERENCE" => (PG["P_REFERENCE"], READ),
                 "BASIS_EVAL" => (PG["BASIS_EVAL"], WRITE),
            )
           ) for PG in [A, E]]

        # space for dofs in each cell
        cell_dofs = CellDat(mesh, ((p+1) ^ domain.ndim, 1), Float64, target_device)
        # indices to combine for each dof
        cell_basis_indices = CellDat(mesh, ((p+1) ^ domain.ndim, domain.ndim), Int32, target_device)
        # inverse mass matrix diagonal
        cell_basis_invmass = CellDat(mesh, ((p+1) ^ domain.ndim, 1), Float64, target_device)
        # populate cell dats
        jacobian_volume = get_to_reference_scaling(mesh)
        for cellx in 1:mesh.cell_count
            for (dofx, indx) in enumerate(Iterators.product([1:p+1 for dx in 1:domain.ndim]...))
                # 2 / (2n+1)
                imass = 1.0
                for dimx in 1:domain.ndim
                    cell_basis_indices[cellx, dofx, dimx] = indx[dimx] + (dimx-1) * (p + 1)
                    imass *= (2.0 * (indx[dimx] - 1.0) + 1.0) / 2.0
                end
                cell_basis_invmass[cellx, dofx, 1] = imass * jacobian_volume
            end
        end

        # pairloop to assemble dofs
        assemble_pairloop = PairLoop(
            target_device,
            Kernel(
                "assemble_pairloop",
                """
                contribution = Q[ix, 1]
                for dimx in 1:$(domain.ndim)
                    contribution *= BASIS_EVAL[ix, INDICES[dimx]]
                end
                DOFS[1] += contribution * IMASS[1]
                """
            ),
            Dict(
                 "INDICES" => (cell_basis_indices, READ),
                 "IMASS" => (cell_basis_invmass, READ),
                 "DOFS" => (cell_dofs, INC_ZERO),
            ),
            Dict(
                "BASIS_EVAL" => (A["BASIS_EVAL"], READ),
                "Q" => (A["Q"], READ),
            ),
        )


        # pairloop to evaluate field at points in E
        eval_pairloop = PairLoop(
            target_device,
            Kernel(
                "eval_pairloop",
                """
                contribution = DOFS[1]
                for dimx in 1:$(domain.ndim)
                    contribution *= BASIS_EVAL[ix, INDICES[dimx]]
                end
                Q[ix, 1] += contribution
                """
            ),
            Dict(
                 "INDICES" => (cell_basis_indices, READ),
                 "IMASS" => (cell_basis_invmass, READ),
                 "DOFS" => (cell_dofs, READ),
            ),
            Dict(
                "BASIS_EVAL" => (E["BASIS_EVAL"], READ),
                "Q" => (E["Q"], INC_ZERO),
            ),
        )

        n = new(
            p, 
            particle_group_source, 
            particle_group_eval, 
            nx, ny, 
            mesh, 
            reference_position_loop_A,
            reference_position_loop_E,
            basis_eval_loop_A,
            basis_eval_loop_E,
            cell_bin_loop_A,
            cell_bin_loop_E,
            cell_origins,
            cell_widths,
            cell_dofs,
            cell_basis_indices,
            cell_basis_invmass,
            assemble_pairloop,
            eval_pairloop,
        )
        
        return n
    end
    
end


"""
Set the locations of the dofs in the cell
"""
function set_eval_positions(dgp, positions=nothing, cells=nothing)

    @assert dgp.particle_group_eval.npart_local == 0
    
    data = Dict()
    if !isnothing(cells)
        data[cellid_particle_dat(dgp.particle_group_eval, dgp.mesh)] = cells
    end

    if !isnothing(positions)
        data["P"] = positions
        add_particles(dgp.particle_group_eval, data)
    else
        add_particles(dgp.particle_group_eval)
    end

    # eval basis functions for eval positions
    # bin particles into cells
    if isnothing(cells)
        execute(dgp.cell_bin_loop_E)
    end

    # map to refence positions
    execute(dgp.reference_position_loop_E)

    # eval basis functions at particle locations
    execute(dgp.basis_eval_loop_E)

end


"""
Project the particle information onto the Legendre DOFs
"""
function project(dgp)
    # bin particles into cells
    execute(dgp.cell_bin_loop_A)

    # map to refence positions
    execute(dgp.reference_position_loop_A)

    # eval basis functions at particle locations
    execute(dgp.basis_eval_loop_A)
    
    # assemble dofs
    execute(dgp.assemble_pairloop)
end


"""
Evaluate the projected field at the evaluation positions.
"""
function evaluate(dgp)
    execute(dgp.eval_pairloop)
end


"""
Project particle data and evaluate at the required positions.
"""
function project_evaluate(dgp)
    project(dgp)
    evaluate(dgp)
end


"""
Free a DGProject2D struct
"""
function free(dgp::DGProject2D)
    PPMD.free(dgp.particle_group_eval)
end


"""
Utility function to create a uniform grid of particles with gaussian weight.
"""
function uniform_grid_gaussian_weights(N_side, A)
    rank = MPI.Comm_rank(A.domain.comm)
    extent = A.domain.extent
    target_device = A.compute_target
    if rank == 0
        positions, weights = uniform_2d(N_side, extent[1])
        add_particles(
            A,
            Dict(
                 "P" => positions,
                 "Q" => weights * ones(Float64, (N_side * N_side, 1))
            )
        )
    else
        add_particles(A)
    end

    # compute the correct values at the eval points
    assign_loop = ParticleLoop(
        target_device,
        Kernel(
            "assign_func_eval",
            """
            x = P[ix, 1]
            y = P[ix, 2]
            Q[ix, 1] = (2.0 / sqrt(pi)) * exp(-(2.0 * ((x - 0.5)^2 + (y - 0.5)^2))) * Q[ix, 1]
            """
        ),
        Dict(
             "P" => (A["P"], READ),
             "Q" => (A["Q"], WRITE),
        )
    )
    execute(assign_loop)
end


"""
Utility interface to get the evaluations back in Python.
"""
function get_function_evaluations(dgp::DGProject2D)
    return dgp.particle_group_eval["Q"][1:dgp.particle_group_eval.npart_local, :]
end


end # module
