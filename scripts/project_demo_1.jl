using PPMD
using JFPI

using SpecialFunctions


"""
[0, 1] to [-1, 1]
"""
function interval_to_ref(x)
    return (x - 0.5) * 2.0
end


"""
[-1, 1] to [0, 1]
"""
function ref_to_interval(x)
    return (x + 1.0) * 0.5
end

"""
Create a 2D gaussian on [0, E]^2
"""
function gaussian_2d(N_side, E)
    eps = 1E-15
    density_left = erf(interval_to_ref(0.0 + eps))
    density_right = erf(interval_to_ref(1.0 - eps))
    function_yvals = LinRange(density_left, density_right, N_side)
    positions_1d = [ref_to_interval(erfinv(fx)) * E for fx in function_yvals]
    
    positions = zeros(Float64, (N_side*N_side, 2))
    index = 1

    for iy in 1:N_side
        for ix in 1:N_side
            positions[index, 1] = positions_1d[ix]
            positions[index, 2] = positions_1d[iy]
            index += 1
        end
    end
    
    weight = (density_right - density_left) / (N_side * 2.0)
    weight = weight ^ 2

    return positions, weight
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


function main()
    
    target_device = KACPU()
    #target_device = KACUDADevice(128)
    
    N_side = 100
    N_sample_side = 50
    p = 1
    extent = 1.0

    extents = (extent, extent)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)
    rank = MPI.Comm_rank(domain.comm)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(2, position=true),
             "P_REFERENCE" => ParticleDat(2),
             "BASIS_EVAL" => ParticleDat(domain.ndim * (p+1)),
             "Q" => ParticleDat(1),
             "QC" => ParticleDat(1),
        ),
        target_device
    )
    
    dg_project_2d = DGProject2D(A, p, 16, 16)


    reset_profile()
    
    if rank == 0
        #positions, weights = gaussian_2d(N_side, extent)
        positions, weights = uniform_2d(N_side, extent)
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

    # set eval positions
    if rank == 0
        positions, weights = uniform_2d(N_sample_side, extent)
        set_eval_positions(dg_project_2d, positions)
    else
        set_eval_positions(dg_project_2d)
    end

    write(ParticleGroupVTK("particle_positions", A))


    for ix in 1:100
        project_evaluate(dg_project_2d)
        if rank == 0
            @show ix
        end
    end
    
    N_step = 1000
    reset_profile()
    time_start = time()
    for ix in 1:N_step
        project_evaluate(dg_project_2d)
    end
    time_end = time()
    if rank == 0
        print_profile()
        @show (time_end - time_start)/N_step
    end


    write(ParticleGroupVTK("function_evals", dg_project_2d.particle_group_eval))
    free(A)
    JFPI.free(dg_project_2d)

end

main()



