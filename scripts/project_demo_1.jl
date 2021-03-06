using PPMD
using JFPI

using SpecialFunctions
using MPI

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
    
    N_side = 1000
    N_sample_side = 50
    p = 2
    extent = 1.0

    extents = (extent, extent)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)
    rank = MPI.Comm_rank(domain.comm)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(2, position=true),
             "Q" => ParticleDat(1),
        ),
        target_device
    )
    
    dg_project_2d = DGProject2D(A, p, 16, 16)

    JFPI.uniform_grid_gaussian_weights(N_side, A)

    reset_profile()

    # set eval positions
    if rank == 0
        positions, weights = uniform_2d(N_sample_side, extent)
        set_eval_positions(dg_project_2d, positions)
    else
        set_eval_positions(dg_project_2d)
    end

    write(ParticleGroupVTK("particle_positions", A))
    
    project_evaluate(dg_project_2d, "Q")

    write(ParticleGroupVTK("function_evals", dg_project_2d.particle_group_eval))


    PPMD.free(A)
    JFPI.free(dg_project_2d)

end

main()



