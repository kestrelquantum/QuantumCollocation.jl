module TrajectoryInterpolations

export DataInterpolation

using NamedTrajectories

using Interpolations: Extrapolation, constant_interpolation, linear_interpolation
using TestItems


struct DataInterpolation
    times::Vector{Float64}
    values::Matrix{Float64}
    interpolants::Vector{Extrapolation}
    timestep_components::Vector{Int}
    values_components::Vector{Int}

    function DataInterpolation(
        times::AbstractVector{Float64},
        values::AbstractMatrix{Float64};
        timestep_components::AbstractVector{Int}=Int[],
        kind::Symbol=:linear
    )
        comps = setdiff(1:size(values, 1), timestep_components)
        if kind == :linear
            interpolants = [linear_interpolation(times, values[c, :]) for c in comps]
        elseif kind == :constant
            interpolants = [constant_interpolation(times, values[c, :]) for c in comps]
        else
            error("Unknown interpolation kind: $kind")
        end
        return new(times, values, interpolants, timestep_components, comps)
    end

    function DataInterpolation(
        T::Int, Δt::Real, values::AbstractMatrix{Float64}; kwargs...
    )
        times = range(0, Δt * (T - 1), step=Δt)
        return DataInterpolation(times, values; kwargs...)
    end

    function DataInterpolation(
        traj::NamedTrajectory; timestep_name::Symbol=:Δt, kwargs...
    )
        if timestep_name ∈ keys(traj.components)
            timestep_components = traj.components[timestep_name]
        else
            timestep_components = Int[]
        end
        return DataInterpolation(
            get_times(traj), traj.data; timestep_components=timestep_components, kwargs...
        )
    end
end

function (traj_int::DataInterpolation)(times::AbstractVector)
    values = zeros(eltype(traj_int.values), size(traj_int.values, 1), length(times))
    for (c, interp) in zip(traj_int.values_components, traj_int.interpolants)
        values[c, :] = interp(times)
    end
    if !isempty(traj_int.timestep_components)
        timesteps = times[2:end] .- times[1:end-1]
        # NOTE: Arbitrary choice of the last timestep
        values[traj_int.timestep_components, :] = vcat(timesteps, timesteps[end])
    end
    return values
end

function (traj_int::DataInterpolation)(T::Int)
    times = range(traj_int.times[1], traj_int.times[end], length=T)
    return traj_int(times)
end

# *************************************************************************** #

@testitem "Trajectory interpolation test" begin
    include("../test/test_utils.jl")

    # fixed time
    traj = named_trajectory_type_1()

    interp = DataInterpolation(traj)
    new_data = interp(get_times(traj))
    @test new_data ≈ traj.data

    new_data = interp(2 * traj.T)
    @test size(new_data) == (size(traj.data, 1), 2 * traj.T)

    # free time
    free_traj = named_trajectory_type_1(free_time=true)

    interp = DataInterpolation(free_traj)
    new_free_data = interp(get_times(traj))

    # Replace the final timestep with the original value (can't be known a priori)
    new_free_data[free_traj.components.Δt, end] = free_traj.data[free_traj.components.Δt, end]
    @test new_free_data ≈ free_traj.data

    new_free_data = interp(2 * traj.T)
    @test size(new_free_data) == (size(free_traj.data, 1), 2 * traj.T)
end

@testitem "Component interpolation test" begin
    include("../test/test_utils.jl")

    traj = named_trajectory_type_1()

    # interpolate with times
    interp_val1 = DataInterpolation(get_times(traj), traj.a)
    @test size(interp_val1(2 * traj.T)) == (size(traj.a, 1), 2 * traj.T)

    # interpolate with steps
    interp_val2 = DataInterpolation(traj.T, traj.timestep, traj.a)
    @test size(interp_val2(3 * traj.T)) == (size(traj.a, 1), 3 * traj.T)

    # check if times match
    @test interp_val1.times ≈ interp_val2.times
end


end
