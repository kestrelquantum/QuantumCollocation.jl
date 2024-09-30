module TrajectoryInitialization

export unitary_geodesic
export linear_interpolation
export unitary_linear_interpolation
export initialize_unitary_trajectory
export initialize_quantum_state_trajectory
export convert_fixed_time
export convert_free_time

using NamedTrajectories

using Distributions
using ExponentialAction
using LinearAlgebra
using TestItemRunner

using ..Isomorphisms
using ..QuantumSystems
using ..Rollouts
using ..EmbeddedOperators
using ..DirectSums


# ----------------------------------------------------------------------------- #
#                           Initial states                                      #
# ----------------------------------------------------------------------------- #

"""
    unitary_linear_interpolation(
        U_init::AbstractMatrix,
        U_goal::AbstractMatrix,
        samples::Int
    )

Compute a linear interpolation of unitary operators with `samples` samples.
"""
function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    Ũ⃗s = [Ũ⃗_init + (Ũ⃗_goal - Ũ⃗_init) * t for t ∈ range(0, 1, length=samples)]
    Ũ⃗ = hcat(Ũ⃗s...)
    return Ũ⃗
end

function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int
)
    return unitary_linear_interpolation(U_init, U_goal.operator, samples)
end

"""
    unitary_geodesic(
        operator::EmbeddedOperator,
        samples::Int;
        kwargs...
    )

    unitary_geodesic(
        U_goal::AbstractMatrix{<:Number},
        samples::Int;
        kwargs...
    )

    unitary_geodesic(
        U₀::AbstractMatrix{<:Number},
        U₁::AbstractMatrix{<:Number},
        samples::Number;
        kwargs...
    )

    unitary_geodesic(
        U₀::AbstractMatrix{<:Number},
        U₁::AbstractMatrix{<:Number},
        timesteps::AbstractVector{<:Number};
        return_generator=false
    )

Compute a geodesic connecting two unitary operators.
"""
function unitary_geodesic end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int;
    kwargs...
)
    U1 = unembed(U_init, U_goal)
    U2 = unembed(U_goal)
    Ũ⃗ = unitary_geodesic(U1, U2, samples; kwargs...)
    return hcat([
        operator_to_iso_vec(embed(iso_vec_to_operator(Ũ⃗ₜ), U_goal))
        for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    return unitary_geodesic(U_init, U_goal, range(0, 1, samples); kwargs...)
end

function unitary_geodesic(
    U_goal::OperatorType,
    samples::Int;
    kwargs...
)
    if U_goal isa EmbeddedOperator
        U_goal = U_goal.operator
    end
    return unitary_geodesic(Matrix{ComplexF64}(I(size(U_goal, 1))), U_goal, samples; kwargs...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    timesteps::AbstractVector{<:Number};
    return_unitary_isos=true,
    return_generator=false
)
    """
    Compute the effective generator of the geodesic connecting U₀ and U₁.
        U_goal = exp(-im * H * T) U_init
        log(U_goal * U_init') = -im * H * T

    Allow for the possibiltiy of unequal timesteps and ranges outside [0,1].

    Returns the geodesic.
    Optionally returns the effective Hamiltonian generating the geodesic.
    """
    t₀ = timesteps[1]
    T = timesteps[end] - t₀
    H = im * log(U_goal * U_init') / T
    # -im prefactor is not included in H
    U_geo = [exp(-im * H * (t - t₀)) * U_init for t ∈ timesteps]
    if !return_unitary_isos
        if return_generator
            return U_geo, H
        else
            return U_geo
        end
    else
        Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
        if return_generator
            return Ũ⃗_geo, H
        else
            return Ũ⃗_geo
        end
    end
end

linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) =
    hcat(range(x, y, n)...)

# ============================================================================= #

const VectorBound = Union{AbstractVector{R}, Tuple{AbstractVector{R}, AbstractVector{R}}} where R <: Real
const ScalarBound = Union{R, Tuple{R, R}} where R <: Real

function initialize_unitaries(
    U_init::AbstractMatrix{<:Number},
    U_goal::OperatorType,
    T::Int;
    geodesic=true
)
    if geodesic
        Ũ⃗ = unitary_geodesic(U_init, U_goal, T)
    else
        Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, T)
    end
    return Ũ⃗
end

# ----------------------------------------------------------------------------- #
#                           Initial controls                                    #
# ----------------------------------------------------------------------------- #

function initialize_controls(
    n_drives::Int,
    n_derivatives::Int,
    T::Int,
    bounds::VectorBound,
    drive_derivative_σ::Float64,
)
    if bounds isa AbstractVector
        a_dists = [Uniform(-bounds[i], bounds[i]) for i = 1:n_drives]
    elseif bounds isa Tuple
        a_dists = [Uniform(aᵢ_lb, aᵢ_ub) for (aᵢ_lb, aᵢ_ub) ∈ zip(bounds...)]
    else
        error("bounds must be a Vector or Tuple")
    end

    controls = Matrix{Float64}[]

    a = hcat([
        zeros(n_drives),
        vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
        zeros(n_drives)
    ]...)
    push!(controls, a)

    for _ in 1:n_derivatives
        push!(controls, randn(n_drives, T) * drive_derivative_σ)
    end

    return controls
end

function initialize_controls(a::AbstractMatrix, Δt::AbstractVecOrMat, n_derivatives::Int)
    controls = Matrix{Float64}[a]
    for n in 1:n_derivatives
        # next derivative
        push!(controls,  derivative(controls[end], Δt))

        # to avoid constraint violation error at initial iteration for da, dda, ...
        if n > 1
            controls[end-1][:, end] = controls[end-1][:, end-1] + Δt[end-1] * controls[end][:, end-1]
        end
    end
    return controls
end

initialize_controls(a::AbstractMatrix, Δt::Real, n_derivatives::Int) =
    initialize_controls(a, fill(Δt, size(a, 2)), n_derivatives)

# ----------------------------------------------------------------------------- #
#                           Trajectory initialization                           #
# ----------------------------------------------------------------------------- #

"""
    initialize_unitary_trajectory


Initialize a trajectory for a unitary control problem. The trajectory is initialized with
data that should be consistently the same type (in this case, Float64).

"""
function initialize_unitary_trajectory(
    U_goal::OperatorType,
    T::Int,
    Δt::Union{Float64, AbstractVecOrMat{<:Float64}},
    n_drives::Int,
    control_bounds::Tuple{Vararg{VectorBound}};
    state_name=:Ũ⃗,
    control_name=:a,
    timestep_name=:Δt,
    U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(U_goal, 1))),
    n_derivatives::Int=0,
    geodesic=true,
    bound_unitary=false,
    free_time=false,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, AbstractVector{<:AbstractQuantumSystem}, Nothing}=nothing,
    global_data::Union{NamedTuple, Nothing}=nothing,
    rollout_integrator::Function=expv,
    state_names::AbstractVector{<:Symbol}=[state_name],
)
    control_derivative_names = [
        Symbol("d"^i * string(control_name))
            for i = 1:n_derivatives
    ]

    control_names = (control_name, control_derivative_names...)

    control_bounds = NamedTuple{control_names}(control_bounds)

    if free_time
        if Δt isa Real
            Δt = fill(Δt, 1, T)
        elseif Δt isa AbstractVector
            Δt = reshape(Δt, 1, :)
        else
            @assert size(Δt) == (1, T) "Δt must be a Real, AbstractVector, or 1x$(T) AbstractMatrix"
        end
        timestep = timestep_name
    else
        @assert Δt isa Real "Δt must be a Real if free_time is false"
        timestep = Δt
    end

    Ũ⃗_init = operator_to_iso_vec(U_init)
    if U_goal isa EmbeddedOperator
        Ũ⃗_goal = operator_to_iso_vec(U_goal.operator)
    else
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
    end

    # Constraints
    Ũ⃗_inits = repeat([Ũ⃗_init], length(state_names))
    initial = (;
        (state_names .=> Ũ⃗_inits)...,
        a = zeros(n_drives),
    )

    final = (
        a = zeros(n_drives),
    )

    Ũ⃗_goals = repeat([Ũ⃗_goal], length(state_names))
    goal = (; (state_names .=> Ũ⃗_goals)...)

    # Bounds
    bounds = control_bounds

    if bound_unitary
        Ũ⃗_dim = length(Ũ⃗_init)
        Ũ⃗_bounds = repeat([(-ones(Ũ⃗_dim), ones(Ũ⃗_dim))], length(state_names))
        bounds = merge(bounds, (; (state_names .=> Ũ⃗_bounds)...))
    end

    # Initial state and control values
    if isnothing(a_guess)
        Ũ⃗ = initialize_unitaries(U_init, U_goal, T, geodesic=geodesic)
        Ũ⃗_values = repeat([Ũ⃗], length(state_names))
        a_values = initialize_controls(
            n_drives,
            n_derivatives,
            T,
            bounds[control_name],
            drive_derivative_σ
        )
    else
        @assert size(a_guess, 1) == n_drives "a_guess must have the same number of drives as n_drives"
        @assert size(a_guess, 2) == T "a_guess must have the same number of timesteps as T"
        @assert !isnothing(system) "system must be provided if a_guess is provided"

        if Δt isa AbstractMatrix
            ts = vec(Δt)
        elseif Δt isa Float64
            ts = fill(Δt, T)
        else
            ts = Δt
        end

        if system isa AbstractVector
            @assert length(system) == length(state_names) "systems must have the same length as state_names"
            Ũ⃗_values = map(system) do sys
                unitary_rollout(Ũ⃗_init, a_guess, ts, sys; integrator=rollout_integrator)
            end
        else
            Ũ⃗ = unitary_rollout(Ũ⃗_init, a_guess, ts, system; integrator=rollout_integrator)
            Ũ⃗_values = repeat([Ũ⃗], length(state_names))
        end
        Ũ⃗_values = Matrix{Float64}.(Ũ⃗_values)
        a_values = initialize_controls(a_guess, ts, n_derivatives)
    end

    # Trajectory
    names = [state_names..., control_names...]
    values = [Ũ⃗_values..., a_values...]

    if free_time
        push!(names, timestep_name)
        push!(values, Δt)
        controls = (control_names[end], :Δt)
        bounds = merge(bounds, (Δt = Δt_bounds,))
    else
        controls = (control_names[end],)
    end

    return NamedTrajectory(
        (; (names .=> values)...);
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal,
        global_data= isnothing(global_data) ? (;) : global_data
    )
end

function initialize_quantum_state_trajectory(
    ψ̃_goals::AbstractVector{<:AbstractVector{<:Real}},
    ψ̃_inits::AbstractVector{<:AbstractVector{<:Real}},
    T::Int,
    Δt::Union{Real, AbstractVector{<:Real}},
    n_drives::Int,
    all_a_bounds::NamedTuple{anames, <:Tuple{Vararg{VectorBound}}} where anames;
    n_derivatives::Int=2,
    free_time=false,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, AbstractVector{<:AbstractQuantumSystem}, Nothing}=nothing,
    global_data::Union{NamedTuple, Nothing}=nothing,
    rollout_integrator::Function=exp,
    ψ̃_keys::AbstractVector{<:Symbol}=[Symbol("ψ̃$i") for i = 1:length(ψ̃_goals)],
    a_keys::AbstractVector{<:Symbol}=[Symbol("d"^i * "a") for i = 0:n_derivatives]
)
    @assert length(ψ̃_inits) == length(ψ̃_goals) "ψ̃_inits and ψ̃_goals must have the same length"
    @assert length(ψ̃_keys) == length(ψ̃_goals) "ψ̃_keys and ψ̃_goals must have the same length"

    if free_time
        if Δt isa Real
            Δt = fill(Δt, 1, T)
        elseif Δt isa AbstractVector
            Δt = reshape(Δt, 1, :)
        else
            @assert size(Δt) == (1, T) "Δt must be a Real, AbstractVector, or 1x$(T) AbstractMatrix"
        end
    end

    # Constraints
    state_initial = (; (ψ̃_keys .=> ψ̃_inits)...)
    control_initial = (a = zeros(n_drives),)
    initial = merge(state_initial, control_initial)

    final = (a = zeros(n_drives),)

    goal = (; (ψ̃_keys .=> ψ̃_goals)...)

    # Bounds
    bounds = all_a_bounds

    # Initial state and control values
    if isnothing(a_guess)
        ψ̃_values = NamedTuple([
            k => linear_interpolation(ψ̃_init, ψ̃_goal, T)
                for (k, ψ̃_init, ψ̃_goal) in zip(ψ̃_keys, ψ̃_inits, ψ̃_goals)
        ])
        a_values = initialize_controls(
            n_drives,
            n_derivatives,
            T,
            bounds[a_keys[1]],
            drive_derivative_σ
        )
    else
        ψ̃_values = NamedTuple([
            k => rollout(ψ̃_init, a_guess, Δt, system, integrator=rollout_integrator)
                for (k, ψ̃_init) in zip(ψ̃_keys, ψ̃_inits)
        ])
        a_values = initialize_controls(a_guess, Δt, n_derivatives)
    end

    # Trajectory
    keys = [ψ̃_keys..., a_keys...]
    values = [ψ̃_values..., a_values...]

    if free_time
        push!(keys, :Δt)
        push!(values, Δt)
        controls = (a_keys[end], :Δt)
        timestep = :Δt
        bounds = merge(bounds, (Δt = Δt_bounds,))
    else
        controls = (a_keys[end],)
        @assert Δt isa Real "Δt must be a Real if free_time is false"
        timestep = Δt
    end

    return NamedTrajectory(
        (; (keys .=> values)...);
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal,
        global_data=isnothing(global_data) ? (;) : global_data
    )
end

# ============================================================================= #

remove_component(
    names::NTuple{N, Symbol} where N,
    remove_name::Symbol
) = ([n for n in names if n != remove_name]...,)

function remove_component(
    container,
    names::NTuple{N, Symbol} where N,
    remove_name::Symbol,
)
    keys = Symbol[]
    vals = []
    for symb in names
        if symb != remove_name
            push!(keys, symb)
            push!(vals, container[symb])
        end
    end
    return (; (keys .=> vals)...)
end

function convert_fixed_time(
    traj::NamedTrajectory;
    Δt_symb=:Δt,
    timestep = sum(get_timesteps(traj)) / traj.T
)
    @assert Δt_symb ∈ traj.control_names "Problem must be free time"
    return NamedTrajectory(
        remove_component(traj, traj.names, Δt_symb);
        controls=remove_component(traj.control_names, Δt_symb),
        timestep=timestep,
        bounds=remove_component(traj.bounds, keys(traj.bounds), Δt_symb),
        initial=remove_component(traj.initial, keys(traj.initial), Δt_symb),
        final=remove_component(traj.final, keys(traj.final), Δt_symb),
        goal=remove_component(traj.goal, keys(traj.goal), Δt_symb)
    )
end

function convert_free_time(
    traj::NamedTrajectory,
    Δt_bounds::Union{ScalarBound, BoundType};
    Δt_symb=:Δt,
)
    @assert Δt_symb ∉ traj.control_names "Problem must not be free time"

    Δt_bound = (; Δt_symb => Δt_bounds,)
    time_data = (; Δt_symb => get_timesteps(traj))
    comp_data = get_components(traj)

    return NamedTrajectory(
        merge_outer(comp_data, time_data);
        controls=merge_outer(traj.control_names, (Δt_symb,)),
        timestep=Δt_symb,
        bounds=merge_outer(traj.bounds, Δt_bound),
        initial=traj.initial,
        final=traj.final,
        goal=traj.goal
    )
end

# ============================================================================= #

@testitem "Random drive initialization" begin
    T = 10
    n_drives = 2
    n_derivates = 2
    drive_bounds = [1.0, 2.0]
    drive_derivative_σ = 0.01

    a, da, dda = TrajectoryInitialization.initialize_controls(n_drives, n_derivates, T, drive_bounds, drive_derivative_σ)

    @test size(a) == (n_drives, T)
    @test size(da) == (n_drives, T)
    @test size(dda) == (n_drives, T)
    @test all([-drive_bounds[i] < minimum(a[i, :]) < drive_bounds[i] for i in 1:n_drives])
end

@testitem "Geodesic" begin
    using LinearAlgebra

    ## Group 1: identity to X (π rotation)

    # Test π rotation
    U_α = GATES[:I]
    U_ω = GATES[:X]
    Us, H = unitary_geodesic(
        U_α, U_ω, range(0, 1, 4), return_generator=true
    )

    @test size(Us, 2) == 4
    @test Us[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H' - H ≈ zeros(2, 2)
    @test norm(H) ≈ π

    # Test modified timesteps (10x)
    Us10, H10 = unitary_geodesic(
        U_α, U_ω, range(-5, 5, 4), return_generator=true
    )

    @test size(Us10, 2) == 4
    @test Us10[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us10[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H10' - H10 ≈ zeros(2, 2)
    @test norm(H10) ≈ π/10

    # Test wrapped call
    Us_wrap, H_wrap = unitary_geodesic(U_ω, 10, return_generator=true)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)
    rotation = [exp(-im * H_wrap * t) for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us_wrap, Us_test)


    ## Group 2: √X to X (π/2 rotation)

    # Test geodesic not at identity
    U₀ = sqrt(GATES[:X])
    U₁ = GATES[:X]
    Us, H = unitary_geodesic(U₀, U₁, 10, return_generator=true)
    @test Us[:, 1] ≈ operator_to_iso_vec(U₀)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)

    rotation = [exp(-im * H * t) * U₀ for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us, Us_test)
    Us_wrap = unitary_geodesic(U_ω, 4)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)

end

@testitem "Free and fixed time conversion" begin
    using NamedTrajectories
    include("../test/test_utils.jl")

    free_traj = named_trajectory_type_1(free_time=true)
    fixed_traj = named_trajectory_type_1(free_time=false)
    Δt_bounds = free_traj.bounds[:Δt]

    # Test free to fixed time
    @test :Δt ∉ convert_fixed_time(free_traj).control_names

    # Test fixed to free time
    @test :Δt ∈ convert_free_time(fixed_traj, Δt_bounds).control_names

    # Test inverses
    @test convert_free_time(convert_fixed_time(free_traj), Δt_bounds) == free_traj
    @test convert_fixed_time(convert_free_time(fixed_traj, Δt_bounds)) == fixed_traj
end

@testitem "unitary trajectory initialization" begin
    using NamedTrajectories
    U_goal = GATES[:X]
    T = 10
    Δt = 0.1
    n_drives = 2
    all_a_bounds = (a = [1.0, 1.0],)

    traj = initialize_unitary_trajectory(
        U_goal, T, Δt, n_drives, all_a_bounds
    )

    @test traj isa NamedTrajectory
end

@testitem "quantum state trajectory initialization" begin
    using NamedTrajectories

    ψ̃_init = ket_to_iso([0.0, 1.0])
    ψ̃_goal = ket_to_iso([1.0, 0.0])

    T = 10
    Δt = 0.1
    n_drives = 2
    all_a_bounds = (a = [1.0, 1.0],)

    traj = initialize_quantum_state_trajectory(
        [ψ̃_goal], [ψ̃_init], T, Δt, n_drives, all_a_bounds
    )

    @test traj isa NamedTrajectory
end


end
