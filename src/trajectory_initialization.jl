module TrajectoryInitialization

export unitary_geodesic
export linear_interpolation
export unitary_linear_interpolation
export initialize_unitary_trajectory
export initialize_quantum_state_trajectory
export convert_fixed_time
export convert_free_time

using NamedTrajectories
using LinearAlgebra
using Distributions

using ..QuantumUtils
using ..QuantumSystems
using ..Rollouts
using ..EmbeddedOperators
using ..DirectSums

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
    U_goal::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
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
    Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
    if return_generator
        return Ũ⃗_geo, H
    else
        return Ũ⃗_geo
    end
end

linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) =
    hcat(range(x, y, n)...)

# ============================================================================= #

const VectorBound = Union{AbstractVector{R}, Tuple{AbstractVector{R}, AbstractVector{R}}} where R <: Real
const ScalarBound = Union{R, Tuple{R, R}} where R <: Real

function initialize_unitaries(
    U_init::AbstractMatrix{<:Number},
    U_goal::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
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

function initialize_controls(
    n_drives::Int,
    T::Int,
    a_bounds::VectorBound,
    drive_derivative_σ::Float64
)
    if a_bounds isa AbstractVector
        a_dists = [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
    elseif a_bounds isa Tuple
        a_dists = [Uniform(aᵢ_lb, aᵢ_ub) for (aᵢ_lb, aᵢ_ub) ∈ zip(a_bounds...)]
    else
        error("a_bounds must be a Vector or Tuple")
    end

    a = hcat([
        zeros(n_drives),
        vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
        zeros(n_drives)
    ]...)

    da = randn(n_drives, T) * drive_derivative_σ
    dda = randn(n_drives, T) * drive_derivative_σ
    return a, da, dda
end

function initialize_controls(a::AbstractMatrix, Δt::AbstractVecOrMat)
    da = derivative(a, Δt)
    dda = derivative(da, Δt)

    # to avoid constraint violation error at initial iteration
    da[:, end] = da[:, end-1] + Δt[end-1] * dda[:, end-1]

    return a, da, dda
end

function initialize_unitary_trajectory(
    U_goal::Union{EmbeddedOperator, AbstractMatrix{<:Number}},
    T::Int,
    Δt::Real,
    n_drives::Int,
    a_bounds::VectorBound,
    dda_bounds::VectorBound;
    U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(U_goal, 1))),
    geodesic=true,
    bound_unitary=false,
    free_time=false,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, AbstractVector{<:AbstractQuantumSystem}, Nothing}=nothing,
    rollout_integrator::Function=exp,
    Ũ⃗_keys::AbstractVector{<:Symbol}=[:Ũ⃗],
)
    if free_time
        if Δt isa Float64
            Δt = fill(Δt, 1, T)
        end
    end

    Ũ⃗_init = operator_to_iso_vec(U_init)
    if U_goal isa EmbeddedOperator
        Ũ⃗_goal = operator_to_iso_vec(U_goal.operator)
    else
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
    end

    # Constraints
    Ũ⃗_inits = repeat([Ũ⃗_init], length(Ũ⃗_keys))
    initial = (;
        (Ũ⃗_keys .=> Ũ⃗_inits)...,
        a = zeros(n_drives),
    )

    final = (
        a = zeros(n_drives),
    )

    Ũ⃗_goals = repeat([Ũ⃗_goal], length(Ũ⃗_keys))
    goal = (; (Ũ⃗_keys .=> Ũ⃗_goals)...)

    # Bounds
    bounds = (a = a_bounds, dda = dda_bounds,)

    if bound_unitary
        Ũ⃗_dim = length(Ũ⃗_init)
        Ũ⃗_bounds = repeat([(-ones(Ũ⃗_dim), ones(Ũ⃗_dim))], length(Ũ⃗_keys))
        bounds = merge(bounds, (; (Ũ⃗_keys .=> Ũ⃗_bounds)...))
    end

    # Initial state and control values
    if isnothing(a_guess)
        Ũ⃗ = initialize_unitaries(U_init, U_goal, T, geodesic=geodesic)
        Ũ⃗_values = repeat([Ũ⃗], length(Ũ⃗_keys))
        a, da, dda = initialize_controls(n_drives, T, a_bounds, drive_derivative_σ)
    else
        @assert !isnothing(system) "system must be provided if a_guess is provided"
        if system isa AbstractVector
            @assert length(system) == length(Ũ⃗_keys) "systems must have the same length as Ũ⃗_keys"
            Ũ⃗_values = map(system) do sys
                unitary_rollout(Ũ⃗_init, a_guess, Δt, sys; integrator=rollout_integrator)
            end
        else
            unitary_rollout(Ũ⃗_init, a_guess, Δt, system; integrator=rollout_integrator)
            Ũ⃗_values = repeat([Ũ⃗], length(Ũ⃗_keys))
        end
        a, da, dda = initialize_controls(a_guess, Δt)
    end

    # Trajectory
    keys = [Ũ⃗_keys..., :a, :da, :dda]
    values = [Ũ⃗_values..., a, da, dda]

    if free_time
        push!(keys, :Δt)
        push!(values, Δt)
        controls = (:dda, :Δt)
        timestep = :Δt
        bounds = merge(bounds, (Δt = Δt_bounds,))
    else
        controls = (:dda,)
        timestep = Δt
    end

    return NamedTrajectory(
        (; (keys .=> values)...);
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end

function initialize_quantum_state_trajectory(
    ψ̃_goals::AbstractVector{<:AbstractVector{<:Real}},
    ψ̃_inits::AbstractVector{<:AbstractVector{<:Real}},
    T::Int,
    Δt::Real,
    n_drives::Int,
    a_bounds::VectorBound,
    dda_bounds::VectorBound;
    free_time=false,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, AbstractVector{<:AbstractQuantumSystem}, Nothing}=nothing,
    rollout_integrator::Function=exp,    
    ψ̃_keys::AbstractVector{<:Symbol}=[Symbol("ψ̃$i") for i = 1:length(ψ̃_goals)]
)
    @assert length(ψ̃_inits) == length(ψ̃_goals) "ψ̃_inits and ψ̃_goals must have the same length"
    @assert length(ψ̃_keys) == length(ψ̃_goals) "ψ̃_keys and ψ̃_goals must have the same length"

    if free_time
        if Δt isa Float64
            Δt = fill(Δt, 1, T)
        end
    end

    # Constraints
    state_initial = (; (ψ̃_keys .=> ψ̃_inits)...)
    control_initial = (a = zeros(n_drives),)
    initial = merge(state_initial, control_initial)

    final = (a = zeros(n_drives),)

    goal = (; (ψ̃_keys .=> ψ̃_goals)...)

    # Bounds
    bounds = (a = a_bounds, dda = dda_bounds,)

    # Initial state and control values
    if isnothing(a_guess)
        ψ̃s = NamedTuple([
            k => linear_interpolation(ψ̃_init, ψ̃_goal, T)
                for (k, ψ̃_init, ψ̃_goal) in zip(ψ̃_keys, ψ̃_inits, ψ̃_goals)
        ])
        a, da, dda = initialize_controls(n_drives, T, a_bounds, drive_derivative_σ)
    else
        ψ̃s = NamedTuple([
            k => rollout(ψ̃_init, a_guess, Δt, system, integrator=rollout_integrator)
                for (i, ψ̃_init) in zip(ψ̃_keys, ψ̃_inits)
        ])
        a, da, dda = initialize_controls(a_guess, Δt)
    end

    # Trajectory
    keys = [ψ̃_keys..., :a, :da, :dda]
    values = [ψ̃s..., a, da, dda]

    if free_time
        push!(keys, :Δt)
        push!(values, Δt)
        controls = (:dda, :Δt)
        timestep = :Δt
        bounds = merge(bounds, (Δt = Δt_bounds,))
    else
        controls = (:dda,)
        timestep = Δt
    end

    return NamedTrajectory(
        (; (keys .=> values)...);
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
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

end
