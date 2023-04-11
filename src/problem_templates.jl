module ProblemTemplates

export UnitarySmoothPulseProblem

using ..QuantumSystems
using ..QuantumUtils
using ..Rollouts
using ..Objectives
using ..Constraints
using ..Integrators
using ..Problems

using NamedTrajectories
using LinearAlgebra
using Distributions

function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    U_goal::AbstractMatrix{<:Number},
    T::Int,
    Δt::Float64;
    a_bound::Float64=Inf,
    a_bounds::Vector{Float64}=fill(a_bound, length(H_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=Inf,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(H_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R_a::Union{Float64, Vector{Float64}}=1.0e-2,
    R_da::Union{Float64, Vector{Float64}}=1.0e-2,
    R_dda::Union{Float64, Vector{Float64}}=1.0e-2,
    max_iter::Int=100,
    linear_solver::String="mumps",
    constraints::Vector{<:LinearConstraint}=LinearConstraint[],
    nl_constraints::Vector{<:NonlinearConstraint}=NonlinearConstraint[],
    timesteps_all_equal::Bool=true
)
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    Ũ⃗ = unitary_geodesic(U_goal, T)

    Δt = fill(Δt, 1, T)

    if isnothing(a_guess)
        a_dists =  [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
        a = hcat([
            zeros(n_drives),
            vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
            zeros(n_drives)
        ]...)
        da = randn(n_drives, T) * drive_derivative_σ
        dda = randn(n_drives, T) * drive_derivative_σ
    else
        a = a_guess
        da = derivative(a, Δt)
        dda = derivative(da, Δt)
    end

    components = (
        Ũ⃗ = Ũ⃗,
        a = a,
        da = da,
        dda = dda,
        Δt = Δt,
    )

    bounds = (
        a = a_bounds,
        dda = dda_bounds,
        Δt = (Δt_min, Δt_max),
    )

    initial = (
        Ũ⃗ = operator_to_iso_vec(I(size(U_goal, 1))),
        a = zeros(n_drives),
    )

    final = (
        a = zeros(n_drives),
    )

    goal = (
        Ũ⃗ = operator_to_iso_vec(U_goal),
    )

    traj = NamedTrajectory(
        components;
        controls=(:dda, :Δt),
        timestep=Δt[1],
        dynamical_timesteps=true,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )

    J = QuantumUnitaryObjective(:Ũ⃗, traj, Q)
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    integrators = [
        UnitaryPadeIntegrator(system, :Ũ⃗, :a, :Δt),
        DerivativeIntegrator(:a, :da, :Δt, traj.dims[:a]),
        DerivativeIntegrator(:da, :dda, :Δt, traj.dims[:da]),
    ]

    if timesteps_all_equal
        push!(constraints, TimeStepsEqualConstraint(:Δt, traj))
    end

    return QuantumControlProblem(
        traj,
        system,
        J,
        integrators;
        constraints=constraints,
        nl_constraints=nl_constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
    )
end



end