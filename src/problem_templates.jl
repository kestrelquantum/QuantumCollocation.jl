module ProblemTemplates

export UnitarySmoothPulseProblem
export UnitaryMinimumTimeProblem

using ..QuantumSystems
using ..QuantumUtils
using ..Rollouts
using ..Objectives
using ..Constraints
using ..Integrators
using ..Problems
using ..IpoptOptions

using NamedTrajectories
using LinearAlgebra
using Distributions
using JLD2

function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    U_goal::AbstractMatrix{<:Number},
    T::Int,
    Δt::Float64;
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=Inf,
    a_bounds::Vector{Float64}=fill(a_bound, length(H_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=Inf,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(H_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    ipopt_options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    timesteps_all_equal::Bool=true,
    verbose=false,
)
    U_goal = Matrix{ComplexF64}(U_goal)

    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        Δt = fill(Δt, 1, T)

        if isnothing(a_guess)
            Ũ⃗ = unitary_geodesic(U_goal, T)
            a_dists =  [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
            a = hcat([
                zeros(n_drives),
                vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
                zeros(n_drives)
            ]...)
            da = randn(n_drives, T) * drive_derivative_σ
            dda = randn(n_drives, T) * drive_derivative_σ
        else
            Ũ⃗ = unitary_rollout(a, Δt, system)
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
            Ũ⃗ = operator_to_iso_vec(1.0I(size(U_goal, 1))),
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
            timestep=:Δt,
            bounds=bounds,
            initial=initial,
            final=final,
            goal=goal
        )
    end

    J = QuantumUnitaryObjective(:Ũ⃗, traj, Q)
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    integrators = [
        UnitaryPadeIntegrator(system, :Ũ⃗, :a, :Δt),
        DerivativeIntegrator(:a, :da, :Δt, traj),
        DerivativeIntegrator(:da, :dda, :Δt, traj),
    ]

    if timesteps_all_equal
        push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
    end

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
    )
end

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    objective += MinimumTimeObjective(trajectory; D=D)

    final_fidelity = unitary_fidelity(trajectory[end].Ũ⃗, trajectory.goal.Ũ⃗)

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory
    )

    push!(constraints, fidelity_constraint)

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    data_path::String;
    kwargs...
)
    data = load(data_path)
    system = data["system"]
    trajectory = data["trajectory"]
    objective = Objective(data["params"][:objective_terms])
    integrators = data["params"][:dynamics]
    constraints = AbstractConstraint[
        data["params"][:linear_constraints]...,
        NonlinearConstraint.(data["params"][:nonlinear_constraints])...
    ]
    return UnitaryMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end


end
