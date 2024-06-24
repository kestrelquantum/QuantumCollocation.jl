module Evaluators

export PicoEvaluator

using ..QuantumSystems
using ..Integrators
using ..Dynamics
using ..Objectives
using ..Constraints

using NamedTrajectories
using SparseArrays
using MathOptInterface
using LinearAlgebra
const MOI = MathOptInterface

mutable struct PicoEvaluator <: MOI.AbstractNLPEvaluator
    trajectory::NamedTrajectory
    objective::Objective
    dynamics::QuantumDynamics
    n_dynamics_constraints::Int
    nonlinear_constraints::Vector{<:NonlinearConstraint}
    n_nonlinear_constraints::Int
    eval_hessian::Bool
    zdim::Int
    verbose::Bool

    function PicoEvaluator(
        trajectory::NamedTrajectory,
        objective::Objective,
        dynamics::QuantumDynamics,
        nonlinear_constraints::Vector{<:NonlinearConstraint};
        eval_hessian::Bool=true,
        verbose::Bool = false
    )
        n_dynamics_constraints = dynamics.dim * (trajectory.T - 1)
        n_nonlinear_constraints = sum(con.dim for con ∈ nonlinear_constraints; init=0)

        return new(
            trajectory,
            objective,
            dynamics,
            n_dynamics_constraints,
            nonlinear_constraints,
            n_nonlinear_constraints,
            eval_hessian,
            trajectory.dim,
            verbose
        )
    end
end

MOI.initialize(::PicoEvaluator, features) = nothing

function MOI.features_available(evaluator::PicoEvaluator)
    if evaluator.eval_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end


# objective and gradient

@views function MOI.eval_objective(
    evaluator::PicoEvaluator,
    Z⃗::AbstractVector
)
    if evaluator.verbose
        println("evaluating objective")
        time = time_ns()
    end
    L = evaluator.objective.L(Z⃗, evaluator.trajectory)
    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("objective evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end
return L
end

@views function MOI.eval_objective_gradient(
    evaluator::PicoEvaluator,
    ∇::AbstractVector,
    Z⃗::AbstractVector
)
    if evaluator.verbose
        println("evaluating objective gradient...")
        time = time_ns()
    end
    ∇[:] = evaluator.objective.∇L(Z⃗, evaluator.trajectory)
    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("objective gradient evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end
    return nothing
end


# constraints and Jacobian

@views function MOI.eval_constraint(
    evaluator::PicoEvaluator,
    g::AbstractVector,
    Z⃗::AbstractVector
)
    if evaluator.verbose
        println("evaluating constraints...")
        time = time_ns()
    end
    # evaluate dynamics constraints
    g[1:evaluator.n_dynamics_constraints] = evaluator.dynamics.F(Z⃗)

    # indexing offset for nonlinear constraints
    offset = evaluator.n_dynamics_constraints

    # TODO: make this in place as well
    for con ∈ evaluator.nonlinear_constraints
        g[offset .+ (1:con.dim)] = con.g(Z⃗)
        offset += con.dim
    end

    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("constraints evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end
    return nothing
end

function MOI.jacobian_structure(evaluator::PicoEvaluator)

    if evaluator.verbose
        println("evaluating jacobian structure...")
        time = time_ns()
    end
    dynamics_structure = evaluator.dynamics.∂F_structure
    row_offset = evaluator.n_dynamics_constraints
    nl_constraint_structure = []
    for con ∈ evaluator.nonlinear_constraints
        con_structure = [(i + row_offset, j) for (i, j) in con.∂g_structure]
        push!(nl_constraint_structure, con_structure)
        row_offset += con.dim
    end
    structure = vcat(dynamics_structure, nl_constraint_structure...)

    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("jacobian structure evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end

    return structure
end

@views function MOI.eval_constraint_jacobian(
    evaluator::PicoEvaluator,
    J::AbstractVector,
    Z⃗::AbstractVector
)
    if evaluator.verbose
        println("evaluating constraint jacobian...")
        time = time_ns()
    end

    ∂s_dynamics = evaluator.dynamics.∂F(Z⃗)
    for (k, ∂ₖ) in enumerate(∂s_dynamics)
        J[k] = ∂ₖ
    end
    offset = length(∂s_dynamics)
    for con ∈ evaluator.nonlinear_constraints
        ∂s_con = con.∂g(Z⃗)
        for (k, ∂ₖ) in enumerate(∂s_con)
            J[offset + k] = ∂ₖ
        end
        offset += length(∂s_con)
    end

    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("constraint jacobian evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end
    return nothing
end


# Hessian of the Lagrangian

function MOI.hessian_lagrangian_structure(evaluator::PicoEvaluator)
    if evaluator.verbose
        println("evaluating hessian structure...")
        time = time_ns()
    end
    objective_structure = evaluator.objective.∂²L_structure(evaluator.trajectory)
    dynamics_structure = evaluator.dynamics.μ∂²F_structure
    nl_constraint_structure = [con.μ∂²g_structure for con ∈ evaluator.nonlinear_constraints]
    structure = vcat(objective_structure, dynamics_structure, nl_constraint_structure...)
    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("hessian structure evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end
    return structure
end

@views function MOI.eval_hessian_lagrangian(
    evaluator::PicoEvaluator,
    H::AbstractVector{T},
    Z⃗::AbstractVector{T},
    σ::T,
    μ::AbstractVector{T}
) where T

    if evaluator.verbose
        println("evaluating hessian...")
        time = time_ns()
    end

    σ∂²Ls = σ * evaluator.objective.∂²L(Z⃗, evaluator.trajectory)

    for (k, σ∂²Lₖ) in enumerate(σ∂²Ls)
        H[k] = σ∂²Lₖ
    end

    μ_dynamics = μ[1:evaluator.n_dynamics_constraints]

    μ_offset = evaluator.n_dynamics_constraints

    offset = length(evaluator.objective.∂²L_structure(evaluator.trajectory))

    μ∂²Fs = evaluator.dynamics.μ∂²F(Z⃗, μ_dynamics)

    for (k, μ∂²Fₖ) in enumerate(μ∂²Fs)
        H[offset + k] = μ∂²Fₖ
    end

    offset += length(evaluator.dynamics.μ∂²F_structure)

    for con ∈ evaluator.nonlinear_constraints
        μ_con = μ[μ_offset .+ (1:con.dim)]
        μ∂²gs = con.μ∂²g(Z⃗, μ_con)
        for (k, μ∂²gₖ) in enumerate(μ∂²gs)
            H[offset + k] = μ∂²gₖ
        end
        offset += length(μ∂²gs)
        μ_offset += con.dim
    end

    if evaluator.verbose
        time = (time_ns() - time) / 1e9
        print("hessian evaluated in ")
        printstyled("$time s \n\n";
            bold=true,
            color=:green
        )
    end

    return nothing
end

end
