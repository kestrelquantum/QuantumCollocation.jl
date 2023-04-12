module Evaluators

export PicoEvaluator

using ..QuantumSystems
using ..Integrators
using ..Dynamics
using ..Objectives
using ..Constraints

using NamedTrajectories
using MathOptInterface
const MOI = MathOptInterface

mutable struct PicoEvaluator <: MOI.AbstractNLPEvaluator
    trajectory::NamedTrajectory
    objective::Objective
    dynamics::QuantumDynamics
    n_dynamics_constraints::Int
    nonlinear_constraints::Vector{<:NonlinearConstraint}
    n_nonlinear_constraints::Int
    eval_hessian::Bool

    function PicoEvaluator(
        trajectory::NamedTrajectory,
        objective::Objective,
        dynamics::QuantumDynamics,
        nonlinear_constraints::Vector{<:NonlinearConstraint},
        eval_hessian::Bool
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
            eval_hessian
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
    return evaluator.objective.L(Z⃗, evaluator.trajectory)
end

@views function MOI.eval_objective_gradient(
    evaluator::PicoEvaluator,
    ∇::AbstractVector,
    Z⃗::AbstractVector
)
    ∇ .= evaluator.objective.∇L(Z⃗, evaluator.trajectory)
    return nothing
end


# constraints and Jacobian

@views function MOI.eval_constraint(
    evaluator::PicoEvaluator,
    g::AbstractVector,
    Z⃗::AbstractVector
)
    g[1:evaluator.n_dynamics_constraints] .= evaluator.dynamics.F(Z⃗)
    offset = evaluator.n_dynamics_constraints
    for con ∈ evaluator.nonlinear_constraints
        g[offset .+ (1:con.dim)] .= con.g(Z⃗)
        offset += con.dim
    end
    return nothing
end

function MOI.jacobian_structure(evaluator::PicoEvaluator)
    return evaluator.dynamics.∂F_structure
end

@views function MOI.eval_constraint_jacobian(
    evaluator::PicoEvaluator,
    J::AbstractVector,
    Z⃗::AbstractVector
)
    ∂s = evaluator.dynamics.∂F(Z⃗)
    for (k, ∇ₖ) in enumerate(∂s)
        J[k] = ∇ₖ
    end
    return nothing
end


# Hessian of the Lagrangian

function MOI.hessian_lagrangian_structure(evaluator::PicoEvaluator)
    structure = vcat(
        evaluator.objective.∂²L_structure(evaluator.trajectory),
        evaluator.dynamics.μ∂²F_structure
    )
    return structure
end

@views function MOI.eval_hessian_lagrangian(
    evaluator::PicoEvaluator,
    H::AbstractVector{T},
    Z⃗::AbstractVector{T},
    σ::T,
    μ::AbstractVector{T}
) where T
    σ∂²Ls = σ * evaluator.objective.∂²L(Z⃗, evaluator.trajectory)

    for (k, σ∂²Lₖ) in enumerate(σ∂²Ls)
        H[k] = σ∂²Lₖ
    end

    μ∂²Fs = evaluator.dynamics.μ∂²F(Z⃗, μ)

    offset = length(evaluator.objective.∂²L_structure(evaluator.trajectory))

    for (k, μ∂²Fₖ) in enumerate(μ∂²Fs)
        H[offset + k] = μ∂²Fₖ
    end

    return nothing
end

end
