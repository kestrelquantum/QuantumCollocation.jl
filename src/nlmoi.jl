module NLMOI

using ..Evaluators

using MathOptInterface
const MOI = MathOptInterface


MOI.initialize(::PicoEvaluator, features) = nothing


function MOI.features_available(evaluator::PicoEvaluator)
    if evaluator.eval_hessian
        return [:Grad, :Jac, :Hess]
    else
        return [:Grad, :Jac]
    end
end


# objective and gradient

function MOI.eval_objective(
    evaluator::PicoEvaluator,
    Z::AbstractVector
)
    return evaluator.objective.L(Z)
end

function MOI.eval_objective_gradient(
    evaluator::PicoEvaluator,
    ∇::AbstractVector,
    Z::AbstractVector
)
    ∇ .= evaluator.objective.∇L(Z)
    return nothing
end


# constraints and Jacobian

function MOI.eval_constraint(
    evaluator::PicoEvaluator,
    g::AbstractVector,
    Z::AbstractVector
)
    g .= evaluator.dynamics.F(Z)
    return nothing
end

function MOI.jacobian_structure(evaluator::PicoEvaluator)
    return evaluator.dynamics.∂F_structure
end

function MOI.eval_constraint_jacobian(
    evaluator::PicoEvaluator,
    J::AbstractVector,
    Z::AbstractVector
)

    ∂s = evaluator.dynamics.∂F(Z)
    for (k, ∇ₖ) in enumerate(∂s)
        J[k] = ∇ₖ
    end
    return nothing
end


# Hessian of the Lagrangian

function MOI.hessian_lagrangian_structure(
    evaluator::PicoEvaluator
)
    structure = vcat(
        evaluator.objective.∂²L_structure,
        evaluator.dynamics.μ∂²F_structure
    )
    return structure
end

function MOI.eval_hessian_lagrangian(
    evaluator::PicoEvaluator,
    H::AbstractVector{T},
    Z::AbstractVector{T},
    σ::T,
    μ::AbstractVector{T}
) where T

    σ∂²Ls = σ * evaluator.objective.∂²L(Z)

    for (k, σ∂²Lₖ) in enumerate(σ∂²Ls)
        H[k] = σ∂²Lₖ
    end

    μ∂²Fs = evaluator.dynamics.μ∂²F(μ, Z)

    offset = length(evaluator.objective.∂²L_structure)

    for (k, μ∂²Fₖ) in enumerate(μ∂²Fs)
        H[offset + k] = μ∂²Fₖ
    end

    return nothing
end

end
