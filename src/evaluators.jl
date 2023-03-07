module Evaluators

export PicoEvaluator

using ..QuantumSystems
using ..Integrators
using ..Dynamics
using ..Objectives

using MathOptInterface
const MOI = MathOptInterface

struct PicoEvaluator <: MOI.AbstractNLPEvaluator
    objective::Objective
    dynamics::QuantumDynamics
    eval_hessian::Bool
end

end
