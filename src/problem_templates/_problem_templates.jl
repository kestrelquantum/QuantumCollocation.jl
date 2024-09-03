module ProblemTemplates

using ..QuantumSystems
using ..Isomorphisms
using ..EmbeddedOperators
using ..DirectSums
using ..Rollouts
using ..TrajectoryInitialization
using ..Objectives
using ..Constraints
using ..Losses
using ..Integrators
using ..Problems
using ..Options

using Distributions
using TrajectoryIndexingUtils
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using ExponentialAction
using JLD2

using TestItemRunner

include("unitary_smooth_pulse_problem.jl")
include("unitary_minimum_time_problem.jl")
include("unitary_robustness_problem.jl")
include("unitary_direct_sum_problem.jl")
include("unitary_sampling_problem.jl")
include("unitary_bang_bang_problem.jl")

include("quantum_state_smooth_pulse_problem.jl")
include("quantum_state_minimum_time_problem.jl")


function apply_piccolo_options!(
    J::Objective,
    constraints::AbstractVector{<:AbstractConstraint},
    piccolo_options::PiccoloOptions,
    traj::NamedTrajectory;
    operator::Union{Nothing, OperatorType}=nothing
)
    if piccolo_options.leakage_suppression
        state_names = [
            name for name ∈ traj.names
                if startswith(string(name), string(piccolo_options.state_name))
        ]

        if operator isa EmbeddedOperator
            leakage_indices = get_iso_vec_leakage_indices(operator)
            for state_name in state_names
                J += L1Regularizer!(
                    constraints,
                    state_name,
                    traj;
                    R_value=piccolo_options.R_leakage,
                    indices=leakage_indices,
                    eval_hessian=piccolo_options.eval_hessian
                )
            end
        else
            @warn "leakage_suppression is only supported for embedded operators, ignoring."
        end
    end

    if piccolo_options.free_time
        if piccolo_options.timesteps_all_equal
            push!(
                constraints,
                TimeStepsAllEqualConstraint(piccolo_options.timestep_name, traj)
            )
        end
    end

    if !isnothing(piccolo_options.complex_control_norm_constraint_name)
        norm_con = ComplexModulusContraint(
            piccolo_options.complex_control_norm_constraint_name,
            piccolo_options.complex_control_norm_constraint_radius,
            traj;
        )
        push!(constraints, norm_con)
    end

    return
end


end
