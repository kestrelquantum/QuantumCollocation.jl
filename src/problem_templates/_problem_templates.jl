module ProblemTemplates

using ..DirectSums
using ..Rollouts
using ..TrajectoryInitialization
using ..Losses

using Distributions
using TrajectoryIndexingUtils
using NamedTrajectories
using QuantumCollocationCore
using PiccoloQuantumObjects
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
include("quantum_state_sampling_problem.jl")

include("density_operator_smooth_pulse_problem.jl")

function apply_piccolo_options!(
    J::Objective,
    constraints::AbstractVector{<:AbstractConstraint},
    piccolo_options::PiccoloOptions,
    traj::NamedTrajectory,
    operators::Union{Nothing, AbstractVector{<:AbstractPiccoloOperator}},
    state_names::AbstractVector{Symbol},
    timestep_name::Symbol
)
    # TODO: This should be changed to leakage indices (more general, for states)
    # (also, it should be looped over relevant state names, not broadcast)
    if piccolo_options.leakage_suppression && !isnothing(operators)
        for (operator, state_name) in zip(operators, state_names)
            if operator isa EmbeddedOperator
                leakage_indices = get_iso_vec_leakage_indices(operator)
                J += L1Regularizer!(
                    constraints,
                    state_name,
                    traj;
                    R_value=piccolo_options.R_leakage,
                    indices=leakage_indices,
                    eval_hessian=piccolo_options.eval_hessian
                )
            else
                @warn "leakage_suppression is only supported for embedded operators, ignoring."
            end
        end
    end

    if piccolo_options.free_time
        if piccolo_options.timesteps_all_equal
            push!(
                constraints,
                TimeStepsAllEqualConstraint(timestep_name, traj)
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

function apply_piccolo_options!(
    J::Objective,
    constraints::AbstractVector{<:AbstractConstraint},
    piccolo_options::PiccoloOptions,
    traj::NamedTrajectory,
    operator::Union{Nothing, AbstractPiccoloOperator},
    state_name::Symbol,
    timestep_name::Symbol
)
    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_name))
    ]

    if !isnothing(operator)
        operators = fill(operator, length(state_names))
    else
        operators = nothing
    end

    return apply_piccolo_options!(
        J,
        constraints,
        piccolo_options,
        traj,
        operators,
        state_names,
        timestep_name
    )
end


end
