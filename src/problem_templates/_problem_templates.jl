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
using TestItems

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
    state_names::AbstractVector{Symbol},
    timestep_name::Symbol;
    state_leakage_indices::Union{Nothing, AbstractVector{<:AbstractVector{Int}}}=nothing,
)
    if piccolo_options.leakage_suppression
        if isnothing(state_leakage_indices)
            error("You must provide leakage indices for leakage suppression.")
        end
        for (state_name, leakage_indices) ∈ zip(state_names, state_leakage_indices)
            J += L1Regularizer!(
                constraints,
                state_name,
                traj;
                R_value=piccolo_options.R_leakage,
                indices=leakage_indices,
                eval_hessian=piccolo_options.eval_hessian
            )
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
    state_name::Symbol,
    timestep_name::Symbol;
    state_leakage_indices::Union{Nothing, AbstractVector{Int}}=nothing,
)
    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_name))
    ]

    return apply_piccolo_options!(
        J,
        constraints,
        piccolo_options,
        traj,
        state_names,
        timestep_name;
        state_leakage_indices=isnothing(state_leakage_indices) ? nothing : fill(state_leakage_indices, length(state_names)),
    )
end


end
