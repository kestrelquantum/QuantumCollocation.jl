module ProblemTemplates

export UnitarySmoothPulseProblem
export UnitaryMinimumTimeProblem
export UnitaryRobustnessProblem

export QuantumStateSmoothPulseProblem
export QuantumStateMinimumTimeProblem

using ..QuantumSystems
using ..QuantumUtils
using ..EmbeddedOperators
using ..Rollouts
using ..TrajectoryInitialization
using ..Objectives
using ..Constraints
using ..Integrators
using ..Problems
using ..IpoptOptions

using NamedTrajectories
using LinearAlgebra
using Distributions
using JLD2

include("unitary_smooth_pulse_problem.jl")
include("unitary_minimum_time_problem.jl")
include("unitary_robustness_problem.jl")

include("quantum_state_smooth_pulse_problem.jl")
include("quantum_state_minimum_time_problem.jl")

end
