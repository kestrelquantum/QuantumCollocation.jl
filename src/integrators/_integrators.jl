module Integrators

export AbstractIntegrator

export QuantumIntegrator

export QuantumPadeIntegrator
export QuantumStatePadeIntegrator
export UnitaryPadeIntegrator
export UnitaryExponentialIntegrator
export QuantumStateExponentialIntegrator

export DerivativeIntegrator

export G_bilinear

export jacobian
export hessian_of_the_lagrangian

export get_comps

export nth_order_pade
export fourth_order_pade
export sixth_order_pade
export eighth_order_pade
export tenth_order_pade

using ..Isomorphisms
# using ..QuantumSystems
# using ..QuantumObjectUtils
# using ..QuantumSystemUtils

using NamedTrajectories
using TrajectoryIndexingUtils
using LinearAlgebra
using SparseArrays
using ForwardDiff
using TestItemRunner


abstract type AbstractIntegrator end

abstract type QuantumIntegrator <: AbstractIntegrator end

function comps(P::AbstractIntegrator, traj::NamedTrajectory)
    state_comps = traj.components[state(P)]
    u = controls(P)
    if u isa Tuple
        control_comps = Tuple(traj.components[uᵢ] for uᵢ ∈ u)
    else
        control_comps = traj.components[u]
    end
    if traj.timestep isa Float64
        return state_comps, control_comps
    else
        timestep_comp = traj.components[traj.timestep]
        return state_comps, control_comps, timestep_comp
    end
end



include("_integrator_utils.jl")
include("derivative_integrator.jl")
include("pade_integrators.jl")
include("exponential_integrators.jl")


end
