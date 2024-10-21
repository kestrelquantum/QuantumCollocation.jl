module Constraints

export AbstractConstraint

export LinearConstraint

export constrain!

export NonlinearConstraint
export NonlinearEqualityConstraint
export NonlinearInequalityConstraint

using ..Losses
using ..Isomorphisms
using ..StructureUtils
using ..Options

using TrajectoryIndexingUtils
using NamedTrajectories
using ForwardDiff
using SparseArrays
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# TODO:
# - [ ] Do not reference the Z object in the constraint (components only / remove "name")

# ----------------------------------------------------------------------------- #
#                     Abstract Constraints                                      #
# ----------------------------------------------------------------------------- #

abstract type AbstractConstraint end
abstract type LinearConstraint <: AbstractConstraint end
abstract type NonlinearConstraint <: AbstractConstraint end

include("linear_trajectory_constraints.jl")
include("complex_modulus_constraint.jl")
include("fidelity_constraint.jl")
include("l1_slack_constraint.jl")


# ----------------------------------------------------------------------------- #
#                     Linear Constraint                                         #
# ----------------------------------------------------------------------------- #

"""
    constrain!(opt::Ipopt.Optimizer, vars::Vector{MOI.VariableIndex}, cons::Vector{LinearConstraint}, traj::NamedTrajectory; verbose=false)

Supplies a set of LinearConstraints to  IPOPT using MathOptInterface

"""
function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{LinearConstraint},
    traj::NamedTrajectory;
    verbose=false
)
    for con in cons
        if verbose
            println("applying constraint: ", con.label)
        end
        con(opt, vars, traj)
    end
end


# ----------------------------------------------------------------------------- #
#                     Nonlinear Constraint                                      #
# ----------------------------------------------------------------------------- #


function NonlinearConstraint(params::Dict)
    return eval(params[:type])(; delete!(params, :type)...)
end

"""
    struct NonlinearEqualityConstraint

Represents a nonlinear equality constraint.

# Fields
- `g::Function`: the constraint function
- `∂g::Function`: the Jacobian of the constraint function
- `∂g_structure::Vector{Tuple{Int, Int}}`: the structure of the Jacobian
   i.e. all non-zero entries
- `μ∂²g::Function`: the Hessian of the constraint function
- `μ∂²g_structure::Vector{Tuple{Int, Int}}`: the structure of the Hessian
- `dim::Int`: the dimension of the constraint function
- `params::Dict{Symbol, Any}`: a dictionary of parameters

"""
struct NonlinearEqualityConstraint <: NonlinearConstraint
    g::Function
    ∂g::Function
    ∂g_structure::Vector{Tuple{Int, Int}}
    μ∂²g::Union{Nothing, Function}
    μ∂²g_structure::Union{Nothing, Vector{Tuple{Int, Int}}}
    dim::Int
    params::Dict{Symbol, Any}
end

"""
    struct NonlinearInequalityConstraint

Represents a nonlinear inequality constraint.

# Fields
- `g::Function`: the constraint function
- `∂g::Function`: the Jacobian of the constraint function
- `∂g_structure::Vector{Tuple{Int, Int}}`: the structure of the Jacobian
   i.e. all non-zero entries
- `μ∂²g::Function`: the Hessian of the constraint function
- `μ∂²g_structure::Vector{Tuple{Int, Int}}`: the structure of the Hessian
- `dim::Int`: the dimension of the constraint function
- `params::Dict{Symbol, Any}`: a dictionary of parameters containing additional
   information about the constraint

"""
struct NonlinearInequalityConstraint <: NonlinearConstraint
    g::Function
    ∂g::Function
    ∂g_structure::Vector{Tuple{Int, Int}}
    μ∂²g::Union{Nothing, Function}
    μ∂²g_structure::Union{Nothing, Vector{Tuple{Int, Int}}}
    dim::Int
    params::Dict{Symbol, Any}
end



end
