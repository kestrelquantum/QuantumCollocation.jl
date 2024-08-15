module Objectives

export Objective
export NullObjective

using ..Isomorphisms
using ..QuantumSystems
using ..EmbeddedOperators
using ..Losses
using ..Constraints

using TrajectoryIndexingUtils
using NamedTrajectories

using LinearAlgebra
using SparseArrays
using Symbolics
using TestItemRunner

include("quantum_objective.jl")
include("unitary_infidelity_objective.jl")
include("regularizer_objective.jl")
include("minimum_time_objective.jl")
include("unitary_robustness_objective.jl")

# TODO:
# - [ ] Do not reference the Z object in the objective (components only / remove "name")

"""
    sparse_to_moi(A::SparseMatrixCSC)

Converts a sparse matrix to tuple of vector of nonzero indices and vector of nonzero values
"""
function sparse_to_moi(A::SparseMatrixCSC)
    inds = collect(zip(findnz(A)...))
    vals = [A[i,j] for (i,j) ∈ inds]
    return (inds, vals)
end

# ----------------------------------------------------------------------------- #
#                           Objective                                           #
# ----------------------------------------------------------------------------- #

"""
    Objective

A structure for defining objective functions.

The `terms` field contains all the arguments needed to construct the objective function.

Fields:
    `L`: the objective function
    `∇L`: the gradient of the objective function
    `∂²L`: the Hessian of the objective function
    `∂²L_structure`: the structure of the Hessian of the objective function
    `terms`: a vector of dictionaries containing the terms of the objective function
"""
struct Objective
	L::Function
	∇L::Function
	∂²L::Union{Function, Nothing}
	∂²L_structure::Union{Function, Nothing}
    terms::Vector{Dict}
end

function Base.:+(obj1::Objective, obj2::Objective)
	L = (Z⃗, Z) -> obj1.L(Z⃗, Z) + obj2.L(Z⃗, Z)
	∇L = (Z⃗, Z) -> obj1.∇L(Z⃗, Z) + obj2.∇L(Z⃗, Z)
	if isnothing(obj1.∂²L) && isnothing(obj2.∂²L)
		∂²L = Nothing
		∂²L_structure = Nothing
	elseif isnothing(obj1.∂²L)
		∂²L = (Z⃗, Z) -> obj2.∂²L(Z⃗, Z)
		∂²L_structure = obj2.∂²L_structure
	elseif isnothing(obj2.∂²L)
		∂²L = (Z⃗, Z) -> obj1.∂²L(Z⃗, Z)
		∂²L_structure = obj1.∂²L_structure
	else
		∂²L = (Z⃗, Z) -> vcat(obj1.∂²L(Z⃗, Z), obj2.∂²L(Z⃗, Z))
		∂²L_structure = Z -> vcat(obj1.∂²L_structure(Z), obj2.∂²L_structure(Z))
	end
    terms = vcat(obj1.terms, obj2.terms)
	return Objective(L, ∇L, ∂²L, ∂²L_structure, terms)
end

Base.:+(obj::Objective, ::Nothing) = obj
Base.:+(obj::Objective) = obj

function Objective(terms::AbstractVector{<:Dict})
    return +(Objective.(terms)...)
end

function Base.:*(num::Real, obj::Objective)
	L = (Z⃗, Z) -> num * obj.L(Z⃗, Z)
	∇L = (Z⃗, Z) -> num * obj.∇L(Z⃗, Z)
    if isnothing(obj.∂²L)
        ∂²L = nothing
        ∂²L_structure = nothing
    else
        ∂²L = (Z⃗, Z) -> num * obj.∂²L(Z⃗, Z)
        ∂²L_structure = obj.∂²L_structure
    end
	return Objective(L, ∇L, ∂²L, ∂²L_structure, obj.terms)
end

Base.:*(obj::Objective, num::Real) = num * obj

function Objective(term::Dict)
    return eval(term[:type])(; delete!(term, :type)...)
end

# ----------------------------------------------------------------------------- #
#                           Null objective                                      #
# ----------------------------------------------------------------------------- #

function NullObjective()
    params = Dict(:type => :NullObjective)
	L(Z⃗::AbstractVector{R}, Z::NamedTrajectory) where R<:Real = 0.0
    ∇L(Z⃗::AbstractVector{R}, Z::NamedTrajectory) where R<:Real = zeros(R, Z.dim * Z.T + Z.global_dim)
    ∂²L_structure(Z::NamedTrajectory) = []
    function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory; return_moi_vals=true)
        n = Z.dim * Z.T + Z.global_dim
        return return_moi_vals ? [] : spzeros(n, n)
    end
	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

end
