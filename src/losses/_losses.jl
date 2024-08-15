module Losses

export Loss

using ..QuantumObjectUtils
using ..Isomorphisms
using ..QuantumSystems
using ..StructureUtils

using NamedTrajectories
using TrajectoryIndexingUtils

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Symbolics
using TestItemRunner

# TODO:
# - [ ] Do not reference the Z object in the loss (components only / remove "name")

# ----------------------------------------------------------------------------- #
#                           Abstract Loss                                       #
# ----------------------------------------------------------------------------- #

abstract type AbstractLoss end

include("_experimental_loss_functions.jl")
include("quantum_loss.jl")
include("quantum_state_infidelity_loss.jl")
include("unitary_trace_loss.jl")
include("unitary_infidelity_loss.jl")

# ----------------------------------------------------------------------------- #
#                               Loss                                            #
# ----------------------------------------------------------------------------- #

struct Loss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function Loss(
        Z::NamedTrajectory,
        J::Function,
        x::Symbol
    )
        @assert x ∈ Z.names
        @assert Z.goal[x] isa AbstractVector

        x_goal = Z.goal[x]

        J = x̄ -> J(x̄, x_goal)
        ∇J = x̄ -> ForwardDiff.gradient(J, x̄)

        Symbolics.@variables x̄[1:Z.dims[x]]
        x̄ = collect(x̄)

        ∇²J_symbolic = Symbolics.sparsehessian(J(x̄), x̄)
        rows, cols, _ = findnz(∇²J_symbolic)
        rowcols = collect(zip(rows, cols))
        filter!((row, col) -> row ≥ col, rowcols)
        ∇²J_structure = rowcols

        ∇²J_expression = Symbolics.build_function(∇²J_symbolic, x̄)
        ∇²J = eval(∇²J_expression[1])

        return new(J, ∇J, ∇²J, ∇²J_structure, x)
    end
end

function (loss::Loss)(Z::NamedTrajectory; gradient=false, hessian=false)
    @assert !(gradient && hessian)
    if !(gradient || hessian)
        return loss.l(Z[end][loss.name])
    elseif gradient
        return loss.∇l(Z[end][loss.name])
    elseif hessian
        return loss.∇²l(Z[end][loss.name])
    end
end


end
