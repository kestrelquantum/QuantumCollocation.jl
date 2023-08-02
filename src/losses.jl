module Losses

export Loss

export QuantumLoss
export QuantumLossGradient
export QuantumLossHessian

export InfidelityLoss
export UnitaryInfidelityLoss
export UnitaryTraceLoss

export structure

export infidelity
export unitary_infidelity

using ..QuantumUtils
using ..QuantumSystems
using ..StructureUtils


using NamedTrajectories

using TrajectoryIndexingUtils
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Symbolics

#
# loss functions
#


# TODO: renormalize vectors in place of abs
#       ⋅ penalize loss to remain near unit norm
#       ⋅ Σ α * (1 - ψ̃'ψ̃), α = 1e-3

abstract type AbstractLoss end

"""
    infidelity(ψ̃::AbstractVector, ψ̃goal::AbstractVector)

Returns the infidelity between two quantum statevectors specified 
in the ``\\mathbb{C}^n \\to \\mathbb{R}^{2n}`` isomorphism space.

"""
function infidelity(ψ̃::AbstractVector, ψ̃goal::AbstractVector)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    return abs(1 - abs2(ψgoal'ψ))
end

"""
    isovec_unitary_fidelity(Ũ::AbstractVector, Ũgoal::AbstractVector)

Returns the fidelity between two unitary operators, specified as an 
isomorphic vector. 

"""
@inline @views function isovec_unitary_fidelity(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    n = Int(sqrt(length(Ũ⃗) ÷ 2))
    U⃗ᵣ = Ũ⃗[1:end ÷ 2]
    U⃗ᵢ = Ũ⃗[end ÷ 2 + 1:end]
    Ū⃗ᵣ = Ũ⃗_goal[1:end ÷ 2]
    Ū⃗ᵢ = Ũ⃗_goal[end ÷ 2 + 1:end]
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    return 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
end

@views function unitary_infidelity(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    ℱ = isovec_unitary_fidelity(Ũ⃗, Ũ⃗_goal)
    return abs(1 - ℱ)
end

@views function unitary_infidelity_gradient(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    n = Int(sqrt(length(Ũ⃗) ÷ 2))
    U⃗ᵣ = Ũ⃗[1:end ÷ 2]
    U⃗ᵢ = Ũ⃗[end ÷ 2 + 1:end]
    Ū⃗ᵣ = Ũ⃗_goal[1:end ÷ 2]
    Ū⃗ᵢ = Ũ⃗_goal[end ÷ 2 + 1:end]
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    ℱ = 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
    ∇ᵣℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵣ - Tᵢ * Ū⃗ᵢ)
    ∇ᵢℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵢ + Tᵢ * Ū⃗ᵣ)
    ∇ℱ = [∇ᵣℱ; ∇ᵢℱ]
    return -sign(1 - ℱ) * ∇ℱ
end

@views function unitary_infidelity_hessian(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    n = Int(sqrt(length(Ũ⃗) ÷ 2))
    U⃗ᵣ = Ũ⃗[1:end ÷ 2]
    U⃗ᵢ = Ũ⃗[end ÷ 2 + 1:end]
    Ū⃗ᵣ = Ũ⃗_goal[1:end ÷ 2]
    Ū⃗ᵢ = Ũ⃗_goal[end ÷ 2 + 1:end]
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    Wᵣᵣ = Ū⃗ᵣ * Ū⃗ᵣ'
    Wᵢᵢ = Ū⃗ᵢ * Ū⃗ᵢ'
    Wᵣᵢ = Ū⃗ᵣ * Ū⃗ᵢ'
    Wᵢᵣ = Wᵣᵢ'
    ℱ = 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
    ∇ᵣℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵣ - Tᵢ * Ū⃗ᵢ)
    ∇ᵢℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵢ + Tᵢ * Ū⃗ᵣ)
    ∂ᵣ²ℱ = 1 / ℱ * (-∇ᵣℱ * ∇ᵣℱ' + 1 / n^2 * (Wᵣᵣ + Wᵢᵢ))
    ∂ᵢ²ℱ = 1 / ℱ * (-∇ᵢℱ * ∇ᵢℱ' + 1 / n^2 * (Wᵣᵣ + Wᵢᵢ))
    ∂ᵣ∂ᵢℱ = 1 / ℱ * (-∇ᵢℱ * ∇ᵣℱ' + 1 / n^2 * (Wᵢᵣ - Wᵣᵢ))
    ∂²ℱ = [∂ᵣ²ℱ ∂ᵣ∂ᵢℱ; ∂ᵣ∂ᵢℱ' ∂ᵢ²ℱ]
    return -sign(1 - ℱ) * ∂²ℱ
end



struct UnitaryInfidelityLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function UnitaryInfidelityLoss(
        name::Symbol,
        Ũ⃗_goal::AbstractVector
    )
        l = Ũ⃗ -> unitary_infidelity(Ũ⃗, Ũ⃗_goal)
        ∇l = Ũ⃗ -> unitary_infidelity_gradient(Ũ⃗, Ũ⃗_goal)
        ∇²l = Ũ⃗ -> unitary_infidelity_hessian(Ũ⃗, Ũ⃗_goal)
        Ũ⃗_dim = length(Ũ⃗_goal)
        ∇²l_structure = []
        for (i, j) ∈ Iterators.product(1:Ũ⃗_dim, 1:Ũ⃗_dim)
            if i ≤ j
                push!(∇²l_structure, (i, j))
            end
        end
        return new(l, ∇l, ∇²l, ∇²l_structure, name)
    end
end

function (loss::UnitaryInfidelityLoss)(
    Ũ⃗_end::AbstractVector{<:Real};
    gradient=false,
    hessian=false
)
    @assert !(gradient && hessian)
    if !(gradient || hessian)
        return loss.l(Ũ⃗_end)
    elseif gradient
        return loss.∇l(Ũ⃗_end)
    elseif hessian
        return loss.∇²l(Ũ⃗_end)
    end
end







function unitary_trace_loss(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    U = iso_vec_to_operator(Ũ⃗)
    Ugoal = iso_vec_to_operator(Ũ⃗_goal)
    return 1 / 2 * tr(sqrt((U - Ugoal)' * (U - Ugoal)))
end


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


struct UnitaryTraceLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function UnitaryTraceLoss(
        name::Symbol,
        Ũ⃗_goal::AbstractVector
    )
        l = Ũ⃗ -> unitary_trace_loss(Ũ⃗, Ũ⃗_goal)
        ∇l = Ũ⃗ -> ForwardDiff.gradient(l, Ũ⃗)
        ∇²l = Ũ⃗ -> ForwardDiff.hessian(l, Ũ⃗)
        Ũ⃗_dim = length(Ũ⃗_goal)
        ∇²l_structure = []
        for (i, j) ∈ Iterators.product(1:Ũ⃗_dim, 1:Ũ⃗_dim)
            if i ≤ j
                push!(∇²l_structure, (i, j))
            end
        end
        return new(l, ∇l, ∇²l, ∇²l_structure, name)
    end
end

function (loss::UnitaryTraceLoss)(
    Ũ⃗_end::AbstractVector{<:Real};
    gradient=false,
    hessian=false
)
    @assert !(gradient && hessian)

    if !(gradient || hessian)
        return loss.l(Ũ⃗_end)
    elseif gradient
        return loss.∇l(Ũ⃗_end)
    elseif hessian
        return loss.∇²l(Ũ⃗_end)
    end
end





struct InfidelityLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    wfn_name::Symbol

    function InfidelityLoss(
        name::Symbol,
        ψ̃_goal::AbstractVector
    )
        l = ψ̃ -> infidelity(ψ̃, ψ̃_goal)
        ∇l = ψ̃ -> ForwardDiff.gradient(l, ψ̃)

        Symbolics.@variables ψ̃[1:length(ψ̃_goal)]
        ψ̃ = collect(ψ̃)

        ∇²l_symbolic = Symbolics.sparsehessian(l(ψ̃), ψ̃)
        K, J, _ = findnz(∇²l_symbolic)
        kjs = collect(zip(K, J))
        filter!(((k, j),) -> k ≤ j, kjs)
        ∇²l_structure = kjs

        ∇²l_expression = Symbolics.build_function(∇²l_symbolic, ψ̃)
        ∇²l = eval(∇²l_expression[1])

        return new(l, ∇l, ∇²l, ∇²l_structure, name)
    end
end

function (loss::InfidelityLoss)(
    ψ̃_end::AbstractVector{<:Real};
    gradient=false,
    hessian=false
)
    @assert !(gradient && hessian)

    if !(gradient || hessian)
        return loss.l(ψ̃_end)
    elseif gradient
        return loss.∇l(ψ̃_end)
    elseif hessian
        return loss.∇²l(ψ̃_end)
    end
end








struct QuantumLoss
    cs::Vector{Function}
    isodim::Int

    function QuantumLoss(
        sys::AbstractSystem,
        loss::Symbol = :infidelity_loss
    )
        if loss == :energy_loss
            cs = [ψ̃ⁱ -> eval(loss)(ψ̃ⁱ, sys.H_target) for i = 1:sys.nqstates]
        elseif loss == :neg_entropy_loss
            cs = [ψ̃ⁱ -> eval(loss)(ψ̃ⁱ) for i = 1:sys.nqstates]
        else
            cs = [
                ψ̃ⁱ -> eval(loss)(
                    ψ̃ⁱ,
                    sys.ψ̃goal[slice(i, sys.isodim)]
                ) for i = 1:sys.nqstates
            ]
        end
        return new(cs, sys.isodim)
    end
end

function (qloss::QuantumLoss)(ψ̃::AbstractVector)
    loss = 0.0
    for (i, cⁱ) in enumerate(qloss.cs)
        loss += cⁱ(ψ̃[slice(i, qloss.isodim)])
    end
    return loss
end

struct QuantumLossGradient
    ∇cs::Vector{Function}
    isodim::Int

    function QuantumLossGradient(
        loss::QuantumLoss;
        simplify=true
    )
        Symbolics.@variables ψ̃[1:loss.isodim]

        ψ̃ = collect(ψ̃)

        ∇cs_symbs = [
            Symbolics.gradient(c(ψ̃), ψ̃; simplify=simplify)
                for c in loss.cs
        ]

        ∇cs_exprs = [
            Symbolics.build_function(∇c, ψ̃)
                for ∇c in ∇cs_symbs
        ]

        ∇cs = [
            eval(∇c_expr[1])
                for ∇c_expr in ∇cs_exprs
        ]

        return new(∇cs, loss.isodim)
    end
end

@views function (∇c::QuantumLossGradient)(
    ψ̃::AbstractVector
)
    ∇ = similar(ψ̃)

    for (i, ∇cⁱ) in enumerate(∇c.∇cs)

        ψ̃ⁱ_slice = slice(i, ∇c.isodim)

        ∇[ψ̃ⁱ_slice] = ∇cⁱ(ψ̃[ψ̃ⁱ_slice])
    end

    return ∇
end

struct QuantumLossHessian
    ∇²cs::Vector{Function}
    ∇²c_structures::Vector{Vector{Tuple{Int, Int}}}
    isodim::Int

    function QuantumLossHessian(
        loss::QuantumLoss;
        simplify=true
    )

        Symbolics.@variables ψ̃[1:loss.isodim]
        ψ̃ = collect(ψ̃)

        ∇²c_symbs = [
            Symbolics.sparsehessian(
                c(ψ̃),
                ψ̃;
                simplify=simplify
            ) for c in loss.cs
        ]

        ∇²c_structures = []

        for ∇²c_symb in ∇²c_symbs
            K, J, _ = findnz(∇²c_symb)

            KJ = collect(zip(K, J))

            filter!(((k, j),) -> k ≤ j, KJ)

            push!(∇²c_structures, KJ)
        end

        ∇²c_exprs = [
            Symbolics.build_function(∇²c_symb, ψ̃)
                for ∇²c_symb in ∇²c_symbs
        ]

        ∇²cs = [
            eval(∇²c_expr[1])
                for ∇²c_expr in ∇²c_exprs
        ]

        return new(∇²cs, ∇²c_structures, loss.isodim)
    end
end

function structure(
    H::QuantumLossHessian,
    T::Int,
    vardim::Int
)
    H_structure = []

    T_offset = index(T, 0, vardim)

    for (i, KJⁱ) in enumerate(H.∇²c_structures)

        i_offset = index(i, 0, H.isodim)

        for kj in KJⁱ
            push!(H_structure, (T_offset + i_offset) .+ kj)
        end
    end

    return H_structure
end

@views function (H::QuantumLossHessian)(ψ̃::AbstractVector)

    Hs = []

    for (i, ∇²cⁱ) in enumerate(H.∇²cs)

        ψ̃ⁱ = ψ̃[slice(i, H.isodim)]

        for (k, j) in H.∇²c_structures[i]

            Hⁱᵏʲ = ∇²cⁱ(ψ̃ⁱ)[k, j]

            append!(Hs, Hⁱᵏʲ)
        end
    end

    return Hs
end


#
# primary loss functions
#


function energy_loss(
    ψ̃::AbstractVector,
    H::AbstractMatrix
)
    ψ = iso_to_ket(ψ̃)
    return real(ψ' * H * ψ)
end


# TODO: figure out a way to implement this without erroring and Von Neumann entropy being always 0 for a pure state
function neg_entropy_loss(
    ψ̃::AbstractVector
)
    ψ = iso_to_ket(ψ̃)
    ρ = ψ * ψ'
    ρ = Hermitian(ρ)
    return tr(ρ * log(ρ))
end




#
# experimental loss functions
#

function pure_real_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    return -(ψ'ψgoal)
end

function geodesic_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - amp), abs(1 + amp))
end

function real_loss(ψ̃, ψ̃goal)
    ψ = iso_to_ket(ψ̃)
    ψgoal = iso_to_ket(ψ̃goal)
    amp = ψ'ψgoal
    return min(abs(1 - real(amp)), abs(1 + real(amp)))
end

function iso_infidelity(ψ̃, ψ̃f)
    ψ = iso_to_ket(ψ̃)
    ψf = iso_to_ket(ψ̃f)
    return 1 - abs2(ψ'ψf)
end


function quaternionic_loss(ψ̃, ψ̃goal)
    return min(
        abs(1 - dot(ψ̃, ψ̃goal)),
        abs(1 + dot(ψ̃, ψ̃goal))
    )

end

end
