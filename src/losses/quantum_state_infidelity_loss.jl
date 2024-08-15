export fidelity
export iso_fidelity

export InfidelityLoss

###
### InfidelityLoss
###

@doc raw"""
    fidelity(ψ, ψ_goal)

Calculate the fidelity between two quantum states `ψ` and `ψ_goal`.
"""
function fidelity(
    ψ::AbstractVector, 
    ψ_goal::AbstractVector;
    subspace::AbstractVector{Int}=1:length(ψ)
)
    ψ = ψ[subspace]
    ψ_goal = ψ_goal[subspace]
    return abs2(ψ_goal' * ψ)
end

@doc raw"""
    iso_fidelity(ψ̃, ψ̃_goal)

Calculate the fidelity between two quantum states in their isomorphic form `ψ̃` and `ψ̃_goal`.
"""
function iso_fidelity(
    ψ̃::AbstractVector, 
    ψ̃_goal::AbstractVector;
    subspace::AbstractVector{Int}=1:length(iso_to_ket(ψ̃))
)
    ψ = iso_to_ket(ψ̃)
    ψ_goal = iso_to_ket(ψ̃_goal)
    return fidelity(ψ, ψ_goal, subspace=subspace)
end

"""
    iso_infidelity(ψ̃, ψ̃goal)

Returns the iso_infidelity between two quantum statevectors specified
in the ``\\mathbb{C}^n \\to \\mathbb{R}^{2n}`` isomorphism space.

"""
function iso_infidelity(
    ψ̃::AbstractVector, 
    ψ̃goal::AbstractVector,
    subspace::AbstractVector{Int}=1:length(iso_to_ket(ψ̃))
)
    return abs(1 - iso_fidelity(ψ̃, ψ̃goal, subspace=subspace))
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
        l = ψ̃ -> iso_infidelity(ψ̃, ψ̃_goal)
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