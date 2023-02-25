module QuantumUtils

export GATES
export ⊗
export apply
export ket_to_iso
export iso_to_ket
export annihilate
export create
export quad
export cavity_state
export number
export normalize
export fidelity
export infidelity
export population
export populations

using LinearAlgebra


"""
    kronicker product utility
"""

⊗(A::AbstractVecOrMat, B::AbstractVecOrMat) = kron(A, B)


"""
    quantum gates
"""

const GATES = Dict(
    :X => [0 1;
           1 0],

    :Y => [0 -im;
           im 0],

    :Z => [1 0;
           0 -1],

    :H => [1 1;
           1 -1]/√2,

    :CX => [1 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0],

    :XI => [0 0 -im 0;
            0 0 0 -im;
            -im 0 0 0;
            0 -im 0 0],

    :sqrtiSWAP => [1 0 0 0;
                   0 1/sqrt(2) 1im/sqrt(2) 0;
                   0 1im/sqrt(2) 1/sqrt(2) 0;
                   0 0 0 1]
)

gate(U::Symbol) = GATES[U]

function apply(U::Symbol, ψ::Vector{<:Number})
    @assert norm(ψ) ≈ 1.0
    @assert gate in keys(GATES) "gate not found"
    Û = gate(gate)
    @assert size(Û, 2) == size(ψ, 1) "gate size does not match ket dim"
    return ComplexF64.(normalize(Û * ψ))
end


"""
    quantum harmonic oscillator operators
"""

function annihilate(levels::Int)
    return diagm(1 => map(sqrt, 1:levels - 1))
end

function create(levels::Int)
    return (annihilate(levels))'
end

function number(levels::Int)
    return create(levels) * annihilate(levels)
end

function quad(levels::Int)
    return number(levels) * (number(levels) - I(levels))
end

function cavity_state(level::Int, cavity_levels::Int)
    state = zeros(ComplexF64, cavity_levels)
    state[level + 1] = 1.
    return state
end

"""
    multimode system utilities
"""

function multimode_state(ψ::String, transmon_levels::Int, cavity_levels::Int)
    @assert length(ψ) == 2

    @assert transmon_levels ∈ 2:4

    transmon_state = ψ[1]

    @assert transmon_state ∈ ['g', 'e']

    cavity_state = parse(Int, ψ[2])

    @assert cavity_state ∈ 0:cavity_levels - 2 "cavity state must be in [0, ..., cavity_levels - 2] (hightest cavity level is prohibited)"

    ψ_transmon = zeros(ComplexF64, transmon_levels)
    ψ_transmon[transmon_state == 'g' ? 1 : 2] = 1.0

    ψ_cavity = zeros(ComplexF64, cavity_levels)
    ψ_cavity[cavity_state + 1] = 1.0

    return ψ_transmon ⊗ ψ_cavity
end


"""
    isomporphism utilities
"""

ket_to_iso(ψ) = [real(ψ); imag(ψ)]

iso_to_ket(ψ̃) = ψ̃[1:div(length(ψ̃), 2)] + im * ψ̃[(div(length(ψ̃), 2) + 1):end]

function normalize(state::Vector{C} where C <: Number)
    return state / norm(state)
end


"""
    quantum metrics
"""

function fidelity(ψ̃, ψ̃_goal)
    ψ = iso_to_ket(ψ̃)
    ψ_goal = iso_to_ket(ψ̃_goal)
    return abs2(ψ' * ψ_goal)
end

infidelity(ψ̃, ψ̃_goal) = 1 - fidelity(ψ̃, ψ̃_goal)


"""
    quantum measurement functions
"""

function population(ψ̃, i)
    @assert i ∈ 0:length(ψ̃) ÷ 2 - 1
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ[i + 1])
end

function populations(ψ̃)
    ψ = iso_to_ket(ψ̃)
    return abs2.(ψ)
end


end
