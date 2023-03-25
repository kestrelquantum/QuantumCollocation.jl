module QuantumUtils

export GATES
export ⊗
export apply
export ket_to_iso
export iso_to_ket
export operator_to_iso_vec
export iso_vec_to_operator
export iso_vec_to_iso_operator
export iso_operator_to_iso_vec
export annihilate
export create
export quad
export cavity_state
export number
export normalize
export fidelity
export unitary_fidelity
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
    :I => [1 0;
           0 1],

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

function apply(gate::Symbol, ψ::Vector{<:Number})
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

function iso_vec_to_operator(Ũ⃗::AbstractVector)
    Ũ⃗_dim = div(length(Ũ⃗), 2)
    ket_dim = Int(sqrt(Ũ⃗_dim))
    U_real_vec = Ũ⃗[1:Ũ⃗_dim]
    U_imag_vec = Ũ⃗[(Ũ⃗_dim + 1):end]
    U_real = reshape(U_real_vec, ket_dim, ket_dim)
    U_imag = reshape(U_imag_vec, ket_dim, ket_dim)
    U = U_real + im * U_imag
    return U
end

function iso_vec_to_iso_operator(Ũ⃗::AbstractVector)
    Ũ⃗_dim = div(length(Ũ⃗), 2)
    ket_dim = Int(sqrt(Ũ⃗_dim))
    U_real_vec = Ũ⃗[1:Ũ⃗_dim]
    U_imag_vec = Ũ⃗[(Ũ⃗_dim + 1):end]
    U_real = reshape(U_real_vec, ket_dim, ket_dim)
    U_imag = reshape(U_imag_vec, ket_dim, ket_dim)
    Ũ = vcat(hcat(U_real, -U_imag), hcat(U_imag, U_real))
    return Ũ
end

function operator_to_iso_vec(U::AbstractMatrix)
    U_real = real(U)
    U_imag = imag(U)
    U_real_vec = vec(U_real)
    U_imag_vec = vec(U_imag)
    Ũ⃗ = [U_real_vec; U_imag_vec]
    return Ũ⃗
end

function iso_operator_to_iso_vec(Ũ::AbstractMatrix)
    ket_dim = size(Ũ, 1) ÷ 2
    U_real = Ũ[1:ket_dim, 1:ket_dim]
    U_imag = Ũ[(ket_dim + 1):end, 1:ket_dim]
    U_real_vec = vec(U_real)
    U_imag_vec = vec(U_imag)
    Ũ⃗ = [U_real_vec; U_imag_vec]
    return Ũ⃗
end



"""
    quantum metrics
"""

function fidelity(ψ̃, ψ̃_goal)
    ψ = iso_to_ket(ψ̃)
    ψ_goal = iso_to_ket(ψ̃_goal)
    return abs2(ψ' * ψ_goal)
end

function unitary_fidelity(Ũ⃗::Vector, Ũ⃗_goal::Vector)
    U = iso_vec_to_operator(Ũ⃗)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)
    N = size(U, 1)
    return 1 / N * abs(tr(U'U_goal))
end

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
