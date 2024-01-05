module QuantumUtils

export GATES
export gate
export ⊗
export apply
export qubit_system_state
export lift
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
export multimode_state
export number
export fidelity
export iso_fidelity
export unitary_fidelity
export population
export populations
export subspace_unitary
export quantum_state
export get_subspace_indices
export get_subspace_leakage_indices
export get_unitary_isomorphism_leakage_indices

using TrajectoryIndexingUtils
using LinearAlgebra


"""
    kronecker product utility
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

function qubit_system_state(ket::String)
    cs = [c for c ∈ ket]
    @assert all(c ∈ "01" for c ∈ cs)
    states = [c == '0' ? [1, 0] : [0, 1] for c ∈ cs]
    ψ = foldr(⊗, states)
    ψ = Vector{ComplexF64}(ψ)
    return ψ
end

function lift(
    U::AbstractMatrix{<:Number},
    qubit_index::Int,
    n_qubits::Int;
    levels::Int=size(U, 1)
)::Matrix{ComplexF64}
    Is = Matrix{Complex}[I(levels) for _ = 1:n_qubits]
    Is[qubit_index] = U
    return foldr(⊗, Is)
end

function lift(
    op::AbstractMatrix{<:Number},
    i::Int,
    sub_levels::Vector{Int}
)::Matrix{ComplexF64}
    @assert size(op, 1) == size(op, 2) == sub_levels[i] "Operator must be square and match dimension of subsystem i"

    Is = [collect(1.0 * typeof(op)(I, l, l)) for l ∈ sub_levels]
    Is[i] = op
    return kron(1.0, Is...)
end




"""
    quantum harmonic oscillator operators
"""

"""
    annihilate(levels::Int)

Get the annihilation operator for a system with `levels` levels.
"""
function annihilate(levels::Int)::Matrix{ComplexF64}
    return diagm(1 => map(sqrt, 1:levels - 1))
end

"""
    create(levels::Int)

Get the creation operator for a system with `levels` levels.
"""
function create(levels::Int)
    return collect(annihilate(levels)')
end

"""
    number(levels::Int)

Get the number operator `n = a'a` for a system with `levels` levels.
"""
function number(levels::Int)
    return create(levels) * annihilate(levels)
end

"""
    quad(levels::Int)

Get the operator `n(n - I)` for a system with `levels` levels.
"""
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

function iso_vec_to_operator(Ũ⃗::AbstractVector{R}) where R <: Real
    Ũ⃗_dim = div(length(Ũ⃗), 2)
    N = Int(sqrt(Ũ⃗_dim))
    isodim = 2N
    U = Matrix{Complex{R}}(undef, N, N)
    for i=0:N-1
        U[:, i+1] .=
            @view(Ũ⃗[i*2N .+ (1:N)]) +
            one(R) * im * @view(Ũ⃗[i*2N .+ (N+1:2N)])
    end
    return U
end

function iso_vec_to_iso_operator(Ũ⃗::AbstractVector{R}) where R <: Real
    N = Int(sqrt(length(Ũ⃗) ÷ 2))
    Ũ = Matrix{R}(undef, 2N, 2N)
    U_real = Matrix{R}(undef, N, N)
    U_imag = Matrix{R}(undef, N, N)
    for i=0:N-1
        U_real[:, i+1] .= @view(Ũ⃗[i*2N .+ (1:N)])
        U_imag[:, i+1] .= @view(Ũ⃗[i*2N .+ (N+1:2N)])
    end
    Ũ[1:N, 1:N] .= U_real
    Ũ[1:N, (N + 1):end] .= -U_imag
    Ũ[(N + 1):end, 1:N] .= U_imag
    Ũ[(N + 1):end, (N + 1):end] .= U_real
    return Ũ
end


function operator_to_iso_vec(U::AbstractMatrix)
    N = size(U,1)
    Ũ⃗ = Vector{Float64}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:N)] .= real(@view(U[:, i+1]))
        Ũ⃗[i*2N .+ (N+1:2N)] .= imag(@view(U[:, i+1]))
    end
    return Vector{Float64}(Ũ⃗)
end

function iso_operator_to_iso_vec(Ũ::AbstractMatrix{R}) where R <: Real
    N = size(Ũ, 1) ÷ 2
    Ũ⃗ = Vector{R}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:2N)] .= @view Ũ[:, i+1]
    end
    return Ũ⃗
end


"""
    quantum metrics
"""

function fidelity(ψ, ψ_goal)
    return abs2(ψ_goal'ψ)
end

function iso_fidelity(ψ̃, ψ̃_goal)
    ψ = iso_to_ket(ψ̃)
    ψ_goal = iso_to_ket(ψ̃_goal)
    return fidelity(ψ, ψ_goal)
end

function unitary_fidelity(U::Matrix, U_goal::Matrix; subspace=nothing)
    if isnothing(subspace)
        N = size(U, 1)
        return 1 / N * abs(tr(U_goal'U))
    else
        U_goal = U_goal[subspace, subspace]
        U = U[subspace, subspace]
        N = size(U, 1)
        return 1 / N * abs(tr(U_goal'U))
    end
end

function unitary_fidelity(Ũ⃗::AbstractVector{<:Real}, Ũ⃗_goal::AbstractVector{<:Real}; subspace=nothing)
    U = iso_vec_to_operator(Ũ⃗)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)
    return unitary_fidelity(U, U_goal; subspace=subspace)
end

"""
    quantum measurement functions
"""

function population(ψ̃, i)
    @assert i ∈ 0:length(ψ̃) ÷ 2 - 1
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ[i + 1])
end

function populations(ψ̃::AbstractVector{<:Real})
    ψ = iso_to_ket(ψ̃)
    return abs2.(ψ)
end

function populations(ψ::AbstractVector{<:Complex})
    return abs2.(ψ)
end

"""
    unitary subspace utilities
"""

"""
    subspace_unitary(
        levels::Vector{Int},
        gate_name::Symbol,
        qubit::Union{Int, Vector{Int}}
    )

Get a unitary matrix for a gate acting on a subspace of a multilevel system.

TODO: reimplement this as `embed_operator` with more methods.
"""
function subspace_unitary(
    levels::Vector{Int},
    gate_name::Symbol,
    qubit::Union{Int, Vector{Int}}
)
    if qubit isa Int
        @assert length(string(gate_name)) == 1
        @assert gate_name ∈ keys(GATES)
        gate = zeros(ComplexF64, levels[qubit], levels[qubit])
        gate[1:2, 1:2] = GATES[gate_name]
    else
        @assert length(qubit) == 2 "only 2-qubit gates are supported, for now"
        @assert all(qubit .== qubit[1]:qubit[end]) "Qubits must be consecutive"
        @assert length(string(gate_name)) == length(qubit)
        @assert first(string(gate_name)) == 'C' "Only controlled gates are supported, for now"
        @assert Symbol(last(string(gate_name))) ∈ keys(GATES)
        @assert gate_name == :CX "Only CX gates are supported, for now"
        g1 = cavity_state(0, levels[qubit[1]])
        e1 = cavity_state(1, levels[qubit[1]])
        g2 = cavity_state(0, levels[qubit[2]])
        e2 = cavity_state(1, levels[qubit[2]])
        gg = g1 ⊗ g2
        ge = g1 ⊗ e2
        eg = e1 ⊗ g2
        ee = e1 ⊗ e2
        gate = gg * gg' + ge * ge' + ee * eg' + eg * ee'
    end

    # fill with ones to handle kron of possibly only one element
    U_init = [[1.0 + 0.0im;;]]
    U_goal = [[1.0 + 0.0im;;]]
    added_gate = false
    for (i, level) ∈ enumerate(levels)
        gᵢ = cavity_state(0, level)
        eᵢ = cavity_state(1, level)
        Idᵢ =  gᵢ * gᵢ' + eᵢ * eᵢ'
        push!(U_init, Idᵢ)
        if i ∈ qubit
            if added_gate
                continue
            else
                push!(U_goal, gate)
                added_gate = true
            end
        else
            push!(U_goal, Idᵢ)
        end
    end
    U_init = kron(U_init...)
    U_goal = kron(U_goal...)
    return U_init, U_goal
end

function quantum_state(
    ket::String,
    levels::Vector{Int};
    level_dict=Dict(:g => 0, :e => 1, :f => 2, :h => 2),
    return_states=false
)
    kets = []

    for x ∈ split(ket, ['(', ')'])
        if x == ""
            continue
        elseif all(Symbol(xᵢ) ∈ keys(level_dict) for xᵢ ∈ x)
            append!(kets, x)
        elseif occursin("+", x)
            superposition = split(x, '+')
            @assert all(all(Symbol(xᵢ) ∈ keys(level_dict) for xᵢ ∈ x) for x ∈ superposition) "Invalid ket: $x"
            @assert length(superposition) == 2 "Only two states can be superposed for now"
            push!(kets, x)
        else
            error("Invalid ket: $x")
        end
    end

    states = []

    for (ψᵢ, l) ∈ zip(kets, levels)
        if ψᵢ isa AbstractString && occursin("+", ψᵢ)
            superposition = split(ψᵢ, '+')
            superposition_states = [level_dict[Symbol(x)] for x ∈ superposition]
            @assert all(state ≤ l - 1 for state ∈ superposition_states) "Level $ψᵢ is not allowed for $l levels"
            superposition_state = sum([cavity_state(state, l) for state ∈ superposition_states])
            normalize!(superposition_state)
            push!(states, superposition_state)
        else
            state = level_dict[Symbol(ψᵢ)]
            @assert state ≤ l - 1 "Level $ψᵢ is not allowed for $l levels"
            push!(states, cavity_state(state, l))
        end
    end

    if return_states
        return states
    else
        return kron(states...)
    end
end

function get_subspace_indices(
    subspaces::Vector{<:AbstractVector{Int}},
    subsystem_levels::AbstractVector{Int}
)
    @assert length(subspaces) == length(subsystem_levels)

    basis = kron([""], [string.(1:level) for level ∈ subsystem_levels]...)

    subspace_indices = findall(
        b -> all(
            l ∈ subspaces[i]
                for (i, l) ∈ enumerate([parse(Int, bᵢ) for bᵢ ∈ b])
        ),
        basis
    )

    return subspace_indices
end

get_subspace_indices(levels::AbstractVector{Int}; subspace=1:2, kwargs...) =
    get_subspace_indices(fill(subspace, length(levels)), levels; kwargs...)

function get_subspace_leakage_indices(
    subspace_levels::AbstractVector{Int},
    levels::AbstractVector{Int};
)
    basis = kron([""], [string.(1:level) for level ∈ levels]...)
    subspace_indices = findall(
        b -> all(
            l ≤ subspace_levels[i]
                for (i, l) ∈ enumerate([parse(Int, bᵢ) for bᵢ ∈ b])
        ),
        basis
    )
    return setdiff(1:length(basis), subspace_indices)
end

get_subspace_leakage_indices(levels::Int; kwargs...) =
    get_subspace_leakage_indices([levels]; kwargs...)

get_subspace_leakage_indices(levels::AbstractVector{Int}; subspace_max=2, kwargs...) =
    get_subspace_leakage_indices(fill(subspace_max, length(levels)), levels; kwargs...)

function get_unitary_isomorphism_leakage_indices(levels::AbstractVector{Int}; kwargs...)
    N = prod(levels)
    subspace_inds = get_subspace_indices(levels; kwargs...)
    leakage_inds = get_subspace_leakage_indices(levels; kwargs...)
    iso_leakage_inds = Int[]
    for sⱼ ∈ subspace_inds
        for lᵢ ∈ leakage_inds
            push!(iso_leakage_inds, index(sⱼ, lᵢ, 2N))
            push!(iso_leakage_inds, index(sⱼ, lᵢ + N, 2N))
        end
    end
    return iso_leakage_inds
end










end
