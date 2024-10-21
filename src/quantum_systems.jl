module QuantumSystems

export AbstractQuantumSystem
export QuantumSystem
export CompositeQuantumSystem
export QuantumSystemCoupling

export iso
export lift

using ..Isomorphisms
using ..QuantumObjectUtils

using LinearAlgebra
using SparseArrays
using TestItemRunner

# TODO:
# [ ] subtypes? SingleQubitSystem, TwoQubitSystem, TransmonSystem, MultimodeSystem, etc.
# [ ] add frame info to type
# [ ] add methods to combine composite quantum systems

# ----------------------------------------------------------------------------- #
# AbstractQuantumSystem
# ----------------------------------------------------------------------------- #

"""
    AbstractQuantumSystem

Abstract type for defining systems.
"""
abstract type AbstractQuantumSystem end

# ----------------------------------------------------------------------------- #
# QuantumSystem
# ----------------------------------------------------------------------------- #

"""
    QuantumSystem <: AbstractQuantumSystem

A struct for storing the isomorphisms of the system's drift and drive Hamiltonians,
as well as the system's parameters.
"""
struct QuantumSystem <: AbstractQuantumSystem
    H_drift::SparseMatrixCSC{ComplexF64, Int}
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}
    G_drift::SparseMatrixCSC{Float64, Int}
    G_drives::Vector{SparseMatrixCSC{Float64, Int}}
    dissipation_operators::Union{Nothing, Vector{SparseMatrixCSC{ComplexF64, Int}}}
    levels::Int
    constructor::Union{Function, Nothing}
    params::Dict{Symbol, <:Any}
end

"""
    QuantumSystem(
        H_drift::Matrix{<:Number},
        H_drives::Vector{Matrix{<:Number}};
        params=Dict{Symbol, Any}(),
        kwargs...
    )::QuantumSystem

Constructs a `QuantumSystem` object from the drift and drive Hamiltonian terms.
"""
function QuantumSystem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}};
    dissipation_operators=nothing,
    constructor::Union{Function, Nothing}=nothing,
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}(),
    kwargs...
)
    H_drift = sparse(H_drift)
    H_drives = sparse.(H_drives)
    G_drift = Isomorphisms.G(H_drift)
    G_drives = Isomorphisms.G.(H_drives)
    params = merge(params, Dict{Symbol, Any}(kwargs...))
    levels = size(H_drift, 1)
    return QuantumSystem(
        H_drift,
        H_drives,
        G_drift,
        G_drives,
        dissipation_operators,
        levels,
        constructor,
        params
    )
end

function QuantumSystem(H_drives::Vector{<:AbstractMatrix{<:Number}}; kwargs...)
    return QuantumSystem(
        zeros(eltype(H_drives[1]), size(H_drives[1])),
        H_drives;
        kwargs...
    )
end

function QuantumSystem(H_drift::AbstractMatrix{<:Number}; kwargs...)
    return QuantumSystem(
        H_drift,
        Matrix{ComplexF64}[];
        kwargs...
    )
end

function (sys::QuantumSystem)(; params...)
    @assert !isnothing(sys.constructor) "No constructor provided."
    @assert all([
        key ∈ keys(sys.params) for key ∈ keys(params)
    ]) "Invalid parameter(s) provided."
    return sys.constructor(; merge(sys.params, Dict(params...))...)
end

function QuantumSystem(
    H_drift::SparseMatrixCSC{ComplexF64, Int64},
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int64}};
    dissipation_operators=nothing,
    constructor::Union{Function, Nothing}=nothing,
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}(),
    kwargs...
)
    H_drift = sparse(H_drift)
    H_drives = sparse.(H_drives)
    G_drift = Isomorphisms.G(H_drift)
    G_drives = Isomorphisms.G.(H_drives)
    params = merge(params, Dict{Symbol, Any}(kwargs...))
    levels = size(H_drift, 1)
    return QuantumSystem(
        H_drift,
        H_drives,
        G_drift,
        G_drives,
        dissipation_operators,
        levels,
        constructor,
        params
    )
end

function Base.copy(sys::QuantumSystem)
    return QuantumSystem(
        copy(sys.H_drift),
        copy.(sys.H_drives);
        constructor=sys.constructor,
        params=copy(sys.params)
    )
end

# ----------------------------------------------------------------------------- #
# Quantum System couplings
# ----------------------------------------------------------------------------- #

@doc raw"""
    lift(U::AbstractMatrix{<:Number}, qubit_index::Int, n_qubits::Int; levels::Int=size(U, 1))

Lift an operator `U` acting on a single qubit to an operator acting on the entire system of `n_qubits`.
"""
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

@doc raw"""
    lift(op::AbstractMatrix{<:Number}, i::Int, subsystem_levels::Vector{Int})

Lift an operator `op` acting on the i-th subsystem to an operator acting on the entire system with given subsystem levels.
"""
function lift(
    op::AbstractMatrix{<:Number},
    i::Int,
    subsystem_levels::Vector{Int}
)::Matrix{ComplexF64}
    @assert size(op, 1) == size(op, 2) == subsystem_levels[i] "Operator must be square and match dimension of subsystem i"

    Is = [collect(1.0 * typeof(op)(I, l, l)) for l ∈ subsystem_levels]
    Is[i] = op
    return kron(1.0, Is...)
end

"""
    QuantumSystemCoupling <: AbstractQuantumSystem

"""
struct QuantumSystemCoupling
    term::SparseMatrixCSC{ComplexF64, Int}
    g_ij::Float64
    pair::Tuple{Int, Int}
    subsystem_levels::Vector{Int}
    constructor::Union{Function, Nothing}
    params::Dict{Symbol, <:Any}

    function QuantumSystemCoupling(op::AbstractMatrix{<:ComplexF64}, args...)
        return new(
            sparse(op),
            args...
        )
    end
end

function (coupling::QuantumSystemCoupling)(;
    g_ij::Float64=coupling.g_ij,
    pair::Tuple{Int, Int}=coupling.pair,
    subsystem_levels::Vector{Int}=coupling.subsystem_levels,
    params...
)
    @assert !isnothing(coupling.constructor) "No constructor provided."
    @assert all([
        key ∈ keys(coupling.params) for key ∈ keys(params)
    ]) "Invalid parameter(s) provided: $(filter(param -> param ∉ keys(coupling.params), keys(params)))"
    return coupling.constructor(
        g_ij,
        pair,
        subsystem_levels;
        merge(coupling.params, Dict(params...))...
    )
end

function Base.copy(coupling::QuantumSystemCoupling)
    return QuantumSystemCoupling(
        copy(coupling.op),
        coupling.g_ij,
        coupling.pair,
        coupling.subsystem_levels,
        coupling.constructor,
        copy(coupling.params)
    )
end


struct CompositeQuantumSystem <: AbstractQuantumSystem
    H_drift::SparseMatrixCSC{ComplexF64, Int}
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int}}
    G_drift::SparseMatrixCSC{Float64, Int}
    G_drives::Vector{SparseMatrixCSC{Float64, Int}}
    levels::Int
    subsystem_levels::Vector{Int}
    params::Dict{Symbol, Any}
    subsystems::Vector{QuantumSystem}
    couplings::Vector{QuantumSystemCoupling}
end

function CompositeQuantumSystem(
    subsystems::Vector{QuantumSystem},
    couplings::Vector{QuantumSystemCoupling}=QuantumSystemCoupling[];
    subsystem_frame_index::Int=1,
    frame_ω::Float64=subsystems[subsystem_frame_index].params[:ω],
    lab_frame::Bool=false
)
    # set all subsystems to the same frame_ω
    subsystems = [sys(; frame_ω=frame_ω) for sys ∈ subsystems]

    if lab_frame
        subsystems = [sys(; lab_frame=true) for sys ∈ subsystems]
        couplings = [coupling(; lab_frame=true) for coupling ∈ couplings]
    end

    subsystem_levels = [sys.levels for sys ∈ subsystems]
    levels = prod(subsystem_levels)

    # add lifted subsystem drift Hamiltonians
    H_drift = sparse(zeros(levels, levels))
    for (i, sys) ∈ enumerate(subsystems)
        H_drift += lift(sys.H_drift, i, subsystem_levels)
    end

    # add lifated couplings to the drift Hamiltonian
    for coupling ∈ couplings
        H_drift += coupling.term
    end

    # add lifted subsystem drive Hamiltonians
    H_drives = SparseMatrixCSC{ComplexF64, Int}[]
    for (i, sys) ∈ enumerate(subsystems)
        for H_drive ∈ sys.H_drives
            push!(H_drives, lift(H_drive, i, subsystem_levels))
        end
    end

    G_drift = Isomorphisms.G(H_drift)
    G_drives = Isomorphisms.G.(H_drives)
    levels = size(H_drift, 1)
    subsystem_levels = [sys.levels for sys ∈ subsystems]
    params = Dict{Symbol, Any}()

    return CompositeQuantumSystem(
        H_drift,
        H_drives,
        G_drift,
        G_drives,
        levels,
        subsystem_levels,
        params,
        subsystems,
        couplings
    )
end

function (csys::CompositeQuantumSystem)(;
    subsystem_params::Dict{Int, <:Dict{Symbol, <:Any}}=Dict{Int, Dict{Symbol, Any}}(),
    coupling_params::Dict{Int, <:Dict{Symbol, <:Any}}=Dict{Int, Dict{Symbol, Any}}(),
    lab_frame::Bool=false,
    subsystem_frame_index::Int=1,
    frame_ω::Float64=csys.subsystems[subsystem_frame_index].params[:ω],
    subsystem_levels::Union{Nothing, Int, Vector{Int}}=nothing,
)
    subsystems = deepcopy(csys.subsystems)
    couplings = deepcopy(csys.couplings)

    # if lab frame then set all subsystems and couplings to lab frame
    if lab_frame

        # set lab frame in subsystem_params for all subsystems
        for i = 1:length(csys.subsystems)
            if i ∈ keys(subsystem_params)
                subsystem_params[i][:lab_frame] = true
            else
                subsystem_params[i] = Dict{Symbol, Any}(:lab_frame => true)
            end
        end

        # set lab frame in coupling_params for all couplings
        for i = 1:length(csys.couplings)
            if i ∈ keys(coupling_params)
                coupling_params[i][:lab_frame] = true
            else
                coupling_params[i] = Dict{Symbol, Any}(:lab_frame => true)
            end
        end
    end

    # if subsystem_levels is provided then set all subsystems and couplings to subsystem_levels
    if !isnothing(subsystem_levels)

        if subsystem_levels isa Int
            subsystem_levels = fill(subsystem_levels, length(csys.subsystems))
        else
            @assert(
                length(subsystem_levels) == length(csys.subsystems),
                """\n
                    number of subsystem_levels ($(length(subsystem_levels))) must match number of subsystems ($(length(csys.subsystems))).
                """
            )
        end

        for i = 1:length(csys.subsystems)
            if i ∈ keys(subsystem_params)
                subsystem_params[i][:levels] = subsystem_levels[i]
            else
                subsystem_params[i] = Dict{Symbol, Any}(
                    :levels => subsystem_levels[i]
                )
            end
        end

        for i = 1:length(csys.couplings)
            if i ∈ keys(coupling_params)
                coupling_params[i][:subsystem_levels] = subsystem_levels
            else
                coupling_params[i] = Dict{Symbol, Any}(
                    :subsystem_levels => subsystem_levels
                )
            end
        end
    end

    # construct subsystems with new parameters
    for (i, sys_params) ∈ subsystem_params
        subsystem_i_new_params = merge(subsystems[i].params, sys_params)
        subsystem_i_new_params[:frame_ω] = frame_ω
        subsystems[i] = subsystems[i](; subsystem_i_new_params...)
    end

    # sometimes redundant, but here to catch any changes in indvidual subsystem levels
    subsystem_levels = [sys.levels for sys ∈ subsystems]

    # construct couplings with new parameters
    if !isempty(csys.couplings)
        for (i, coupling_params) ∈ coupling_params
            couplings[i] = couplings[i](;
                merge(couplings[i].params, coupling_params)...,
                subsystem_levels=subsystem_levels
            )
        end
    end

    return CompositeQuantumSystem(
        subsystems,
        couplings
    )
end

# ============================================================================= #

@testitem "System creation" begin
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    n_drives = length(H_drives)

    system = QuantumSystem(H_drift, H_drives)
end

function is_reachable(
    gate::AbstractMatrix,
    system::QuantumSystem;
    use_drift::Bool=true,
    kwargs...
)
    if !iszero(system.H_drift) && use_drift
        hamiltonians = [system.H_drift, system.H_drives...]
    else
        hamiltonians = system.H_drives
    end
    return is_reachable(gate, hamiltonians; kwargs...)
end



end
