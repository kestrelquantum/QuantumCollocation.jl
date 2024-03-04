module QuantumSystems

export AbstractQuantumSystem
export QuantumSystem
export CompositeQuantumSystem
export QuantumSystemCoupling

export iso

using ..QuantumUtils

using LinearAlgebra
using SparseArrays

const Im2 = [
    0 -1;
    1  0
]

@doc raw"""
    G(H::AbstractMatrix)::Matrix{Float64}

Returns the isomorphism of ``-iH``:

```math
G(H) = \widetilde{- i H} = \mqty(1 & 0 \\ 0 & 1) \otimes \Im(H) - \mqty(0 & -1 \\ 1 & 0) \otimes \Re(H)
```

where ``\Im(H)`` and ``\Re(H)`` are the imaginary and real parts of ``H`` and the tilde indicates the standard isomorphism of a complex valued matrix:

```math
\widetilde{H} = \mqty(1 & 0 \\ 0 & 1) \otimes \Re(H) + \mqty(0 & -1 \\ 1 & 0) \otimes \Im(H)
```
"""
G(H::AbstractMatrix{<:Number}) = I(2) ⊗ imag(H) - Im2 ⊗ real(H)

iso(H::AbstractMatrix{<:Number}) = I(2) ⊗ real(H) + Im2 ⊗ imag(H)


"""
    H(G::AbstractMatrix{<:Number})::Matrix{ComplexF64}

Returns the inverse of `G(H) = iso(-iH)`, i.e. returns H

"""
function H(G::AbstractMatrix{<:Number})
    dim = size(G, 1) ÷ 2
    H_imag = G[1:dim, 1:dim]
    H_real = -G[dim+1:end, 1:dim]
    return H_real + 1.0im * H_imag
end

"""
```julia
AbstractQuantumSystem
```

Abstract type for defining systems.
"""
abstract type AbstractQuantumSystem end

# TODO: make subtypes: SingleQubitSystem, TwoQubitSystem, TransmonSystem, MultimodeSystem, etc.

"""
QuantumSystem
   |
    -> EmbeddedOperator
   |                  |
   |                   -> QuantumObjective
   |                  |
    -> Matrix (goal) -
"""

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
    levels::Int
    constructor::Union{Function, Nothing}
    params::Dict{Symbol, <:Any}
end

# TODO: add frame info to type

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
    constructor::Union{Function, Nothing}=nothing,
    params::Dict{Symbol, <:Any}=Dict{Symbol, Any}(),
    kwargs...
)
    H_drift = sparse(H_drift)
    H_drives = sparse.(H_drives)
    G_drift = G(H_drift)
    G_drives = G.(H_drives)
    params = merge(params, Dict{Symbol, Any}(kwargs...))
    levels = size(H_drift, 1)
    return QuantumSystem(
        H_drift,
        H_drives,
        G_drift,
        G_drives,
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

function Base.copy(sys::QuantumSystem)
    return QuantumSystem(
        copy(sys.H_drift),
        copy.(sys.H_drives);
        constructor=sys.constructor,
        params=copy(sys.params)
    )
end




# ------------------------------------------------------------------
# Quantum System couplings
# ------------------------------------------------------------------

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
    lab_frame::Bool=false,
    params::Dict{Symbol, Any}=Dict{Symbol, Any}()
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

    # add lifted couplings to the drift Hamiltonian
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

    G_drift = G(H_drift)
    G_drives = G.(H_drives)
    levels = size(H_drift, 1)
    subsystem_levels = [sys.levels for sys ∈ subsystems]
    params = merge(
        params,
        Dict{Symbol, Any}(
            :lab_frame => lab_frame,
        )
    )

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
        couplings;
        lab_frame=lab_frame
    )
end

QuantumUtils.quantum_state(ket::String, csys::CompositeQuantumSystem; kwargs...) =
    quantum_state(ket, csys.subsystem_levels; kwargs...)

function get_param(csys::CompositeQuantumSystem, key::Symbol; subsystem_index=1)
    return csys.params[key]
end

# TODO: add methods to combine composite quantum systems

end
