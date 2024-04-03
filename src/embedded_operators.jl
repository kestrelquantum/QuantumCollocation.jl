module EmbeddedOperators

export EmbeddedOperator

export embed
export unembed
export get_subspace_indices
export get_subspace_enr_indices
export get_subspace_leakage_indices
export get_unitary_isomorphism_leakage_indices
export get_unitary_isomorphism_subspace_indices
export get_subspace_identity

using LinearAlgebra

using TrajectoryIndexingUtils

using ..QuantumUtils
using ..QuantumSystems


basis_labels(subsystem_levels::AbstractVector{Int}; baseline=1) =
    kron([""], [string.(baseline:levels - 1 + baseline) for levels ∈ subsystem_levels]...)

basis_labels(subsystem_level::Int; kwargs...) = basis_labels([subsystem_level]; kwargs...)

function get_subspace_indices(
    subspaces::Vector{<:AbstractVector{Int}},
    subsystem_levels::AbstractVector{Int}
)
    @assert length(subspaces) == length(subsystem_levels)
    return findall(
        b -> all(l ∈ subspaces[i] for (i, l) ∈ enumerate([parse(Int, bᵢ) for bᵢ ∈ b])),
        basis_labels(subsystem_levels, baseline=1)
    )
end

get_subspace_indices(subspace::AbstractVector{Int}, levels::Int) =
    get_subspace_indices([subspace], [levels])

get_subspace_indices(levels::AbstractVector{Int}; subspace=1:2, kwargs...) =
    get_subspace_indices(fill(subspace, length(levels)), levels; kwargs...)

function get_subspace_enr_indices(excitation_restriction::Int, subsystem_levels::AbstractVector{Int})
    # excitation_number uses baseline of zero
    return findall(
        b -> sum([parse(Int, bᵢ) for bᵢ ∈ b]) ≤ excitation_restriction,
        basis_labels(subsystem_levels, baseline=0)
    )
end

function get_subspace_leakage_indices(
    subspaces::Vector{<:AbstractVector{Int}},
    subsystem_levels::AbstractVector{Int};
)
    subspace_indices = get_subspace_indices(subspaces, subsystem_levels)
    return get_subspace_leakage_indices(subspace_indices)
end

get_subspace_leakage_indices(subspace_indices::AbstractVector{Int}, levels::Int) =
    setdiff(1:levels, subspace_indices)

function get_unitary_isomorphism_subspace_indices(
    subspace_indices::AbstractVector{Int},
    subsystem_levels::AbstractVector{Int}
)
    N = prod(subsystem_levels)
    iso_subspace_indices = Int[]
    for sⱼ ∈ subspace_indices
        for sᵢ ∈ subspace_indices
            push!(iso_subspace_indices, index(sⱼ, sᵢ, 2N))
        end
        for sᵢ ∈ subspace_indices
            push!(iso_subspace_indices, index(sⱼ, sᵢ + N, 2N))
        end
    end
    return iso_subspace_indices
end

function get_unitary_isomorphism_leakage_indices(
    subspace_indices::AbstractVector{Int},
    subsystem_levels::AbstractVector{Int}
)
    N = prod(subsystem_levels)
    leakage_indices = get_subspace_leakage_indices(subspace_indices, N)
    iso_leakage_indices = Int[]
    for sⱼ ∈ subspace_indices
        for lᵢ ∈ leakage_indices
            push!(iso_leakage_indices, index(sⱼ, lᵢ, 2N))
        end
        for lᵢ ∈ leakage_indices
            push!(iso_leakage_indices, index(sⱼ, lᵢ + N, 2N))
        end
    end
    return iso_leakage_indices
end


struct EmbeddedOperator
    operator::Matrix{ComplexF64}
    subspace_indices::Vector{Int}
    subsystem_levels::Vector{Int}
end

Base.size(op::EmbeddedOperator) = size(op.operator)

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    system::QuantumSystem;
    subspace=1:size(op, 1)
)
    op_embedded = embed(op, system)
    return EmbeddedOperator(
        op_embedded,
        get_subspace_indices(subspace, system.levels),
        [system.levels]
    )
end

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    system::CompositeQuantumSystem,
    op_subsystem_index::Int;
    subspaces=fill(1:2, length(system.subsystems)),
)
    op_embedded = embed(op, system, op_subsystem_index; subspaces=subspaces)
    return EmbeddedOperator(
        op_embedded,
        get_subspace_indices(subspaces, system.subsystem_levels),
        system.subsystem_levels
    )
end

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    system::CompositeQuantumSystem,
    op_subsystem_indices::AbstractVector{Int};
    subspaces=fill(1:2, length(system.subsystems)),
)
    op_embedded = embed(op, system, op_subsystem_indices; subspaces=subspaces)
    return EmbeddedOperator(
        op_embedded,
        get_subspace_indices(subspaces, system.subsystem_levels),
        system.subsystem_levels
    )
end

function Base.:*(
    op1::EmbeddedOperator,
    op2::EmbeddedOperator
)
    @assert size(op1) == size(op2) "Operators must be of the same size."
    @assert op1.subspace_indices == op2.subspace_indices "Operators must have the same subspace."
    @assert op1.subsystem_levels == op2.subsystem_levels "Operators must have the same subsystem levels."
    return EmbeddedOperator(
        op1.operator * op2.operator,
        op1.subspace_indices,
        op1.subsystem_levels
    )
end

function EmbeddedOperator(op::Symbol, args...; kwargs...)
    @assert op ∈ keys(GATES) "Operator must be a valid gate. See QuantumCollocation.QuantumUtils.GATES dict for available gates."
    op = GATES[op]
    return EmbeddedOperator(op, args...; kwargs...)
end

function EmbeddedOperator(
    ops::AbstractVector{Symbol},
    sys::CompositeQuantumSystem,
    op_indices::AbstractVector{Int}
)
    ops_embedded = [
        EmbeddedOperator(op, sys, op_indices[i])
            for (op, i) ∈ zip(ops, op_indices)
    ]
    return *(ops_embedded...)
end




function get_subspace_identity(op::EmbeddedOperator)
    return embed(
        Matrix{ComplexF64}(I(length(op.subspace_indices))),
        op.subspace_indices,
        size(op)[1]
    )
end

function embed(op::Matrix{ComplexF64}, subspace_indices::AbstractVector{Int}, levels::Int)
    op_embedded = zeros(ComplexF64, levels, levels)
    op_embedded[subspace_indices, subspace_indices] = op
    return op_embedded
end

function embed(A::AbstractMatrix{<:Number}, op::EmbeddedOperator)
    @assert size(A, 1) == size(A, 2) "Operator must be square."
    @assert size(A, 1) == length(op.subspace_indices) "Operator size must match subspace size."
    return embed(A, op.subspace_indices, size(op)[1])
end

function unembed(op::EmbeddedOperator)::Matrix{ComplexF64}
    return op.operator[op.subspace_indices, op.subspace_indices]
end

function get_subspace_leakage_indices(op::EmbeddedOperator)
    return get_subspace_leakage_indices(op.subspace_indices, size(op)[1])
end

get_unitary_isomorphism_subspace_indices(op::EmbeddedOperator) =
    get_unitary_isomorphism_subspace_indices(op.subspace_indices, op.subsystem_levels)

get_unitary_isomorphism_leakage_indices(op::EmbeddedOperator) =
    get_unitary_isomorphism_leakage_indices(op.subspace_indices, op.subsystem_levels)


# embed(op::AbstractMatrix)

function embed(
    op::Matrix{ComplexF64},
    sys::QuantumSystem;
    subspace=1:size(op, 1)
)::Matrix{ComplexF64}
    @assert size(op, 1) == size(op, 2) "Operator must be square."
    op_embedded = embed(op, subspace, sys.levels)
    return op_embedded
end

embed(op::AbstractMatrix{<:Number}, sys; kwargs...) =
    embed(Matrix{ComplexF64}(op), sys; kwargs...)


function embed(
    op::Matrix{ComplexF64},
    csys::CompositeQuantumSystem,
    op_subsystem_indices::AbstractVector{Int};
    subspaces::Vector{<:AbstractVector{Int}}=fill(1:2, length(csys.subsystems))
)
    @assert size(op, 1) == size(op, 2) "Operator must be square."
    @assert all(diff(op_subsystem_indices) .== 1) "op_subsystem_indices must be consecutive (for now)."

    if size(op, 1) == prod(length.(subspaces[op_subsystem_indices]))
        Is = Matrix{ComplexF64}.(I.(length.(subspaces)))
        Is[op_subsystem_indices[1]] = op
        deleteat!(Is, op_subsystem_indices[2:end])
        op = kron(Is...)
    else
        @assert(
            size(op, 1) == prod(length.(subspaces)),
            """\n
                Operator size ($(size(op, 1))) must match product of subsystem subspaces ($(prod(length.(subspaces)))). Or
            """
        )
    end

    subspace_indices = get_subspace_indices(subspaces, csys.subsystem_levels)

    op_embedded = embed(op, subspace_indices, csys.levels)

    return op_embedded
end

function embed(
    op::Matrix{ComplexF64},
    csys::CompositeQuantumSystem,
    op_subsystem_index::Int;
    kwargs...
)
    return embed(op, csys, [op_subsystem_index]; kwargs...)
end

embed(op::AbstractMatrix{<:Number}, sys::AbstractQuantumSystem, args...; kwargs...) =
    embed(Matrix{ComplexF64}(op), sys, args...; kwargs...)





end
