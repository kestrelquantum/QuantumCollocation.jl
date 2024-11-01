module EmbeddedOperators

export OperatorType
export EmbeddedOperator

export embed
export unembed
export get_subspace_indices
export get_subspace_enr_indices
export get_subspace_leakage_indices
export get_iso_vec_leakage_indices
export get_iso_vec_subspace_indices
export get_subspace_identity

using LinearAlgebra
using TestItemRunner

using TrajectoryIndexingUtils

using ..Isomorphisms
using ..QuantumObjectUtils
using ..QuantumSystems
# using ..QuantumSystemUtils

@doc raw"""
    embed(matrix::Matrix{ComplexF64}, subspace_indices::AbstractVector{Int}, levels::Int)

Embed an operator $U$ in the subspace of a larger system $\mathcal{X} = \mathcal{X}_{\text{subspace}} \oplus \mathcal{X}_{\text{leakage}}$ which is composed of matrices of size $\text{levels} \times \text{levels}$.

# Arguments
- `matrix::Matrix{ComplexF64}`: Operator to embed.
- `subspace_indices::AbstractVector{Int}`: Indices of the subspace to embed the operator in.
- `levels::Int`: Total number of levels in the system.
"""
function embed(op::Matrix{ComplexF64}, subspace_indices::AbstractVector{Int}, levels::Int)
    @assert size(op, 1) == size(op, 2) "Operator must be square."
    op_embedded = zeros(ComplexF64, levels, levels)
    op_embedded[subspace_indices, subspace_indices] = op
    return op_embedded
end

@doc raw"""
    unembed(matrix::AbstractMatrix, subspace_indices::AbstractVector{Int})

Unembed an operator $U$ from a subspace of a larger system $\mathcal{X} = \mathcal{X}_{\text{subspace}} \oplus \mathcal{X}_{\text{leakage}}$ which is composed of matrices of size $\text{levels} \times \text{levels}$.

This is equivalent to calling `matrix[subspace_indices, subspace_indices]`.

# Arguments
- `matrix::AbstractMatrix`: Operator to unembed.
- `subspace_indices::AbstractVector{Int}`: Indices of the subspace to unembed the operator from.
"""
function unembed(matrix::AbstractMatrix, subspace_indices::AbstractVector{Int})
    return matrix[subspace_indices, subspace_indices]
end

# ----------------------------------------------------------------------------- #
#                             Embedded Operator                                 #
# ----------------------------------------------------------------------------- #

"""
    EmbeddedOperator

Embedded operator type to represent an operator embedded in a subspace of a larger quantum system.

# Fields
- `operator::Matrix{ComplexF64}`: Embedded operator of size `prod(subsystem_levels) x prod(subsystem_levels)`.
- `subspace_indices::Vector{Int}`: Indices of the subspace the operator is embedded in.
- `subsystem_levels::Vector{Int}`: Levels of the subsystems in the composite system.
"""
struct EmbeddedOperator
    operator::Matrix{ComplexF64}
    subspace_indices::Vector{Int}
    subsystem_levels::Vector{Int}

    @doc raw"""
        EmbeddedOperator(op::Matrix{<:Number}, subspace_indices::AbstractVector{Int}, subsystem_levels::AbstractVector{Int})

    Create an embedded operator. The operator `op` is embedded in the subspace defined by `subspace_indices` in `subsystem_levels`.

    # Arguments
    - `op::Matrix{<:Number}`: Operator to embed.
    - `subspace_indices::AbstractVector{Int}`: Indices of the subspace to embed the operator in. e.g. `get_subspace_indices([1:2, 1:2], [3, 3])`.
    - `subsystem_levels::AbstractVector{Int}`: Levels of the subsystems in the composite system. e.g. `[3, 3]` for two 3-level systems.
    """
    function EmbeddedOperator(
        op::Matrix{<:Number},
        subspace_indices::AbstractVector{Int},
        subsystem_levels::AbstractVector{Int}
    )

        op_embedded = embed(Matrix{ComplexF64}(op), subspace_indices, prod(subsystem_levels))
        return new(op_embedded, subspace_indices, subsystem_levels)
    end
end

const OperatorType = Union{AbstractMatrix{<:Number}, EmbeddedOperator}

EmbeddedOperator(op::Matrix{<:Number}, subspace_indices::AbstractVector{Int}, levels::Int) =
    EmbeddedOperator(op, subspace_indices, [levels])

function embed(matrix::Matrix{ComplexF64}, op::EmbeddedOperator)
    return embed(matrix, op.subspace_indices, prod(op.subsystem_levels))
end

function unembed(op::EmbeddedOperator)::Matrix{ComplexF64}
    return op.operator[op.subspace_indices, op.subspace_indices]
end

function unembed(matrix::AbstractMatrix, op::EmbeddedOperator)
    return matrix[op.subspace_indices, op.subspace_indices]
end

Base.size(op::EmbeddedOperator) = size(op.operator)
Base.size(op::EmbeddedOperator, dim::Union{Int, Nothing}) = size(op.operator, dim)

function Base.:*(
    op1::EmbeddedOperator,
    op2::EmbeddedOperator
)
    @assert size(op1) == size(op2) "Operators must be of the same size."
    @assert op1.subspace_indices == op2.subspace_indices "Operators must have the same subspace."
    @assert op1.subsystem_levels == op2.subsystem_levels "Operators must have the same subsystem levels."
    return EmbeddedOperator(
        unembed(op1) * unembed(op2),
        op1.subspace_indices,
        op1.subsystem_levels
    )
end

function Base.kron(op1::EmbeddedOperator, op2::EmbeddedOperator)
    levels = [size(op1, 1), size(op2, 2)]
    indices = get_subspace_indices(
        [op1.subspace_indices, op2.subspace_indices], levels
    )
    return EmbeddedOperator(unembed(op1) ⊗ unembed(op2), indices, levels)
end

Isomorphisms.:⊗(A::EmbeddedOperator, B::EmbeddedOperator) = kron(A, B)

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    system::QuantumSystem;
    subspace=1:size(op, 1)
)
    return EmbeddedOperator(
        op,
        get_subspace_indices(subspace, system.levels),
        [system.levels]
    )
end

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    csystem::CompositeQuantumSystem,
    op_subsystem_indices::AbstractVector{Int};
    subspaces=fill(1:2, length(csystem.subsystems)),
)
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
                Operator size ($(size(op, 1))) must match product of subsystem subspaces ($(prod(length.(subspaces)))).
            """
        )
    end

    subspace_indices = get_subspace_indices(subspaces, csystem.subsystem_levels)

    return EmbeddedOperator(
        op,
        subspace_indices,
        csystem.subsystem_levels
    )
end

function EmbeddedOperator(
    op::AbstractMatrix{<:Number},
    csystem::CompositeQuantumSystem,
    op_subsystem_index::Int;
    kwargs...
)
    return EmbeddedOperator(
        op,
        csystem,
        [op_subsystem_index];
        kwargs...
    )
end

function EmbeddedOperator(op::Symbol, args...; kwargs...)
    @assert op ∈ keys(GATES) "Operator must be a valid gate. See QuantumCollocation.QuantumObjectUtils.GATES dict for available gates."
    return EmbeddedOperator(GATES[op], args...; kwargs...)
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

# ----------------------------------------------------------------------------- #
#                            Subspace Indices                                   #
# ----------------------------------------------------------------------------- #

basis_labels(subsystem_levels::AbstractVector{Int}; baseline=1) =
    kron([""], [string.(baseline:levels - 1 + baseline) for levels ∈ subsystem_levels]...)

basis_labels(subsystem_level::Int; kwargs...) = basis_labels([subsystem_level]; kwargs...)

"""
    get_subspace_indices(subspaces::Vector{<:AbstractVector{Int}}, subsystem_levels::AbstractVector{Int})

Get the indices for the subspace of composite quantum system.

Example: for the two-qubit subspace of two 3-level systems:
```julia
subspaces = [1:2, 1:2]
subsystem_levels = [3, 3]
get_subspace_indices(subspaces, subsystem_levels) == [1, 2, 4, 5]
```

# Arguments

- `subspaces::Vector{<:AbstractVector{Int}}`: Subspaces to get indices for. e.g. `[1:2, 1:2]`.
- `subsystem_levels::AbstractVector{Int}`: Levels of the subsystems in the composite system. e.g. `[3, 3]`. Each element corresponds to a subsystem.
"""
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

"""
    get_subspace_indices(subspace::AbstractVector{Int}, levels::Int)

Get the indices for the subspace of simple, non-composite, quantum system. For example:
```julia
get_subspace_indices([1, 2], 3) == [1, 2]
```

# Arguments
- `subspace::AbstractVector{Int}`: Subspace to get indices for. e.g. `[1, 2]`.
- `levels::Int`: Levels of the subsystem. e.g. `3`.
"""
get_subspace_indices(subspace::AbstractVector{Int}, levels::Int) =
    get_subspace_indices([subspace], [levels])

"""
    get_subspace_indices(levels::AbstractVector{Int}; subspace=1:2, kwargs...)

Get the indices for the subspace of composite quantum system. This is a convenience function that allows to specify the subspace as a range that is constant for every subsystem, which defaults to `1:2`, that is qubit systems.

# Arguments
- `levels::AbstractVector{Int}`: Levels of the subsystems in the composite system. e.g. `[3, 3]`.

# Keyword Arguments
- `subspace::AbstractVector{Int}`: Subspace to get indices for. e.g. `1:2`.
"""
get_subspace_indices(levels::AbstractVector{Int}; subspace=1:2) =
    get_subspace_indices(fill(subspace, length(levels)), levels)

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

get_subspace_leakage_indices(op::EmbeddedOperator) =
    get_subspace_leakage_indices(op.subspace_indices, size(op)[1])

get_iso_vec_subspace_indices(op::EmbeddedOperator) =
    get_iso_vec_subspace_indices(op.subspace_indices, op.subsystem_levels)

get_iso_vec_leakage_indices(op::EmbeddedOperator) =
    get_iso_vec_leakage_indices(op.subspace_indices, op.subsystem_levels)

function get_iso_vec_subspace_indices(
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

function get_iso_vec_leakage_indices(
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

function get_subspace_identity(op::EmbeddedOperator)
    return embed(
        Matrix{ComplexF64}(I(length(op.subspace_indices))),
        op.subspace_indices,
        size(op, 1)
    )
end

# =========================================================================== #

@testitem "Basis labels" begin
    levels = [3, 3]
    labels = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
    @test EmbeddedOperators.basis_labels(levels, baseline=1) == labels

    labels = ["1", "2", "3"]
    @test EmbeddedOperators.basis_labels(3, baseline=1) == labels
    @test EmbeddedOperators.basis_labels([3], baseline=1) == labels

    labels = ["0", "1", "2"]
    @test EmbeddedOperators.basis_labels(3, baseline=0) == labels
    @test EmbeddedOperators.basis_labels([3], baseline=0) == labels

    levels = [2, 2]
    labels = ["00", "01", "10", "11"]
    @test EmbeddedOperators.basis_labels(levels, baseline=0) == labels
end

@testitem "Subspace Indices" begin
    @test get_subspace_indices([1, 2], 3) == [1, 2]
    # 2 * 2 = 4 elements
    @test get_subspace_indices([1:2, 1:2], [3, 3]) == [1, 2, 4, 5]
    # 1 * 1 = 1 element
    @test get_subspace_indices([[2], [2]], [3, 3]) == [5]
    # 1 * 2 = 2 elements
    @test get_subspace_indices([[2], 1:2], [3, 3]) == [4, 5]
end

@testitem "Subspace ENR Indices" begin
    # 00, 01, 02x, 10, 11x, 12x, 20x, 21x, 22x
    @test get_subspace_enr_indices(1, [3, 3]) == [1, 2, 4]
    # 00, 01, 02, 10, 11, 12x, 20, 21x, 22x
    @test get_subspace_enr_indices(2, [3, 3]) == [1, 2, 3, 4, 5, 7]
    # 00, 01, 02, 10, 11, 12, 20, 21, 22x
    @test get_subspace_enr_indices(3, [3, 3]) == [1, 2, 3, 4, 5, 6, 7, 8]
    # 00, 01, 02, 10, 11, 12, 20, 21, 22
    @test get_subspace_enr_indices(4, [3, 3]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
end

@testitem "Subspace Leakage Indices" begin
    # TODO: Implement tests
end

@testitem "Embedded operator" begin
    # Embed X
    op = Matrix{ComplexF64}([0 1; 1 0])
    embedded_op = Matrix{ComplexF64}([0 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0])
    @test embed(op, 1:2, 4) == embedded_op
    embedded_op_struct = EmbeddedOperator(op, 1:2, 4)
    @test embedded_op_struct.operator == embedded_op
    @test embedded_op_struct.subspace_indices == 1:2
    @test embedded_op_struct.subsystem_levels == [4]

    # Properties
    @test size(embedded_op_struct) == size(embedded_op)
    @test size(embedded_op_struct, 1) == size(embedded_op, 1)

    # X^2 = I
    x2 = (embedded_op_struct * embedded_op_struct).operator
    id = get_subspace_identity(embedded_op_struct)
    @test x2 == id

    # Embed X twice
    op2 = op ⊗ op
    embedded_op2 = [
        0  0  0  0  1  0  0  0  0;
        0  0  0  1  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0;
        0  1  0  0  0  0  0  0  0;
        1  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  0
    ]
    subspace_indices = get_subspace_indices([1:2, 1:2], [3, 3])
    @test embed(op2, subspace_indices, 9) == embedded_op2
    embedded_op2_struct = EmbeddedOperator(op2, subspace_indices, [3, 3])
    @test embedded_op2_struct.operator == embedded_op2
    @test embedded_op2_struct.subspace_indices == subspace_indices
    @test embedded_op2_struct.subsystem_levels == [3, 3]
end

@testitem "Embedded operator from system" begin
    CZ = GATES[:CZ]
    a = annihilate(3)
    σ_x = a + a'
    σ_y = -1im*(a - a')
    system = QuantumSystem([σ_x ⊗ σ_x, σ_y ⊗ σ_y])

    op_explicit_qubit = EmbeddedOperator(
        CZ,
        system,
        subspace=get_subspace_indices([1:2, 1:2], [3, 3])
    )
    op_implicit_qubit = EmbeddedOperator(CZ, system)
    # This does not work (implicit puts indicies in 1:4)
    @test op_implicit_qubit.operator != op_explicit_qubit.operator
    # But the ops are the same
    @test unembed(op_explicit_qubit) == unembed(op_implicit_qubit)
    @test unembed(op_implicit_qubit) == CZ
end

@testitem "Embedded operator from composite system" begin
    @test_skip nothing
end

@testitem "Embedded operator kron" begin
    Z = GATES[:Z]
    Ẑ = EmbeddedOperator(Z, 1:2, [4])
    @test unembed(Ẑ ⊗ Ẑ) == Z ⊗ Z
end


end
