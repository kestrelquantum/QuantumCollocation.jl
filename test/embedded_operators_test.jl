"""
Tests: EmbeddedOperators submodule
"""

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

end

