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
