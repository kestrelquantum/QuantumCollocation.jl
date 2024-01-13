"""
Tests: QuantumSystems submodule
"""

@testset "Quantum Systems" begin

end

@testset "Lie algebra utilities" begin
    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis = operator_algebra(gen, return_layers=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with complete basis and layers
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis, layers = operator_algebra(gen, return_layers=true)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with subspace
    gen = map(A -> kron_from_dict(A, H_ops), ["X"])
    basis = operator_algebra(gen)
    @test length(basis) == 1

    # Check 2 qubit with complete basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY","XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 2 qubit with linearly dependent basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY", "XI", "XI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == length(gen)

    # Check 2 qubit with pair of 1-qubit subspaces
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
         ["XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == 2 * (2^2 - 1)
end

@testset "Lie algebra utilities" begin
    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis = operator_algebra(gen, return_layers=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with complete basis and layers
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis, layers = operator_algebra(gen, return_layers=true)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with subspace
    gen = map(A -> kron_from_dict(A, H_ops), ["X"])
    basis = operator_algebra(gen)
    @test length(basis) == 1

    # Check 2 qubit with complete basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY","XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 2 qubit with linearly dependent basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY", "XI", "XI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == length(gen)

    # Check 2 qubit with pair of 1-qubit subspaces
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
         ["XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen)
    @test length(basis) == 2 * (2^2 - 1)
end
