"""
Tests: QuantumSystemUtils submodule
"""


@testitem "Lie algebra basis" begin
    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis = operator_algebra(gen, return_layers=false, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with complete basis and layers
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis, layers = operator_algebra(gen, return_layers=true, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with subspace
    gen = map(A -> kron_from_dict(A, H_ops), ["X"])
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == 1

    # Check 2 qubit with complete basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY","XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 2 qubit with linearly dependent basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY", "XI", "XI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == length(gen)

    # Check 2 qubit with pair of 1-qubit subspaces
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
         ["XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == 2 * (2^2 - 1)
end


@testitem "Lie Algebra reachability" begin
    using LinearAlgebra

    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    target = H_ops["Z"]
    @test is_reachable(gen, target, compute_basis=true, verbose=false)

    # Check 2 qubit with complete basis
    complete_gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY","XI", "YI", "IY", "IX"]
    )
    r = [0, 1, 2, 3, 4]
    r /= norm(r)
    target = exp(-im * sum([θ * H for (H, θ) in zip(complete_gen, r)]))
    @test is_reachable(complete_gen, target, compute_basis=true, verbose=false)

    # Check 2 qubit with incomplete basis
    incomplete_gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY", "IY", "IX"]
    )
    @test !is_reachable(incomplete_gen, target, compute_basis=true, verbose=false)
end

@testitem "Lie Algebra subspace reachability" begin
    # TODO: implement tests
end
