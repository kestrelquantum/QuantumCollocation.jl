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
    XI = GATES[:X] ⊗ GATES[:I]
    IX = GATES[:I] ⊗ GATES[:X]
    YI = GATES[:Y] ⊗ GATES[:I]
    IY = GATES[:I] ⊗ GATES[:Y]
    XX = GATES[:X] ⊗ GATES[:X]
    YY = GATES[:Y] ⊗ GATES[:Y]
    ZI = GATES[:Z] ⊗ GATES[:I]
    IZ = GATES[:I] ⊗ GATES[:Z]
    ZZ = GATES[:Z] ⊗ GATES[:Z]

    complete_gen = [XX+YY, XI, YI, IX, IY]
    incomplete_gen = [XI, ZZ]
    r = [0, 1, 2, 3, 4]
    r /= norm(r)
    R2 = exp(-im * sum([θ * H for (H, θ) in zip(complete_gen, r)]))
    CZ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    CX = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    
    # Pass
    @test is_reachable(complete_gen, R2)
    @test is_reachable(complete_gen, CZ)
    @test is_reachable(complete_gen, CX)
    @test is_reachable(complete_gen, XI)

    # Mostly fail
    @test !is_reachable(incomplete_gen, R2)
    @test !is_reachable(incomplete_gen, CZ)
    @test !is_reachable(incomplete_gen, CX)
    @test is_reachable(incomplete_gen, XI)
end

@testitem "Lie Algebra subspace reachability" begin
    # TODO: implement tests
end
