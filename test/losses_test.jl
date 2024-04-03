"""
Tests: Losses submodule
"""

@testitem "Isovec Unitary Fidelity" begin
    using LinearAlgebra

    U_X = get_gate(:X)
    U_Y = get_gate(:Y)

    function test_isovec_unitary_fidelity(U₁::AbstractMatrix, U₂::AbstractMatrix)
        Ũ⃗₁ = operator_to_iso_vec(U₁)
        Ũ⃗₂ = operator_to_iso_vec(U₂)
        return Losses.isovec_unitary_fidelity(Ũ⃗₁, Ũ⃗₂)
    end

    for U in [U_X, U_Y]
        @test U'U ≈ I
    end

    # Test gate fidelity
    @test test_isovec_unitary_fidelity(U_X, U_X) ≈ 1
    @test test_isovec_unitary_fidelity(U_X, U_Y) ≈ 0


    # Test asymmetric fidelity
    U_fn(λ, φ) = [1 -exp(im * λ); exp(im * φ) exp(im * (φ + λ))] / sqrt(2)
    U_1 = U_fn(π/4, π/3)
    U_2 = U_fn(1.5, .33)

    for U in [U_1, U_2]
        @test U'U ≈ I
    end

    @test test_isovec_unitary_fidelity(U_1, U_1) ≈ 1
    @test test_isovec_unitary_fidelity(U_2, U_2) ≈ 1
    @test test_isovec_unitary_fidelity(U_1, U_2) ≈ abs(tr(U_1'U_2)) / 2


    # Test random fidelity
    U_H1 = haar_random(2)
    U_H2 = haar_random(2)

    for U in [U_H1, U_H2]
        @test U'U ≈ I
    end

    @test test_isovec_unitary_fidelity(U_H1, U_H1) ≈ 1
    @test test_isovec_unitary_fidelity(U_H2, U_H2) ≈ 1
    @test test_isovec_unitary_fidelity(U_H1, U_X) ≈ abs(tr(U_H1'U_X)) / 2
    @test test_isovec_unitary_fidelity(U_H1, U_H2) ≈ abs(tr(U_H1'U_H2)) / 2
end


@testitem "Isovec Unitary Fidelity Subspace" begin
    using LinearAlgebra

    function test_isovec_unitary_fidelity(U₁::AbstractMatrix, U₂::AbstractMatrix, args...)
        Ũ⃗₁ = operator_to_iso_vec(U₁)
        Ũ⃗₂ = operator_to_iso_vec(U₂)
        return Losses.isovec_unitary_fidelity(Ũ⃗₁, Ũ⃗₂, args...)
    end

    # Test random fidelity
    test_subspaces = [
        get_subspace_indices([1:2, 1:1], [2, 2]),
        get_subspace_indices([1:2, 2:2], [2, 2]),
    ]

    for ii in test_subspaces
        U_H1 = kron(haar_random(2), haar_random(2))
        U_H1_sub = U_H1[ii, ii]
        U_H2 = kron(haar_random(2), haar_random(2))
        U_H2_sub = U_H2[ii, ii]

        # subspace may not be unitary
        for U in [U_H1, U_H2]
            @test U'U ≈ I
        end

        fid = test_isovec_unitary_fidelity(U_H1, U_H2, (ii, ii))
        fid_sub = test_isovec_unitary_fidelity(U_H1_sub, U_H2_sub)
        @test fid ≈ fid_sub
    end
end


@testitem "Isovec Unitary Fidelity Gradient" begin



end