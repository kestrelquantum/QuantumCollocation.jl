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

    function is_unitary(U::AbstractMatrix)
        return U'U ≈ I
    end

    for U in [U_X, U_Y]
        @test is_unitary(U)
    end

    # Test gate fidelity
    @test test_isovec_unitary_fidelity(U_X, U_X) ≈ 1
    @test test_isovec_unitary_fidelity(U_X, U_Y) ≈ 0


    # Test asymmetric fidelity
    U_fn(λ, φ) = [1 -exp(im * λ); exp(im * φ) exp(im * (φ + λ))] / sqrt(2)
    U_1 = U_fn(π/4, π/3)
    U_2 = U_fn(1.5, .33)

    for U in [U_1, U_2]
        @test is_unitary(U)
    end

    @test test_isovec_unitary_fidelity(U_1, U_1) ≈ 1
    @test test_isovec_unitary_fidelity(U_2, U_2) ≈ 1
    @test test_isovec_unitary_fidelity(U_1, U_2) ≈ abs(tr(U_1'U_2)) / 2


    # Test random fidelity
    U_H1 = haar_random(2)
    U_H2 = haar_random(2)

    for U in [U_H1, U_H2]
        @test is_unitary(U)
    end

    @test test_isovec_unitary_fidelity(U_H1, U_H1) ≈ 1
    @test test_isovec_unitary_fidelity(U_H2, U_H2) ≈ 1
    @test test_isovec_unitary_fidelity(U_H1, U_X) ≈ abs(tr(U_H1'U_X)) / 2
    @test test_isovec_unitary_fidelity(U_H1, U_H2) ≈ abs(tr(U_H1'U_H2)) / 2
end


@testitem "Isovec Unitary Fidelity Subspace" begin
        # Test random fidelity
        U_H1 = kron(haar_random(2), haar_random(2))
        U_H2 = kron(haar_random(2), haar_random(2))
        subspace_indices([2, 2])
    
        for U in [U_H1, U_H2]
            @test is_unitary(U)
        end
    
        @test test_isovec_unitary_fidelity(U_H1, U_H1) ≈ 1
        @test test_isovec_unitary_fidelity(U_H2, U_H2) ≈ 1
        @test test_isovec_unitary_fidelity(U_H1, U_X) ≈ abs(tr(U_H1'U_X)) / 2
        @test test_isovec_unitary_fidelity(U_H1, U_H2) ≈ abs(tr(U_H1'U_H2)) / 2


end


@testitem "Isovec Unitary Fidelity Gradient" begin



end