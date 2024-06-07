"""
Tests: QuantumUtils submodule
"""

@testitem "GATES" begin
    using LinearAlgebra
    @test get_gate(:X) * get_gate(:X) ==  get_gate(:I)
    @test get_gate(:Y) * get_gate(:Y) ==  get_gate(:I)
    @test im*get_gate(:Y)*get_gate(:X) == get_gate(:Z)  # Z = iYX
    @test -im*get_gate(:X)*get_gate(:Y) == get_gate(:Z) # Z = -iXY 
    @test -im*get_gate(:Z)*get_gate(:X) == get_gate(:Y) # Y = -iZX
    @test im*get_gate(:X)*get_gate(:Z) == get_gate(:Y)  # Y = iXZ
    @test im*get_gate(:Z)*get_gate(:Y) == get_gate(:X)  # X = iZY
    @test -im*get_gate(:Y)*get_gate(:Z) == get_gate(:X) # X = -iYZ

    @test isapprox(get_gate(:H)*get_gate(:X)*get_gate(:H)', get_gate(:Z), atol=1e-2) == true # H*X*H† = Z
    @test isapprox(get_gate(:H)*get_gate(:Z)*get_gate(:H)', get_gate(:X), atol=1e-2) == true # H*Z*H† = X
    @test isapprox(get_gate(:H)*get_gate(:Y)*get_gate(:H)', -get_gate(:Y), atol=1e-2) == true # H*Y*H† = -Y
    
    @test isapprox(get_gate(:H)*get_gate(:H), get_gate(:I), atol=1e-2) == true # H² = I
    @test get_gate(:CX)*get_gate(:CX) == I(4) # CNOT² = I
    @test get_gate(:CZ)*get_gate(:CZ) == I(4) # CZ² = I
    @test GATES[:X] ^ 2 == GATES[:I] # X² = I
    @test GATES[:Y] ^ 2 == GATES[:I] # Y² = I
    @test isapprox(GATES[:Z] ^ 2, GATES[:I], atol=1e-2) == true # Z² = I
    @test Int.(round.(real.(GATES[:X]*GATES[:Y] + GATES[:Y]*GATES[:X]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Y]*GATES[:Z] + GATES[:Z]*GATES[:Y]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Z]*GATES[:X] + GATES[:X]*GATES[:Z]))) == zeros(2, 2)
end

@testitem "⊗" begin
    using LinearAlgebra
    @test Int.(real.(GATES[:I] ⊗ GATES[:I])) == I(4)
    @test (GATES[:X] ⊗ GATES[:Y]) ⊗ GATES[:Z] == GATES[:X] ⊗ (GATES[:Y] ⊗ GATES[:Z]) # Associativity
    @test GATES[:X] ⊗ (GATES[:Y] + GATES[:Z]) == GATES[:X] ⊗ GATES[:Y] + GATES[:X] ⊗ GATES[:Z] # Distributivity   
end 

@testitem "Test apply function with Pauli gates" begin
    using LinearAlgebra
    ψ₀ = [1.0 + 0.0im, 0.0 + 0.0im] 
    ψ₁ = [0.0 + 0.0im, 1.0 + 0.0im]
    @test apply(:X, ψ₀) == [0.0 + 0.0im, 1.0 + 0.0im] 
    @test apply(:X, ψ₁) == [1.0 + 0.0im, 0.0 + 0.0im]
    @test apply(:Y, ψ₀) == GATES[:Y] * ψ₀  
    @test apply(:Y, ψ₁) == GATES[:Y] * ψ₁
    @test apply(:H, apply(:H, [1,  0])) == [1, 0]  # H * H = I 
    @test isapprox(apply(:H, apply(:H, apply(:H, [1,  0]))), [0.7071, 0.7071], atol=1e-1) == true
    @test apply(:CX, [1, 0, 0, 0]) == [1, 0, 0, 0]  # CNOT |00⟩ = |00⟩
    @test apply(:CX, [0, 1, 0, 0]) == [0, 1, 0, 0]  # CNOT |01⟩ = |01⟩
    @test apply(:CX, [0, 0, 1, 0]) == [0, 0, 0, 1]  # CNOT |10⟩ = |11⟩
    @test apply(:CX, [0, 0, 0, 1]) == [0, 0, 1, 0]  # CNOT |11⟩ = |10⟩
    @test apply(:CZ, [1, 0, 0, 0]) == [1, 0, 0, 0]  # CZ |00⟩ = |00⟩
    @test apply(:CZ, [0, 1, 0, 0]) == [0, 1, 0, 0]  # CZ |01⟩ = |01⟩
    @test apply(:CZ, [0, 0, 1, 0]) == [0, 0, 1, 0]  # CZ |10⟩ = |10⟩
    @test apply(:CZ, [0, 0, 0, 1]) == [0, 0, 0, -1] # CZ |11⟩ = -|11⟩
end

@testitem "Test qubit_system_state function" begin
    using LinearAlgebra
    @test qubit_system_state("0") == [1, 0]
    @test qubit_system_state("1") == [0, 1]
    @test qubit_system_state("00") == [1, 0, 0, 0]
    @test qubit_system_state("01") == [0, 1, 0, 0]
    @test qubit_system_state("10") == [0, 0, 1, 0]
    @test qubit_system_state("11") == [0, 0, 0, 1]
end

@testitem "Test lift function" begin
    using LinearAlgebra
    U1 = [1 0; 0 1] 
    @test size(lift(U1, 1, 2)) == (4, 4)
    @test size(lift(U1, 2, 2)) == (4, 4)
    @test size(lift(U1, 1, 3)) == (8, 8)
end

@testitem "Test isomorphism utilities" begin
    using LinearAlgebra
    iso_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    @test vec⁻¹([1.0, 2.0, 3.0, 4.0]) == [1.0 3.0; 2.0 4.0]
    @test ket_to_iso([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]
    @test iso_to_ket([1.0, 2.0, 0.0, 0.0]) == [1.0, 2.0]
    @test iso_vec_to_operator(iso_vec) == [1.0 0.0; 0.0 1.0]
    @test iso_vec_to_iso_operator(iso_vec) == [1.0 0.0 -0.0 -0.0; 0.0 1.0 -0.0 -0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    @test operator_to_iso_vec(Complex[1.0 0.0; 0.0 1.0]) == iso_vec
    @test iso_operator_to_iso_vec(iso_vec_to_iso_operator(iso_vec)) == iso_vec
end

@testitem "quantum harmonic oscillator operators" begin
    using LinearAlgebra
    const tol = 1e-10
    levels = 2
    expected₂ = [0.0+0.0im  1.0+0.0im; 
			 0.0+0.0im  0.0+0.0im]
    @test annihilate(levels) == expected₂
    levels = 3
    expected₃ = [0.0+0.0im  1.0+0.0im      0.0+0.0im;
                 0.0+0.0im  0.0+0.0im  1.41421+0.0im;
                 0.0+0.0im  0.0+0.0im      0.0+0.0im]
    @test isapprox(expected₃, annihilate(3), atol=1e-2) == true
    @test annihilate(2) == create(2)' 
    @test annihilate(3) == create(3)'
    @test annihilate(4) == create(4)'
    @test number(3) == create(3)* annihilate(3) 
    @test number(4) == create(4)* annihilate(4)
    @test quad(3) == number(3) * (number(3) - I(3))
    @test quad(4) == number(4) * (number(4) - I(4))
end
