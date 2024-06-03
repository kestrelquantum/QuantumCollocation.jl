"""
Tests: QuantumUtils submodule
"""

@testitem "GATES" begin
    using LinearAlgebra
    @test get_gate(:X) * get_gate(:X) ==  get_gate(:I)
    @test get_gate(:Y) * get_gate(:Y) ==  get_gate(:I)

    # Cayley Pauli Table
    @test im*get_gate(:Y)*get_gate(:X) == get_gate(:Z)  # Z = iYX 
    @test -im*get_gate(:X)*get_gate(:Y) == get_gate(:Z) # Z = -iXY 
    @test -im*get_gate(:Z)*get_gate(:X) == get_gate(:Y) # Y = -iZX
    @test im*get_gate(:X)*get_gate(:Z) == get_gate(:Y)  # Y = iXZ
    @test im*get_gate(:Z)*get_gate(:Y) == get_gate(:X)  # X = iZY
    @test -im*get_gate(:Y)*get_gate(:Z) == get_gate(:X) # X = -iYZ

    # There is some imprecision in floating-point arithmetic, 
    # which is why we use a tolerance to check for approximate equality.

    #H*X*H† = Z
    @test isapprox(GATES[:H]*GATES[:X]*GATES[:H]', GATES[:Z], atol=1e-2) == true
    @test isapprox(get_gate(:H)*get_gate(:X)*get_gate(:H)', get_gate(:Z), atol=1e-2) == true

    #H*Z*H† = X
    @test isapprox(GATES[:H]*GATES[:Z]*GATES[:H]', GATES[:X], atol=1e-2) == true
    @test isapprox(get_gate(:H)*get_gate(:Z)*get_gate(:H)', get_gate(:X), atol=1e-2) == true

    #H*Y*H† = -Y
    @test isapprox(GATES[:H]*GATES[:Y]*GATES[:H]', -GATES[:Y], atol=1e-2) == true
    @test isapprox(get_gate(:H)*get_gate(:Y)*get_gate(:H)', -get_gate(:Y), atol=1e-2) == true
    
    #H*H = I
    @test isapprox(GATES[:H]*GATES[:H], GATES[:I], atol=1e-2) == true
    @test isapprox(get_gate(:H)*get_gate(:H), get_gate(:I), atol=1e-2) == true
    
    #CNOT*CNOT = I
    @test GATES[:CX]*GATES[:CX] == I(4)
    @test get_gate(:CX)*get_gate(:CX) == I(4)

    @test Int.(round.(real.(GATES[:H]*GATES[:H]))) == GATES[:I]
    # X² = I
    @test GATES[:X] ^ 2 == GATES[:I]
    @test isapprox(GATES[:X] ^ 2, GATES[:I], atol=1e-2) == true
    # Y² = I
    @test GATES[:Y] ^ 2 == GATES[:I]
    @test isapprox(GATES[:Y] ^ 2, GATES[:I], atol=1e-2) == true
    # Z² = I
    @test isapprox(GATES[:Z] ^ 2, GATES[:I], atol=1e-2) == true

    @test Int.(round.(real.(GATES[:X]*GATES[:Y] + GATES[:Y]*GATES[:X]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Y]*GATES[:Z] + GATES[:Z]*GATES[:Y]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Z]*GATES[:X] + GATES[:X]*GATES[:Z]))) == zeros(2, 2)
end

@testitem "⊗" begin
    using LinearAlgebra
    @test Int.(real.(GATES[:I] ⊗ GATES[:I])) == I(4)
    #Associativity
    @test (GATES[:X] ⊗ GATES[:Y]) ⊗ GATES[:Z] == GATES[:X] ⊗ (GATES[:Y] ⊗ GATES[:Z])
    #Distributivity    
    @test GATES[:X] ⊗ (GATES[:Y] + GATES[:Z]) == GATES[:X] ⊗ GATES[:Y] + GATES[:X] ⊗ GATES[:Z]
end 

@testitem "Test apply function with Pauli gates" begin
    using LinearAlgebra
    # Define the initial state
    ψ₀ = [1.0 + 0.0im, 0.0 + 0.0im]  # Example initial state for a single qubit
    ψ₁ = [0.0 + 0.0im, 1.0 + 0.0im]

    # Apply the X gate
    @test apply(:X, ψ₀) == [0.0 + 0.0im, 1.0 + 0.0im] # Applying X gate should flip |0⟩ to |1⟩
    @test apply(:X, ψ₁) == [1.0 + 0.0im, 0.0 + 0.0im] # Applying X gate should flip |1⟩ to |0⟩

    # Define the Y gate matrix manually
    Y = [0.0 + 0.0im 0.0 - 1.0im;
         0.0 + 1.0im 0.0 + 0.0im]
    @test apply(:Y, ψ₀) == GATES[:Y] * ψ₀  # Apply the Y gate to the state |0⟩
    @test apply(:Y, ψ₁) == GATES[:Y] * ψ₁  # Apply the Y gate to the state |1⟩
   
    # H * H = I 
    @test apply(:H, apply(:H, [1,  0])) == [1, 0]
    @test isapprox(apply(:H, apply(:H, apply(:H, [1,  0]))), [0.7071, 0.7071], atol=1e-1) == true
    
    # CNOT |00⟩ = |00⟩
    @test apply(:CX, [1, 0, 0, 0]) == [1, 0, 0, 0] 
    # CNOT |10⟩ = |10⟩
    @test apply(:CX, [0, 1, 0, 0]) == [0, 1, 0, 0] 
    # CNOT |01⟩ = |11⟩
    @test apply(:CX, [0, 0, 1, 0]) == [0, 0, 0, 1] 
    # CNOT |11⟩ = |10⟩
    @test apply(:CX, [0, 0, 0, 1]) == [0, 0, 1, 0] 

end

# Define a test set for qubit_system_state function
@testitem "Test qubit_system_state function" begin
    using LinearAlgebra
    @test qubit_system_state("0") == [1, 0]
    @test qubit_system_state("1") == [0, 1]
    @test qubit_system_state("00") == [1, 0, 0, 0]
    @test qubit_system_state("01") == [0, 1, 0, 0]
    @test qubit_system_state("10") == [0, 0, 1, 0]
    @test qubit_system_state("11") == [0, 0, 0, 1]
end

# Define a test set for lift function
@testitem "Test lift function" begin
    using LinearAlgebra
    #Lift 2x2 identity matrix U to the first qubit in a 2-qubit system
    U1 = [1 0; 0 1]  # Identity matrix
    @test size(lift(U1, 1, 2)) == (4, 4) 

    #Lift 2x2 identity matrix U to the second qubit in a 2-qubit system
    @test size(lift(U1, 2, 2)) == (4, 4) 

    #Lift 2x2 identity matrix U to the first qubit in a 4-qubit system
    @test size(lift(U1, 1, 3)) == (8, 8)
end

# Define test cases for isomorphism utilities
@testitem "Test isomorphism utilities" begin
    using LinearAlgebra
    # vector for conversion
    iso_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    # vec⁻¹ function: Converts a vector into a square matrix.
    @test vec⁻¹([1.0, 2.0, 3.0, 4.0]) == [1.0 3.0; 2.0 4.0]

    # ket_to_iso function: Converts a ket vector into a complex vector with real and imaginary parts.
    @test ket_to_iso([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]

    # iso_to_ket function: Converts a complex vector with real and imaginary parts into a ket vector.
    @test iso_to_ket([1.0, 2.0, 0.0, 0.0]) == [1.0, 2.0]

    # iso_vec_to_operator function: Converts a complex vector into a complex matrix representing an operator.
    @test iso_vec_to_operator(iso_vec) == [1.0 0.0; 0.0 1.0]

    # iso_vec_to_iso_operator function: Converts a complex vector into a real matrix representing an isomorphism operator.
    @test iso_vec_to_iso_operator(iso_vec) == [1.0 0.0 -0.0 -0.0; 0.0 1.0 -0.0 -0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

    # operator_to_iso_vec function: Converts a complex matrix representing an operator into a complex vector.
    @test operator_to_iso_vec(Complex[1.0 0.0; 0.0 1.0]) == iso_vec
    
    # iso_operator_to_iso_vec function: Converts a real matrix representing an isomorphism operator into a complex vector.
    @test iso_operator_to_iso_vec(iso_vec_to_iso_operator(iso_vec)) == iso_vec
end


# Define test cases for quantum harmonic oscillator operators
@testitem "quantum harmonic oscillator operators" begin
    using LinearAlgebra
    # Define a tolerance for floating-point comparisons
    const tol = 1e-10
    # Test with 2 levels
    levels = 2
    # For 2 levels, the expected matrix should have a 1 at (1,2) position
    # since the annihilation operator acts to lower the energy level by one
    expected₂ = [0.0+0.0im  1.0+0.0im; 
			 0.0+0.0im  0.0+0.0im]
    @test annihilate(levels) == expected₂
    # Test with 3 levels
    levels = 3
    # For 3 levels, the expected matrix should have a square root of 2 at (2,3) position
    # since the annihilation operator acts to lower the energy level by one
    expected₃ = [0.0+0.0im  1.0+0.0im      0.0+0.0im;
                 0.0+0.0im  0.0+0.0im  1.41421+0.0im;
                 0.0+0.0im  0.0+0.0im      0.0+0.0im]
    # There is some imprecision in floating-point arithmetic, 
    # which is why we use a tolerance to check for approximate equality
    @test isapprox(expected₃, annihilate(3), atol=1e-2) == true
  
    # `create` returns the adjoint of `annihilate` for the specified levels.
    @test annihilate(2) == create(2)' 
    @test annihilate(3) == create(3)'
    @test annihilate(4) == create(4)'
   
    # `number = a'a` computes the number operator for a system with `levels` levels.
    @test number(3) == create(3)* annihilate(3) 
    @test number(4) == create(4)* annihilate(4)
    
    # `Quad = `n(n - I)` for a quantum system with `levels` energy levels, where `n` is the number operator and `I` is the identity operator.
    @test quad(3) == number(3) * (number(3) - I(3))
    @test quad(4) == number(4) * (number(4) - I(4))
end
