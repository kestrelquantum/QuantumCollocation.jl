@doc raw"""
    RydbergChainSystem(;
        C::Float64=862690*2π,
        distance::Float64=10.0, # μm
        cutoff_order::Int=2, # 1 is nearest neighbor, 2 is next-nearest neighbor, etc.
        local_detune::Bool=false, # If true, include one local detuning pattern.
    ) -> QuantumSystem

Returns a `QuantumSystem` object for the Rydberg atom chain in the spin basis
    |g⟩ = |0⟩ = [1, 0], |r⟩ = |1⟩ = [0, 1].

```math
H = \sum_i 0.5*\Omega_i(t)\cos(\phi_i(t)) \sigma_i^x - 0.5*\Omega_i(t)\sin(\phi_i(t)) \sigma_i^y - \sum_i \Delta_i(t)n_i + \sum_{i<j} \frac{C}{|i-j|^6} n_i n_j
```


# Keyword Arguments
- `C`: The Rydberg interaction strength in MHz*μm^6.
- `distance`: The distance between atoms in μm.
- `cutoff_order`: Interaction range cutoff, 1 is nearest neighbor, 2 is next nearest neighbor.
- `local_detune`: If true, include one local detuning pattern.
"""
function generate_pattern(N::Int, i::Int)
    # Create an array filled with 'I'
    qubits = fill('I', N)
    # Insert 'n' at position i and i+1, ensuring it doesn't exceed the array bounds
    if i <= N && i+1 <= N
        qubits[i] = 'n'
        qubits[i+1] = 'n'
    end
    return join(qubits)
end
function generate_pattern_with_gap(N::Int, i::Int, gap::Int)
    # Create an array filled with 'I'
    qubits = fill('I', N)
    # Insert 'n' at position i and i+gap+1, ensuring it doesn't exceed the array bounds
    if i <= N && i+gap+1 <= N
        qubits[i] = 'n'
        qubits[i+gap+1] = 'n'
    end
    return join(qubits)
end
"""
Embed a character into a string at a specific position.
"""
function lift(x::Char,i::Int, N::Int)
    qubits = fill('I', N)
    qubits[i] = x
    return join(qubits)
end


function RydbergChainSystem(;
    N::Int=3, # number of atoms
    C::Float64=862690*2π,
    distance::Float64=8.7, # μm
    cutoff_order::Int=1, # 1 is nearest neighbor, 2 is next-nearest neighbor, etc.
    local_detune::Bool=false,
    all2all::Bool=true,
    ignore_Y_drive::Bool=false,
)
    PAULIS = Dict("I" => [1 0; 0 1], "X" => [0 1; 1 0], "Y" => [0 -im; im 0], "Z" => [1 0; 0 -1], "n" => [0 0; 0 1])
    if all2all
        H_drift = zeros(ComplexF64, 2^N, 2^N)
        for gap in 0:N-2
            for i in 1:N-gap-1
                H_drift += C*kron_from_dict(generate_pattern_with_gap(N,i,gap),PAULIS)/(((gap+1)*distance)^6)
            end
        end
    else
        if cutoff_order == 1
            H_drift = sum([C*kron_from_dict(generate_pattern(N,i),PAULIS)/(distance^6) for i in 1:N-1])
        elseif cutoff_order == 2
            H_drift = sum([C*kron_from_dict(generate_pattern(N,i),PAULIS)/(distance^6) for i in 1:N-1])
            H_drift += sum([C*kron_from_dict(generate_pattern_with_gap(N,i,1),PAULIS)/((2*distance)^6) for i in 1:N-2])
        else
            error("Higher cutoff order not supported")
        end
    end
    H_drives = Matrix{ComplexF64}[]
    # Add global X drive
    Hx = sum([0.5*kron_from_dict(lift('X',i,N), PAULIS) for i in 1:N])
    push!(H_drives, Hx)
    if !ignore_Y_drive
        # Add global Y drive
        Hy = sum([0.5*kron_from_dict(lift('Y',i,N), PAULIS) for i in 1:N])
        push!(H_drives, Hy)
    end
    # Add global detuning
    H_detune = -sum([kron_from_dict(lift('n',i,N), PAULIS) for i in 1:N])
    push!(H_drives, H_detune)
    params = Dict{Symbol, Any}(
        :N => N,
        :C => C,
        :distance => distance,
        :cutoff_order => cutoff_order,
        :local_detune => local_detune,
        :all2all => all2all,
    )
    return QuantumSystem(
        H_drift,
        H_drives;
        constructor=RydbergChainSystem,
        params=params,
    )
end
