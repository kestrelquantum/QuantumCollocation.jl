using QuantumOptics
using QuantumToolbox

function QuantumOpticsSystem(
    Op_drift::QuantumObject{SparseMatrixCSC{ComplexF64, Int64}, OperatorQuantumObject},
    Op_drives::Vector{QuantumObject{SparseMatrixCSC{ComplexF64, Int64}, OperatorQuantumObject}};
)
    # Check for Hermitian matrices
    @assert QuantumToolbox.ishermitian(Op_drift) "Non-Hermitian Hamiltonian provided."
    @assert all([
        QuantumToolbox.ishermitian(Op_drive) for Op_drive in Op_drives
        ]) "Non-Hermitian Hamiltonian provided."

    # Extract matrices
    H_drift::SparseMatrixCSC{ComplexF64, Int64} = get_data(Op_drift) 
    H_drives::Vector{SparseMatrixCSC{ComplexF64, Int64}} = [get_data(Op_drive) for Op_drive in Op_drives]

    return QuantumSystem(
        H_drift,
        H_drives;
        constructor=QuantumOpticsSystem,
    )
end