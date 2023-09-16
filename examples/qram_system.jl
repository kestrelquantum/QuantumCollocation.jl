using QuantumCollocation
import QuantumCollocation: lift
using NamedTrajectories
using LinearAlgebra

find(qubits, q) = findfirst(qubits .== q)

"""
    QRAMSystem(; qubits, drives, levels, α, χ)

Constructs a `QuantumSystem` for the QRAM experiment.

f_ge: [4130.2001349834745, 3456.366348279435, 4785.140528992779, 4398.3047410188055] # [MHz]
f_ef: [3904.734739900621, 3356.000127770774, 4595.4955356447535, 4225.668529012362] # [MHz]
gs: [54.98834995, 55.21613685, 44.21773961, 7.06638874, 2.63846655, 6.19737439] # [MHz] g01, g12, g13, g02, g03, g23 from theoretical comparison for ZZ shift

"""
function QRAMSystem(;
    qubits=[1,2,3,4],
    drives=[1,2,3,4],
    levels=fill(3, length(qubits)),
    f_ge=[4130.2001349834745, 3456.366348279435, 4785.140528992779, 4398.3047410188055] * 1e-3, # [GHz]
    f_ef=[3904.734739900621, 3356.000127770774, 4595.4955356447535, 4225.668529012362] * 1e-3, # [GHz]
    ω=f_ge,
    ω_d_index=1,
    ω_d=ω[ω_d_index],
    α=f_ge-f_ef, # GHz (linear units)
    dispersive=false,
    χ=Symmetric([
        0 -5.10982939 -0.18457118 -0.50235316;
        0       0     -0.94914758 -1.07618574;
        0       0           0     -0.44607489;
        0       0           0           0
    ]) * 1e-3, # GHz (linear units),
    gᵢⱼ=Symmetric([
        0 54.98834995  7.06638874  2.63846655;
        0       0      55.21613685 44.21773961;
        0       0            0     6.19737439;
        0       0            0          0
    ]) * 1e-3, # GHz (linear units),
)
    @assert length(levels) == length(qubits)
    @assert unique(qubits) == qubits
    @assert unique(drives) == drives
    @assert all(drive ∈ qubits for drive ∈ drives)

    â_dag(i) = create(levels[find(qubits, i)])
    â(i) = annihilate(levels[find(qubits, i)])

    if dispersive
        H_q = sum(
            -α[i] / 2 * lift(â_dag(i)^2 * â(i)^2, find(qubits, i), levels)
                for i ∈ qubits
        )
    else
        δ = ω .- ω_d
        H_q = sum(
            δ[i] * lift(â_dag(i) * â(i), find(qubits, i), levels) -
            α[i] / 2 * lift(â_dag(i)^2 * â(i)^2, find(qubits, i), levels)
                for i ∈ qubits
        )
    end

    if length(qubits) == 1
        H_drift = 2π * H_q
    else
        if dispersive
            H_c = sum(
                χ[i, j] *
                lift(â_dag(i) * â(i), find(qubits, i), levels) *
                lift(â_dag(j) * â(j), find(qubits, j), levels)
                    for i ∈ qubits, j ∈ qubits if j > i
            )
        else
            H_c = sum(
                gᵢⱼ[i, j] * (
                    lift(â_dag(i), i, levels) *
                    lift(â(j), j, levels) +
                    lift(â_dag(j), j, levels) *
                    lift(â(i), i, levels)
                ) for i ∈ qubits, j ∈ qubits if j > i
            )
        end
        H_drift = 2π * (H_q + H_c)
    end


    H_d_real(i) = 1 / 2 * lift(â_dag(i) + â(i), find(qubits, i), levels)

    H_d_imag(i) = 1 / 2im * lift(â(i) - â_dag(i), find(qubits, i), levels)

    H_drives::Vector{Matrix{ComplexF64}} =
        vcat([[H_d_real(i), H_d_imag(i)] for i ∈ drives]...)

    H_drives .*= 2π

    @assert all(H == H' for H ∈ [H_drift, H_drives...])

    return QuantumSystem(H_drift, H_drives)
end
