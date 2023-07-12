using Revise
using QuantumCollocation
using NamedTrajectories
using TrajectoryIndexingUtils
using BenchmarkTools
using LinearAlgebra

T = 5
H_drift = kron(GATES[:Z], GATES[:X], GATES[:X], GATES[:X])
H_drives = [kron(GATES[:X], I(2), I(2), I(2)), kron(GATES[:Y], GATES[:Y], I(2), GATES[:Y])]
n_drives = length(H_drives)

system = QuantumSystem(H_drift, H_drives)

U_init = I(16)
U_goal = kron(GATES[:X], GATES[:X], GATES[:X], GATES[:X])

Ũ⃗_init = operator_to_iso_vec(U_init)
Ũ⃗_goal = operator_to_iso_vec(U_goal)

dt = 0.1

Z = NamedTrajectory(
    (
        Ũ⃗ = unitary_geodesic(U_goal, T),
        a = randn(n_drives, T),
        g = randn(n_drives, T),
        da = randn(n_drives, T),
        Δt = fill(dt, 1, T),
    ),
    controls=(:da,),
    timestep=:Δt,
    goal=(Ũ⃗ = Ũ⃗_goal,)
)

P = UnitaryPadeIntegrator(system, :Ũ⃗, (:a, :g))

#P(Z.datavec[slice(2, Z.dim)], Z.datavec[slice(3, Z.dim)], Z)

z_2 = Z.datavec[slice(2, Z.dim)]
z_3 = Z.datavec[slice(3, Z.dim)]

[z_2[Z.components[:g]]... for s ∈ P.drive_symb]