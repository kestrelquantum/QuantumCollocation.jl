using Revise
using QuantumCollocation
using NamedTrajectories
using TrajectoryIndexingUtils
using BenchmarkTools
using LinearAlgebra

T = 100
H_drift = GATES[:Z]
H_drives = [GATES[:X], GATES[:Y]]
n_drives = length(H_drives)

system = QuantumSystem(H_drift, H_drives)

U_init = I(2)
U_goal = GATES[:X]

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

P = UnitaryPadeIntegrator(system, :Ũ⃗, :a)
D = DerivativeIntegrator(:a, :da, Z)

f = [P, D]

dynamics = QuantumDynamics(f, Z)

# test dynamics jacobian
shape = (Z.dims.states * (Z.T - 1), Z.dim * Z.T)

@btime J_dynamics, J_struc = dynamics.∂F(Z.datavec), dynamics.∂F_structure

shape = (Z.dim * Z.T, Z.dim * Z.T)

μ = ones(Z.dims.states * (Z.T - 1))

@btime HoL_dynamics, HoLstruc = dynamics.μ∂²F(Z.datavec, μ), dynamics.μ∂²F_structure
