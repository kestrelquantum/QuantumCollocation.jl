using Pico
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using Distributions
using Manifolds

max_iter = 200
linear_solver = "pardiso"

U_init = 1.0 * I(8)

e0 = [1, 0]
e1 = [0, 1]

Id = GATES[:I]

e00 = e0 * e0'
e11 = e1 * e1'
e01 = e0 * e1'
e10 = e1 * e0'

U_goal_analytic =
    e00 ⊗ Id ⊗ e00 +
    e10 ⊗ Id ⊗ e01 +
    e01 ⊗ Id ⊗ e10 +
    e11 ⊗ Id ⊗ e11

U_goal = Matrix{ComplexF64}(U_goal_analytic)

a_dag = create(2)
a = annihilate(2)

ωs = 2π * [5.18, 5.12, 5.06]
ω_d = 2π * 5.12

ξs = 2π * fill(0.340, 3) # ξ = 0.340 GHz

J_12 = 5.0e-3
J_23 = 5.0e-3


H_drift = sum(
    (ωs[q] - ω_d) * lift(a_dag, q, 3) * lift(a, q, 3) -
    ξs[q] / 2 * lift(a_dag, q, 3) * lift(a_dag, q, 3) * lift(a, q, 3) * lift(a, q, 3)
        for q = 1:3
)

# dispersive coupling
H_drift +=
    J_12 * (
        lift(a_dag, 1, 3) * lift(a, 2, 3) +
        lift(a, 1, 3) * lift(a_dag, 2, 3)
    ) +
    J_23 * (
        lift(a_dag, 2, 3) * lift(a, 3, 3) +
        lift(a, 2, 3) * lift(a_dag, 3, 3)
    )

H_drives_Re = [lift(a, j, 3) + lift(a_dag, j, 3) for j = 1:3]
H_drives_Im = [1im * (lift(a, j, 3) - lift(a_dag, j, 3)) for j = 1:3]

H_drives = vcat(H_drives_Re, H_drives_Im)
n_drives = length(H_drives)

system = QuantumSystem(H_drift, H_drives)

U_dim = *(size(U_init)...)

N = size(U_init, 1)

n = 2 * N

n_controls = n * (n - 1) ÷ 2


Ũ⃗_init = operator_to_iso_vec(U_init)
Ũ⃗_goal = operator_to_iso_vec(U_goal)

Ũ⃗_dim = length(Ũ⃗_init)

# -------------------------------------------
# Setting up geodesic control problem
# -------------------------------------------

load_trajectory = false

if load_trajectory
    traj_path = "examples/scripts/trajectories/three_qubits/swap_gate/geodesic_T_50_dt_4.0_Δt_min_2.0_Δt_max_4.0_R_G_0.01_Q_100.0_F_1.0000471425548512_00000.jld2"
    traj = NamedTrajectories.load_traj(traj_path)
else
    T = 50
    dt = 4.0
    Δt_min = 0.5 * dt
    Δt_max = 1.0 * dt


    Ũ⃗, G⃗_geodesic = unitary_geodesic(U_goal, T)

    Ũ⃗[:, 1]
    Ũ⃗[:, 1] |> iso_vec_to_operator |> U -> unitary_fidelity(U, U_goal)

    G⃗ = G⃗_geodesic + rand(Normal(0, 0.01), size(G⃗_geodesic))

    Δt = fill(dt, 1, T)

    comps = (
        Ũ⃗ = Ũ⃗,
        G⃗ = G⃗,
        Δt = Δt
    )

    bounds = (
        Δt = (Δt_min, Δt_max),
    )

    initial = (
        Ũ⃗ = Ũ⃗_init,
    )

    final = (;
    )

    goal = (
        Ũ⃗ = Ũ⃗_goal,
    )

    traj = NamedTrajectory(
        comps;
        controls=(:G⃗, :Δt),
        dt=dt,
        dynamical_dts=true,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end

P = FourthOrderPade(system)

@views function f(zₜ, zₜ₊₁)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components.Ũ⃗]
    Ũ⃗ₜ = zₜ[traj.components.Ũ⃗]
    G⃗ₜ = zₜ[traj.components.G⃗]
    Δtₜ = zₜ[traj.components.Δt][1]

    G = skew_symmetric(G⃗ₜ, n)

    δŨ⃗ = P(Ũ⃗ₜ₊₁, Ũ⃗ₜ, Δtₜ; G_additional=G, operator=true)
    return δŨ⃗
end

Q = 100.0

loss = :UnitaryInfidelityLoss

J = QuantumObjective(:Ũ⃗, traj, loss, Q)

R_G⃗ = 1e-2

J += QuadraticRegularizer(:G⃗, traj, R_G⃗ * ones(n_controls))


options = Options(
    max_iter=max_iter,
    linear_solver=linear_solver,
)

initprob = QuantumControlProblem(system, traj, J, f;
    options=options,
)

println("Solving geodesic control problem...")
solve!(initprob)

F = unitary_fidelity(initprob.trajectory[end].Ũ⃗, initprob.trajectory.goal.Ũ⃗)

println()
println("Initial geodesic problem solved...")
println()
println("Final fidelity: $F")
println()

# save the trajectory
experiment = "geodesic_T_$(T)_dt_$(dt)_Δt_min_$(Δt_min)_Δt_max_$(Δt_max)_R_G_$(R_G⃗)_Q_$(Q)_F_$(F)"
save_dir = "examples/scripts/trajectories/three_qubits/swap_gate"
save_path = generate_file_path("jld2", experiment, save_dir)
save(save_path, initprob.trajectory)
