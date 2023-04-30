using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions

# setting maximum number of iterations
max_iter = 500

# defining levels for single qubit system
n_levels = 2

# definining pauli matrices
σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

# definining initial value of wavefunction
ψ_init_1 = [1.0, 0.0]
ψ_init_2 = [0.0, 1.0]
ψ_init_3 = (ψ_init_1 + im * ψ_init_2) / √2
ψ_init_4 = (ψ_init_1 - ψ_init_2) / √2

# gate to be applied
gate = σx

# defining goal value of wavefunction
ψ_goal_1 = gate * ψ_init_1
ψ_goal_2 = gate * ψ_init_2
ψ_goal_3 = gate * ψ_init_3
ψ_goal_4 = gate * ψ_init_4

# defining pauli ladder operators
σ₋ = 0.5 * (σx + 1im * σy)
σ₊ = 0.5 * (σx - 1im * σy)

# defining drive Hamiltonians for system
H_drives = [σ₋ + σ₊, 1im * (σ₋ - σ₊), σz]

# defining drift hamiltonian to be zeros (not used in this system)
H_drift = zeros(n_levels, n_levels)

# building quantum system
system = QuantumSystem(H_drift, H_drives)

# convert wavefunctions to isomorphic vector representation
ψ̃_init_1 = ket_to_iso(ψ_init_1)
ψ̃_goal_1 = ket_to_iso(ψ_goal_1)

ψ̃_init_2 = ket_to_iso(ψ_init_2)
ψ̃_goal_2 = ket_to_iso(ψ_goal_2)

ψ̃_init_3 = ket_to_iso(ψ_init_3)
ψ̃_goal_3 = ket_to_iso(ψ_goal_3)

ψ̃_init_4 = ket_to_iso(ψ_init_4)
ψ̃_goal_4 = ket_to_iso(ψ_goal_4)

# getting dimension of the isomorphic vector representation
ψ̃_dim = length(ψ̃_init_1)

# defining time parameters
max_duration = 10 # μs = 10e-6 s
T = 100
dt = max_duration / T
dt_max = 2.0 * dt
dt_min = 0.5 * dt

# boudns on controls
# TODO: implement nonlinear constraints for abs val of γ ∈ C
γ_bound = 0.3 * 2π # rad / μs = 2π * 3e5 rad / s
α_bound = 0.1 * 2π # rad / μs = 2π * 1e5 rad / s

# dimensions of controls
γ_dim = 2
α_dim = 1

# initialization distributions for controls
γ_dist = Uniform(-γ_bound, γ_bound)
α_dist = Uniform(-α_bound, α_bound)

γ = foldr(hcat, [zeros(γ_dim), rand(γ_dist, γ_dim, T - 2), zeros(γ_dim)])
α = foldr(hcat, [zeros(α_dim), rand(α_dist, α_dim, T - 2), zeros(α_dim)])
Δt = dt * ones(1, T)

u = vcat(γ, α)

ψ̃1 = rollout(ψ̃_init_1, u, Δt, system)
ψ̃2 = rollout(ψ̃_init_2, u, Δt, system)
ψ̃3 = rollout(ψ̃_init_3, u, Δt, system)
ψ̃4 = rollout(ψ̃_init_4, u, Δt, system)

# defining components for trajectory
comps = (
    ψ̃1 = ψ̃1,
    ψ̃2 = ψ̃2,
    ψ̃3 = ψ̃3,
    ψ̃4 = ψ̃4,
    γ = γ,
    dγ = randn(γ_dim, T),
    ddγ = randn(γ_dim, T),
    α = α,
    dα = randn(α_dim, T),
    ddα = randn(α_dim, T),
    Δt = Δt
)

ddu_bound = 2e-1

# defining bounds
bounds = (
    γ = fill(γ_bound, γ_dim),
    α = fill(α_bound, α_dim),
    ddγ = fill(ddu_bound, γ_dim),
    ddα = fill(ddu_bound, α_dim),
    Δt = (dt_min, dt_max)
)

# defining initial values
initial = (
    ψ̃1 = ψ̃_init_1,
    ψ̃2 = ψ̃_init_2,
    ψ̃3 = ψ̃_init_3,
    ψ̃4 = ψ̃_init_4,
    γ = zeros(γ_dim),
    α = zeros(α_dim)
)

# defining final values
final = (
    γ = zeros(γ_dim),
    α = zeros(α_dim),
)

# defining goal states
goal = (
    ψ̃1 = ψ̃_goal_1,
    ψ̃2 = ψ̃_goal_2,
    ψ̃3 = ψ̃_goal_3,
    ψ̃4 = ψ̃_goal_4
)

# creating named trajectory
traj = NamedTrajectory(
    comps;
    controls=(:ddγ, :ddα, :Δt),
    timestep=dt,
    dynamical_dts=true,
    bounds=bounds,
    initial=initial,
    final=final,
    goal=goal
)

# creating fourth order pade integrator
P = FourthOrderPade(system)

# defining dynamics function
function f(zₜ, zₜ₊₁)
    # wavefunction states
    ψ̃1ₜ₊₁ = zₜ₊₁[traj.components.ψ̃1]
    ψ̃1ₜ = zₜ[traj.components.ψ̃1]

    ψ̃2ₜ₊₁ = zₜ₊₁[traj.components.ψ̃2]
    ψ̃2ₜ = zₜ[traj.components.ψ̃2]

    ψ̃3ₜ₊₁ = zₜ₊₁[traj.components.ψ̃3]
    ψ̃3ₜ = zₜ[traj.components.ψ̃3]

    ψ̃4ₜ₊₁ = zₜ₊₁[traj.components.ψ̃4]
    ψ̃4ₜ = zₜ[traj.components.ψ̃4]

    # γ states + augmented states + controls
    γₜ₊₁ = zₜ₊₁[traj.components.γ]
    γₜ = zₜ[traj.components.γ]

    dγₜ₊₁ = zₜ₊₁[traj.components.dγ]
    dγₜ = zₜ[traj.components.dγ]

    ddγₜ = zₜ[traj.components.ddγ]

    # α states + augmented states + controls
    αₜ₊₁ = zₜ₊₁[traj.components.α]
    αₜ = zₜ[traj.components.α]

    dαₜ₊₁ = zₜ₊₁[traj.components.dα]
    dαₜ = zₜ[traj.components.dα]

    ddαₜ = zₜ[traj.components.ddα]

    # time step
    Δtₜ = zₜ[traj.components.Δt][1]

    # controls for pade integrator
    uₜ = [γₜ; αₜ]
    δψ̃1 = P(ψ̃1ₜ₊₁, ψ̃1ₜ, uₜ, Δtₜ)
    δψ̃2 = P(ψ̃2ₜ₊₁, ψ̃2ₜ, uₜ, Δtₜ)
    δψ̃3 = P(ψ̃3ₜ₊₁, ψ̃3ₜ, uₜ, Δtₜ)
    δψ̃4 = P(ψ̃4ₜ₊₁, ψ̃4ₜ, uₜ, Δtₜ)

    # γ dynamics
    δγ = γₜ₊₁ - γₜ - dγₜ * Δtₜ
    δdγ = dγₜ₊₁ - dγₜ - ddγₜ * Δtₜ

    # α dynamics
    δα = αₜ₊₁ - αₜ - dαₜ * Δtₜ
    δdα = dαₜ₊₁ - dαₜ - ddαₜ * Δtₜ

    return vcat(δψ̃1, δψ̃2, δψ̃3, δψ̃4, δγ, δdγ, δα, δdα)
end

# quantum objective weight parameter
Q = 1.0e2

# defining infidelity loss
loss = :InfidelityLoss

# creating quantum objective
J = QuantumObjective((:ψ̃1, :ψ̃2, :ψ̃3, :ψ̃4), traj, loss, Q)

# regularization parameters
R_ddγ = 1e-4
R_ddα = 1e-4

# addign quadratic regularization term on γ to the objective
J += QuadraticRegularizer(:ddγ, traj, R_ddγ * ones(γ_dim))

# adding quadratic regularization term on
J += QuadraticRegularizer(:ddα, traj, R_ddα * ones(α_dim))

# Ipopt options
options = Options(
    max_iter=max_iter,
)

# defining quantum control problem
prob = QuantumControlProblem(system, traj, J, f;
    options=options,
)

# plotting directory
plot_dir = "examples/scripts/plots/single_qubit/state_transfer"

# experiment name
experiment = "T_$(T)_Q_$(Q)_iter_$(max_iter)"

# creating unique plotting path
plot_path = generate_file_path("png", experiment, plot_dir)

# plotting initial trajectory
plot(plot_path, prob.trajectory, [:ψ̃1, :γ, :α]; timestep_name=:Δt)

# solving the problem
solve!(prob)

@info "" prob.trajectory.Δt

# calculating
fid1 = fidelity(prob.trajectory[end].ψ̃1, prob.trajectory.goal.ψ̃1)
fid2 = fidelity(prob.trajectory[end].ψ̃2, prob.trajectory.goal.ψ̃2)
println("Final |0⟩ fidelity:       ", fid1)
println("Final |1⟩ fidelity:       ", fid2)

drives = vcat(prob.trajectory.γ, prob.trajectory.α)
Δts = vec(prob.trajectory.Δt)

# |0⟩ rollout test
ψ₁ = [1, 0]
ψ̃₁ = ket_to_iso(ψ₁)
ψ̃₁_goal = ket_to_iso(gate * ψ₁)
Ψ̃₁ = rollout(ψ̃₁, drives, Δts, system)
println("|0⟩ Rollout fidelity: ", fidelity(Ψ̃₁[:, end], ψ̃₁_goal))

# |1⟩ rollout test
ψ₂ = [0, 1]
ψ̃₂ = ket_to_iso(ψ₂)
ψ̃₂_goal = ket_to_iso(gate * ψ₂)
Ψ̃₂ = rollout(ψ̃₂, drives, Δts, system)
println("|1⟩ Rollout fidelity: ", fidelity(Ψ̃₂[:, end], ψ̃₂_goal))

# new plot name with fidelity included
plot_path = join(split(plot_path, ".")[1:end-1]) * "_fidelity_$(fid1).png"
plot(plot_path, prob.trajectory, [:ψ̃1, :γ, :α]; timestep_name=:Δt)

# save the trajectory
save_dir = "examples/scripts/trajectories/single_qubit/state_transfer"
save_path = generate_file_path("jld2", experiment, save_dir)
save(save_path, prob.trajectory)
