using Pico
using NamedTrajectories
using LinearAlgebra
using Distributions

# defining levels for single qubit system
n_levels = 2

# definining pauli matrices
σx = GATES[:X]
σy = GATES[:Y]
σz = GATES[:Z]

# definining initial value of wavefunction
ψ_init = [1.0, 0.0]

# gate to be applied
gate = σy

# defining goal value of wavefunction
ψ_goal = gate * ψ_init

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
ψ̃_init = ket_to_iso(ψ_init)
ψ̃_goal = ket_to_iso(ψ_goal)

# getting dimension of the isomorphic vector representation
ψ̃_dim = length(ψ̃_init)

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

ψ̃ = rollout(ψ̃_init, u, Δt, system)

# defining components for trajectory
comps = (
    ψ̃ = ψ̃,
    γ = γ,
    dγ = randn(γ_dim, T),
    ddγ = randn(γ_dim, T),
    α = α,
    dα = randn(α_dim, T),
    ddα = randn(α_dim, T),
    Δt = Δt
)

# defining bounds
bounds = (
    γ = fill(γ_bound, γ_dim),
    α = fill(α_bound, α_dim),
    Δt = (dt_min, dt_max)
)

# defining initial values
initial = (
    ψ̃ = ψ̃_init,
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
    ψ̃ = ψ̃_goal,
)

# creating named trajectory
traj = NamedTrajectory(
    comps;
    controls=(:ddγ, :ddα, :Δt),
    dt=dt,
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
    ψ̃ₜ₊₁ = zₜ₊₁[traj.components.ψ̃]
    ψ̃ₜ = zₜ[traj.components.ψ̃]

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
    δψ̃ = P(ψ̃ₜ₊₁, ψ̃ₜ, uₜ, Δtₜ)

    # γ dynamics
    δγ = γₜ₊₁ - γₜ - dγₜ * Δtₜ
    δdγ = dγₜ₊₁ - dγₜ - ddγₜ * Δtₜ

    # α dynamics
    δα = αₜ₊₁ - αₜ - dαₜ * Δtₜ
    δdα = dαₜ₊₁ - dαₜ - ddαₜ * Δtₜ

    return vcat(δψ̃, δγ, δdγ, δα, δdα)
end

# quantum objective weight parameter
Q = 1.0e2

# defining infidelity loss
loss = :InfidelityLoss

# creating quantum objective
J = QuantumObjective(:ψ̃, traj, loss, Q)

# regularization parameters
R_ddγ = 1e-4
R_ddα = 1e-4

# addign quadratic regularization term on γ to the objective
J += QuadraticRegularizer(:ddγ, traj, R_ddγ * ones(γ_dim))

# adding quadratic regularization term on
J += QuadraticRegularizer(:ddα, traj, R_ddα * ones(α_dim))

# setting maximum number of iterations
max_iter = 100

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
plot(plot_path, prob.trajectory, [:ψ̃, :γ, :α]; ignored_labels=[:ψ̃], dt_name=:Δt)

# solving the problem
solve!(prob)

@info "" prob.trajectory.Δt

# calculating unitary fidelity
fid = fidelity(prob.trajectory[end].ψ̃, prob.trajectory.goal.ψ̃)
println("Final fidelity: ", fid)

# rollout test
# ψ₁ = [1, 0]
# ψ̃₁ = ket_to_iso(ψ₁)
# ψ̃_goal = ket_to_iso(σy * ψ₁)
# controls = vcat(prob.trajectory.γ, prob.trajectory.α)
# Ψ̃ = rollout(ψ̃₁, controls, vec(prob.trajectory.Δt), system)
# println("|0⟩ Rollout fidelity:   ", fidelity(Ψ̃[:, end], ψ̃_goal))

# new plot name with fidelity included
plot_path = split(plot_path, ".")[1] * "_fidelity_$(fid).png"
plot(plot_path, prob.trajectory, [:ψ̃, :γ, :α], ignored_labels=[:ψ̃], dt_name=:Δt)
