using QuantumCollocation
using NamedTrajectories

data_path = "examples/single_qubit/results/T_100_Q_100.0_iter_10000_fidelity_0.999999690306333_00000.jld2"

experiment = join(split(split(data_path, "/")[end], ".")[1:end-1], ".")

plot_path = joinpath(@__DIR__, "plots/mintime", experiment * ".png")

D = 1.0e3

prob = UnitaryMinimumTimeProblem(data_path; D=D)


plot(plot_path, prob.trajectory, [:Ũ⃗, :γ, :α]; ignored_labels=[:Ũ⃗])

solve!(prob)

plot(plot_path, prob.trajectory, [:Ũ⃗, :γ, :α]; ignored_labels=[:Ũ⃗])


# calculating unitary fidelity
fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)
println("Final unitary fidelity: ", fid)
println("Duration of trajectory: ", times(prob.trajectory)[end])
println()

drives = vcat(prob.trajectory.γ, prob.trajectory.α)
Δts = vec(prob.trajectory.Δt)


# |0⟩ rollout test
ψ₁ = [1, 0]
ψ̃₁ = ket_to_iso(ψ₁)
ψ̃₁_goal = ket_to_iso(iso_vec_to_operator(prob.trajectory.goal.Ũ⃗) * ψ₁)
Ψ̃₁_fourth_order_pade = rollout(ψ̃₁, drives, Δts, prob.system)
Ψ̃₁_exp = rollout(ψ̃₁, drives, Δts, prob.system; integrator=exp)
println("|0⟩ Fourth order Pade rollout fidelity:   ", fidelity(Ψ̃₁_fourth_order_pade[:, end], ψ̃₁_goal))
println("|0⟩ Exponential rollout fidelity:         ", fidelity(Ψ̃₁_exp[:, end], ψ̃₁_goal))
println()
