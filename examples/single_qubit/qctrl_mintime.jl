using QuantumCollocation
using NamedTrajectories

data_path = "examples/single_qubit/results/Y_gate_T_100_Q_100.0_R_0.0001_R_smoothness_0.001_iter_10000_fidelity_0.9999999962678012_00000.jld2"

experiment = join(split(split(data_path, "/")[end], ".")[1:end-1], ".")

plot_path = joinpath(@__DIR__, "plots/mintime", experiment * ".png")

D = 1.0e9

tol = 1e-12

options = Options(tol=tol)

fidelity_bound = 0.999995

prob = UnitaryMinimumTimeProblem(data_path; options=options, D=D, final_fidelity=fidelity_bound)


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
fid_fourth_order_pade = fidelity(Ψ̃₁_fourth_order_pade[:, end], ψ̃₁_goal)
fid_exp = fidelity(Ψ̃₁_exp[:, end], ψ̃₁_goal)
println("|0⟩ Fourth order Pade rollout fidelity:   ", fid_fourth_order_pade)
println("|0⟩ Exponential rollout fidelity:         ", fid_exp)
println()

save_path = joinpath(@__DIR__, "results/mintime", experiment * ".jld2")

# save the problem
info = Dict(
    "exp fidelity" => fid_exp,
    "duration" => times(prob.trajectory)[end],
)

save_problem(save_path, prob, info)
