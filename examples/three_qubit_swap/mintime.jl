using NamedTrajectories
using QuantumCollocation

data_path = "examples/three_qubit_swap/results/T_500_Δt_0.4_a_bound_0.25132741228718347_dda_bound_0.05_dt_min_0.2_dt_max_0.4_max_iter_100000_00000.jld2"

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
ψ = qubit_system_state("100")
ψ̃ = ket_to_iso(ψ)
ψ̃_goal = ket_to_iso(U_goal * ψ)
Ψ̃ = rollout(ψ̃, prob.trajectory.a, prob.trajectory.Δt, prob.system)
Ψ̃_exp = rollout(ψ̃, prob.trajectory.a, prob.trajectory.Δt, prob.system; integrator=exp)
pade_rollout_fidelity = fidelity(Ψ̃[:, end], ψ̃_goal)
exp_rollout_fidelity = fidelity(Ψ̃_exp[:, end], ψ̃_goal)
println("|100⟩ pade rollout fidelity:  ", pade_rollout_fidelity)
println("|100⟩ exp rollout fidelity:   ", exp_rollout_fidelity)
