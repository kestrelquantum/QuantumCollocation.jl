using Piccolo

data_path = "newresults/T_500_Q_200.0_Δt_0.4_a_bound_0.25132741228718347_dda_bound_0.05dt_min_0.2_dt_max_0.4_max_iter_200_00000.jld2"

experiment = join(split(split(data_path, "/")[end], ".")[1:end-1], ".")

plot_path = joinpath(@__DIR__, "newplots_mintime", experiment * ".png")

D = 1.0e3

final_fidelity = 0.999

prob = UnitaryMinimumTimeProblem(data_path; D=D, final_fidelity=final_fidelity)



plot(plot_path, prob.trajectory, [:Ũ⃗, :γ, :α]; ignored_labels=[:Ũ⃗])

solve!(prob)

plot(plot_path, prob.trajectory, [:Ũ⃗, :γ, :α]; ignored_labels=[:Ũ⃗])


# calculating unitary fidelity
fid = unitary_fidelity(prob)
println("Final unitary fidelity: ", fid)
println("Duration of trajectory: ", times(prob.trajectory)[end])
println()

info = Dict(
    "fidelity" => fid,
    "duration" => times(prob.trajectory)[end],
)
