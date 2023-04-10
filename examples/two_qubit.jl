using Revise
using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using Distributions
using LaTeXStrings
using CairoMakie
using JLD2

max_iter = 1700
linear_solver = "pardiso"


ω1 =  2π * 3.5 #GHz
ω2 = 2π * 3.9 #GHz
J = 0.1 * 2π
alpha = -2π * 0.225
levels = 2

p4_solve_path = "QuantumCollocation.jl/data/twoqubit/good_2_CNOT_nelson_paper_00027.jld2"

#T_400_iter_4700_00000.jld2
data = JLD2.load(p4_solve_path)["data"]
n_wfn_states = data.system.n_wfn_states
ncontrols = data.system.ncontrols
A = hcat([x[n_wfn_states .+ slice(1, ncontrols)] for x in data.trajectory.states]...)
dA = hcat([x[n_wfn_states .+ slice(2, ncontrols)] for x in data.trajectory.states]...)
ddA = hcat(data.trajectory.actions...)

# load_prob = load_problem(p4_solve_path)
# loaded_traj = load_prob.trajectory
# A = loaded_traj.u
# dA = loaded_traj.du
# ddA = loaded_traj.ddu

H_drift = ω1*kron(number(levels), I(levels)) + ω2*kron(I(levels), number(levels)) 
          + 1/2*alpha * kron(quad(levels), I(levels)) + 1/2*alpha*kron(I(levels), quad(levels)) 
          + J*kron(create(levels) + annihilate(levels), create(levels) + annihilate(levels))

H_drives = [kron(create(levels) + annihilate(levels), I(levels)), 
           kron(I(levels), create(levels) + annihilate(levels)),
           kron(I(levels), number(levels))]

n_controls = length(H_drives)

system = QuantumSystem(
    H_drift,
    H_drives
)

U_init = [
    1. 0. 0. 0.;
    0. 1. 0. 0.;
    0. 0. 1. 0.;
    0. 0. 0. 1.
]

U_goal = [
    1. 0. 0. 0.;
    0. 1. 0. 0.;
    0. 0. 0. 1.;
    0. 0. 1. 0.
]

Ũ⃗_init = operator_to_iso_vec(U_init)
Ũ⃗_goal = operator_to_iso_vec(U_goal)
Ũ⃗_dim = length(Ũ⃗_init)


T = 400
dt = 0.025
u_bound = 2π * 0.5
u_dist = Uniform(-u_bound, u_bound)

Ũ⃗ = unitary_geodesic(U_goal, T; return_generator=false)

comps = (
    Ũ⃗ = Ũ⃗,
    u = A,
    du = dA,
    ddu = ddA,
    Δt = dt * ones(1, T)
)

bounds = (
    u = fill(u_bound, n_controls),
)


initial = (
    Ũ⃗ = Ũ⃗_init,
    u = zeros(n_controls),
)

final = (
    u = zeros(n_controls),
)

goal = (
    Ũ⃗ = Ũ⃗_goal,
)

traj = NamedTrajectory(
    comps;
    controls=(:ddu),
    timestep=dt,
    dynamical_timesteps=false,
    bounds=bounds,
    initial=initial,
    final=final,
    goal=goal
)

P10 = TenthOrderPade(system,:Ũ⃗, :u, :Δt)


function f(zₜ, zₜ₊₁)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components.Ũ⃗]
    Ũ⃗ₜ = zₜ[traj.components.Ũ⃗]
    uₜ₊₁ = zₜ₊₁[traj.components.u]
    uₜ = zₜ[traj.components.u]

    duₜ₊₁ = zₜ₊₁[traj.components.du]
    duₜ = zₜ[traj.components.du]

    dduₜ = zₜ[traj.components.ddu]
    Δtₜ = zₜ[traj.components.Δt][1]

    δŨvec = P10(Ũ⃗ₜ₊₁, Ũ⃗ₜ, uₜ, Δtₜ, operator=true)   
    #δŨvec = P10(Ũ⃗ₜ₊₁, Ũ⃗ₜ, uₜ, Δtₜ)

    δu = uₜ₊₁ - uₜ - duₜ * Δtₜ
    δdu = duₜ₊₁ - duₜ - dduₜ * Δtₜ

    return vcat(δŨvec, δu, δdu)
end 

loss =:UnitaryInfidelityLoss

Q = 200000.
R = 1.
 
J = QuantumObjective(:Ũ⃗, traj, loss, Q)
#J += QuadraticRegularizer(:u, traj, R*ones(n_controls))
J += QuadraticRegularizer(:ddu, traj, R * ones(n_controls))

options = Options(
    max_iter=max_iter,
)

prob = QuantumControlProblem(system, traj, J, f;
    options=options,
)


plot_dir = "QuantumCollocation.jl/examples/plots/two_qubit/CNOT_lab"

# experiment name
experiment = "T_$(T)_iter_$(max_iter)"

plot_path = generate_file_path("png", experiment, plot_dir)
save_path = generate_file_path("jld2", experiment, "QuantumCollocation.jl/data/twoqubit/")

function populations(
    U_col::AbstractVector;
    components=1:length(U_col)
)
    return abs2.(U_col[components])
end

solve!(prob, save_path = save_path)

fid = unitary_fidelity(prob.trajectory[end].Ũ⃗, prob.trajectory.goal.Ũ⃗)
println("Final unitary fidelity: ", fid)

dts = vec(prob.trajectory.Δt)

transformations = Dict(
    :Ũ⃗ => Ũ⃗ -> populations(iso_vec_to_operator(Ũ⃗[:,end])[:, end])
)

function plot_twoqubit(
    path::String,
    traj::NamedTrajectory,
    comps::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} = traj.names;

    # data keyword arguments
    transformations::Dict{Symbol, <:Union{Function, Vector{Function}}} =
        Dict{Symbol, Union{Function, Vector{Function}}}(),

    # style keyword arguments
    res::Tuple{Int, Int}=(1200, 800),
    titlesize::Int=25,
    series_color::Symbol=:glasbey_bw_minc_20_n256,
    ignored_labels::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} =
        Symbol[],
    dt_name::Union{Symbol,Nothing}=nothing,
    labelsize=15,
)
    # convert single symbol to vector: comps
    if comps isa Symbol
        comps = [comps]
    end

    # convert single symbol to iterable: ignored labels
    if ignored_labels isa Symbol
        ignored_labels = [ignored_labels]
    end

    @assert all([key ∈ keys(traj.components) for key ∈ comps])
    @assert all([key ∈ keys(traj.components) for key ∈ keys(transformations)])

    ts = times(traj, dt_name)

    # create figure
    fig = Figure(resolution=res)

    # initialize axis count
    ax_count = 0

    # plot transformed components
    for (key, f) in transformations
        if f isa Vector
            for (j, fⱼ) in enumerate(f)

                # data matrix for key componenent of trajectory
                data = traj[key]

                # apply transformation fⱼ to each column of data
                transformed_data = mapslices(fⱼ, data; dims=1)

                # create axis for transformed data
                ax = Axis(
                    fig[ax_count + 1, 1];
                    title=latexstring(key, "(t)", "\\text{ transformation } $j"),
                    titlesize=titlesize,
                    xlabel=L"t"
                )

                # plot transformed data
                series!(
                    ax,
                    ts,
                    transformed_data;
                    color=series_color,
                    #markersize=5,
                    labels=[latexstring(key, "_{$i}") for i = 1:size(transformed_data, 2)]
                )

                # create legend
                Legend(fig[ax_count + 1, 2], ax)

                # increment axis count
                ax_count += 1
            end
        else

            # data matrix for key componenent of trajectory
            data = traj[key]

            # apply transformation f to each column of data
            transformed_data = mapslices(f, data; dims=1)

            # create axis for transformed data
            ax = Axis(
                fig[ax_count + 1, :];
                title="Evolution of |11⟩",
                titlesize=titlesize,
                xlabel = "Time (ns)", #latexstring("t \\text{ (ns)}"),
                ylabel = "Population"
            )

            # plot transformed data
            series!(
                ax,
                ts,
                transformed_data;
                color=series_color,
                #markersize=5,
                labels=[
                latexstring("|00 \\rangle"),
                latexstring("|01 \\rangle"),
                latexstring("|10 \\rangle"),
                latexstring("|11 \\rangle"),
                ]
            )
            Legend(fig[1,2], ax, labelsize=20)
            # increment axis count
            ax_count += 1
        end
    end

    # plot normal components
    for key in comps

        # data matrix for key componenent of trajectory
        data = traj[key]/(2π)

        # create axis for data
        ax = Axis(
            fig[ax_count + 1, 1];
            title="Controls",
            titlesize=titlesize,
            xlabel="Time (ns)",#latexstring("t \\text{ (ns)}"),
            ylabel="Amplitude (GHz)"
        )

        # create labels if key is not in ignored_labels
        if key ∈ ignored_labels
            labels = nothing
        else
            labels = [latexstring(key, "_{$i}") for i = 1:size(data, 1)]
        end

        # plot data
        series!(
            ax,
            ts,
            data;
            color=series_color,
            #markersize=5,
            labels=labels
        )

        # create legend
        if key ∉ ignored_labels
            Legend(fig[ax_count + 1, 2], ax, labelsize=labelsize)
        end

        # increment axis count
        ax_count += 1
    end

    save(path, fig)
end

plot_twoqubit(
    plot_path, 
    prob.trajectory, 
    [:u];
    ignored_labels=[:Ũ⃗], 
    transformations=transformations,
    titlesize = 20,
    labelsize= 25,
    series_color = :Dark2_4
)

Ũ⃗₁ = operator_to_iso_vec(I(4))
U_exp = iso_vec_to_operator(unitary_rollout(Ũ⃗₁, prob.trajectory.u, dts, system, integrator=exp)[:, T])
U_10 = iso_vec_to_operator(unitary_rollout(Ũ⃗₁, prob.trajectory.u, dts, system, integrator=tenth_order_pade)[:, T])
U_6 = iso_vec_to_operator(unitary_rollout(Ũ⃗₁, prob.trajectory.u, dts, system, integrator=sixth_order_pade)[:, T])

fid_exp = 1/4*abs(tr(U_exp'U_goal))
fid_10 = 1/4*abs(tr(U_10'U_goal))
fid_6 = 1/4*abs(tr(U_6'U_goal))

println("6th Order Pade Fidelity: $fid_6")
println("10th Order Pade Fidelity = $fid_10")
println("Exp Fidelity = $fid_exp")




