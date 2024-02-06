using QuantumCollocation
using LaTeXStrings
using CairoMakie
using NamedTrajectories

data_path = joinpath(@__DIR__, "results/mintime/Y_gate_T_100_Q_100.0_R_0.01_R_smoothness_0.001_iter_10000_fidelity_0.9999997188819991_00000.jld2")
traj = load_problem(data_path; return_data=true)["trajectory"]
plot_dir = joinpath(@__DIR__, "plots/paper")
plot_path = joinpath(plot_dir, "fig_single_qubit.svg")

function plot_single_qubit(
    path::String,
    traj::NamedTrajectory,
    comps::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} = traj.names;

    # data keyword arguments
    transformations::Dict{Symbol, <:Union{Function, Vector{Function}}} =
        Dict{Symbol, Union{Function, Vector{Function}}}(),

    # style keyword arguments
    size::Tuple{Int, Int}=(1200, 800),
    titlesize::Int=25,
    series_color::Symbol=:glasbey_bw_minc_20_n256,
    ignored_labels::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} =
        Symbol[],
    dt_name::Union{Symbol,Nothing}=nothing,
    labelsize=27,
    xticksize = 28,
    yticksize = 28,
    xsize = 30,
    ysize = 30,
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

    ts = get_times(traj, dt_name)

    # create figure
    fig = Figure(size=fig_size)

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
                    xlabel=L"t",
                    ylabelsize = ysize,
                    xlabelsize = xsize,
                    xticklabelsize = xticksize,
                    yticklabelsize = yticksize
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
                ylabel = "Population",
                ylabelsize = ysize,
                xlabelsize = xsize,
                xticklabelsize = xticksize,
                yticklabelsize = yticksize
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
            Legend(fig[1,2], ax, labelsize=(labelsize*4) ÷ 5)
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
            ylabel="Amplitude (GHz)",
            ylabelsize = ysize,
            xlabelsize = xsize,
            xticklabelsize = xticksize,
            yticklabelsize = yticksize
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

transformations = Dict(
    :Ũ⃗ => Ũ⃗ -> vec(populations(iso_vec_to_operator(Ũ⃗))),
)

plot_single_qubit(
    plot_path,
    traj,
    [:α, :γ];
    ignored_labels=[:Ũ⃗],
    transformations=transformations,
    titlesize = 25,
    labelsize=30,
    series_color = :Dark2_4,
    dt_name = :Δt
)
