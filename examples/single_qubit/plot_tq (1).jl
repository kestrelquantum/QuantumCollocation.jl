using QuantumCollocation
using LaTeXStrings
using CairoMakie
using Colors
using NamedTrajectories

dir = "QuantumCollocation.jl/examples/T_200_iter_2000_Q_1.0e7_00002.jld2"
prob = load_problem(dir)
plot_dir = "QuantumCollocation.jl/examples/plots/paper_tq"
plot_path = generate_file_path("svg", "paper_fig", plot_dir)

function plot_twoqubit(
    path::String,
    traj::NamedTrajectory,
    comps::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} = traj.names;

    # data keyword arguments
    transformations::Dict{Symbol, <:Union{Function, Vector{Function}}} =
        Dict{Symbol, Union{Function, Vector{Function}}}(),

    # style keyword arguments
    size::Tuple{Int, Int}=(1200, 900),
    titlesize::Int=36,
    series_color=:Dark2_4,
    series_color_cntrls = series_color,
    ignored_labels::Union{Symbol, Vector{Symbol}, Tuple{Vararg{Symbol}}} =
        Symbol[],
    dt_name::Union{Symbol,Nothing}=nothing,
    labelsize=55,
    xticksize = 35,
    yticksize = 35,
    xsize = 36,
    ysize = 36,
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

    ts = [0.; get_times(traj, dt_name)[1:end-1]]

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
            #@Aaron you will need to fix the title
            ax = Axis(
                fig[ax_count + 1, :];
                title= "Evolution of |11⟩", #latexstring("\\text{Evolution of } |11 \\rangle"),
                titlesize=titlesize,
                xlabel = "Time [ns]", #latexstring(" \\text{Time [ns]}"), #"Time (ns)", #latexstring("t \\text{ (ns)}"),
                ylabel = "Population", #latexstring("\\text{Population}"), #"Population",
                ylabelsize = ysize,
                xlabelsize = xsize,
                xticklabelsize = xticksize,
                yticklabelsize = yticksize
            )

            # plot transformed data

            #@Aaron you will need to fix these labels
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
            Legend(fig[1,2], ax, labelsize=(labelsize*4) ÷ 5, framevisible = false, linepoints=[Point2f(-1, 0.5), Point2f(1, 0.5)])
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
            title="Controls", #latexstring("\\text{Controls}"),
            titlesize=titlesize,
            xlabel="Time [ns]", #latexstring(" \\text{Time [ns]}"),#latexstring("t \\text{ (ns)}"),
            ylabel="Amplitude [GHz]", #latexstring("\\text{Amplitude [GHz]}"),
            ylabelsize = ysize,
            xlabelsize = xsize,
            xticklabelsize = xticksize,
            yticklabelsize = yticksize
        )

        # create labels if key is not in ignored_labels
        if key ∈ ignored_labels
            labels = nothing
        else
            #@Aaron you will need to fix these labels to say gamma, alpha, etc
            labels = [latexstring(key, "_{$i}") for i = 1:size(data, 1)]
        end

        # plot data
        series!(
            ax,
            ts,
            data;
            color=series_color_cntrls,
            markersize=8,
            labels=labels
        )

        # create legend
        if key ∉ ignored_labels
            Legend(fig[ax_count + 1, 2], ax,
                   labelsize=labelsize, framevisible = false,
                   linepoints=[Point2f(-1, 0.5), Point2f(1, 0.5)],
                   markerpoints = [Point2f(0., 0.5)],
                   markerstrokewidth = 20)
        end

        # increment axis count
        ax_count += 1
    end

    save(path, fig)
end

function populations(
    U_col::AbstractVector;
    components=1:length(U_col)
)
    return abs2.(U_col[components])
end

transformations = Dict(
    :Ũ⃗ => Ũ⃗ -> populations(iso_vec_to_operator(Ũ⃗[:,end])[:, end])
)

#@Aaron you will need to edit transformations and ignored labels depending on what you want to do
plot_twoqubit(
    plot_path,
    prob,
    [:u];
    ignored_labels=[:Ũ⃗],
    transformations=transformations,
    #@Aaron you should probably change the color scheme of your controls to match
    series_color =  [

    RGB(0.459,0.439,0.702),

    RGB(0.851,0.373,0.008),

    RGB(0.106,0.62,0.467),
    RGB(0.906,0.161,0.541),

    ],
    series_color_cntrls = [
        RGB(0.792,0.0,0.125),
        RGB(0.957,0.647,0.51),
        RGB(0.02,0.443,0.69),
        RGB(0.573,0.773,0.871),
        :cornflowerblue, ] #:Paired_6
    ,
    dt_name = :Δt
)
