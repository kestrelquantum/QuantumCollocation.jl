module Plotting

export unitary_populations_plot

using NamedTrajectories

using ..QuantumUtils
using ..Problems

function unitary_populations_plot(
    traj::NamedTrajectory;
    unitary_columns::AbstractVector{Int}=1:2,
    unitary_name::Symbol=:Ũ⃗,
    control_name::Symbol=:a,
    kwargs...
)

    transformations = OrderedDict(
        unitary_name => [
            x -> populations(iso_vec_to_operator(x)[:, i])
                for i ∈ unitary_columns
        ]
    )

    transformation_titles = OrderedDict(
        unitary_name => [
            L"Populations: $\left| U_{:, %$(i)}(t) \right|^2$"
                for i ∈ unitary_columns
        ]
    )

    plot(traj, [control_name];
        transformations=transformations,
        transformation_titles=transformation_titles,
        include_transformation_labels=true,
        kwargs...
    )
end

function unitary_populations_plot(prob::QuantumControlProblem; kwargs...)
    unitary_populations_plot(prob.trajectory; kwargs...)
end

end
