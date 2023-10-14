module SaveLoadUtils

export save_problem
export load_problem
export generate_file_path

using ..Constraints
using ..Objectives
using ..Integrators
using ..Problems

using JLD2


function save_problem(
    path::String,
    prob::QuantumControlProblem,
    info::Dict{String,<:Any}=Dict{String, Any}()
)
    mkpath(dirname(path))

    data = Dict(
        "system" => prob.system,
        "trajectory" => prob.trajectory,
        "integrators" => prob.integrators,
        "options" => prob.options,
        "params" => prob.params,
    )

    # assert none of the keys in info are already in data
    for key in keys(info)
        if haskey(data, key)
            @warn "Key $(key) in info exists in data dict, removing"
            delete!(info, key)
        end
    end

    merge!(data, info)

    save(path, data)
end

const RESERVED_KEYS = ["system", "trajectory", "options", "params", "integrators"]

function load_problem(path::String; verbose=true, return_data=false, kwargs...)
    data = load(path)

    if verbose
        println("Loading $(return_data ? "data dict" : "problem") from $(path):\n")
        for (key, value) ∈ data
            if key ∉ RESERVED_KEYS
                println("   $(key) = $(value)")
            end
        end
    end

    if return_data
        return data
    else
        if isnothing(data["integrators"])
            @warn "Dynamics was built using a user defined function, which could not be saved: returning data dict instead of problem (keys = [\"trajectory\", \"system\", \"params\"]) "
            return data
        end

        system = data["system"]
        delete!(data, "system")

        trajectory = data["trajectory"]
        delete!(data, "trajectory")

        options = data["options"]
        delete!(data, "options")

        params = data["params"]
        delete!(data, "params")

        integrators = data["integrators"]
        delete!(data, "integrators")

        objective = Objective(params[:objective_terms])
        delete!(params, :objective_terms)

        linear_constraints = params[:linear_constraints]
        delete!(params, :linear_constraints)

        nonlinear_constraints = NonlinearConstraint.(params[:nonlinear_constraints])
        delete!(params, :nonlinear_constraints)

        constraints = AbstractConstraint[linear_constraints; nonlinear_constraints]

        prob = QuantumControlProblem(
            system,
            trajectory,
            objective,
            integrators;
            constraints=constraints,
            options=options,
            verbose=verbose,
            build_trajectory_constraints=false,
            params...,
            kwargs...
        )

        return prob
    end
end

function save_h5(prob::QuantumControlProblem, save_path::String; verbose=true)
    traj = prob.trajectory

    result = Dict(

    )
end

function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

    # remove dot from extension
    extension = split(extension, ".")[end]

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_suffix = -1
    for (_, _, files) in walkdir(path)
        for file_name_ in files
            if occursin("$(file_name)", file_name_) && occursin(".$(extension)", file_name_)

                numeric_suffix = parse(
                    Int,
                    split(split(file_name_, "_")[end], ".")[1]
                )

                max_numeric_suffix = max(
                    numeric_suffix,
                    max_numeric_suffix
                )
            end
        end
    end

    file_path = joinpath(
        path,
        file_name *
        "_$(lpad(max_numeric_suffix + 1, 5, '0')).$(extension)"
    )

    return file_path
end



end
