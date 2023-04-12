module SaveLoadUtils

export save_problem
export load_problem
export generate_file_path

using ..Constraints
using ..Objectives
using ..Integrators
using ..Problems


function save_problem(path::String, prob::QuantumControlProblem)
    mkpath(dirname(path))
    data = Dict(
        "system" => prob.system,
        "trajectory" => prob.trajectory,
        "params" => prob.params,
    )
    save(path, data)
end

function load_problem(path::String)
    data = load(path)
    if data["params"]["dynamics"] == :function
        @warn "Dynamics was built using a user defined function, which could not be saved: returning data dict instead of problem (keys = [\"trajectory\", \"system\", \"params\"]) "
        return data
    else
        integrators = data["params"][:dynamics]
        delete!(data["params"], :dynamics)

        nl_constraints = NonlinearConstraint.(data["params"][:nl_constraints]),
        delete!(data["params"], :nl_constraints)

        objective = Objective(data["params"][:objective_terms])
        delete!(data["params"], :objective_terms)

        return QuantumControlProblem(
            data["system"],
            data["trajectory"],
            objective,
            integrators;
            nl_constraints=nl_constraints,
            data["params"]...
        )
    end
end

function generate_file_path(extension, file_name, path)
    # Ensure the path exists.
    mkpath(path)

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
