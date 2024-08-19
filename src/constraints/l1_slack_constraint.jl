export L1SlackConstraint


struct L1SlackConstraint <: LinearConstraint
    var_name::Symbol
    slack_names::Vector{Symbol}
    indices::AbstractVector{Int}
    times::AbstractVector{Int}
    label::String

    function L1SlackConstraint(
        name::Symbol,
        traj::NamedTrajectory;
        indices=1:traj.dims[name],
        times=(name ∈ keys(traj.initial) ? 2 : 1):traj.T,
        label="L1 slack constraint on $name"
    )
        @assert all(i ∈ 1:traj.dims[name] for i ∈ indices)
        s1_name = Symbol("s1_$name")
        s2_name = Symbol("s2_$name")
        slack_names = [s1_name, s2_name]
        add_component!(traj, s1_name, rand(length(indices), traj.T))
        add_component!(traj, s2_name, rand(length(indices), traj.T))
        return new(name, slack_names, indices, times, label)
    end
end

function (con::L1SlackConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for t ∈ con.times
        for (s1, s2, x) in zip(
            slice(t, traj.components[con.slack_names[1]], traj.dim),
            slice(t, traj.components[con.slack_names[2]], traj.dim),
            slice(t, traj.components[con.var_name][con.indices], traj.dim)
        )
            MOI.add_constraints(
                opt,
                vars[s1],
                MOI.GreaterThan(0.0)
            )
            MOI.add_constraints(
                opt,
                vars[s2],
                MOI.GreaterThan(0.0)
            )
            t1 = MOI.ScalarAffineTerm(1.0, vars[s1])
            t2 = MOI.ScalarAffineTerm(-1.0, vars[s2])
            t3 = MOI.ScalarAffineTerm(-1.0, vars[x])
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction([t1, t2, t3], 0.0),
                MOI.EqualTo(0.0)
            )
        end
    end
end