module Constraints

export constrain!
export constraints

export AbstractConstraint

export EqualityConstraint
export BoundsConstraint
export TimeStepBoundsConstraint
export TimeStepEqualityConstraint
export TimeStepsAllEqualConstraint
export L1SlackConstraint

using ..IndexingUtils

using NamedTrajectories
using Ipopt
using MathOptInterface
const MOI = MathOptInterface


abstract type AbstractConstraint end

function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{AbstractConstraint};
    verbose=false
)
    for con in cons
        if verbose
            println("applying constraint: ", con.name)
        end
        con(opt, vars)
    end
end

function constraints(traj::NamedTrajectory)
    cons = AbstractConstraint[]

    # add bounds constraints
    for (name, bound) ∈ traj.bounds
        ts = [2:traj.T-1]
        js = traj.components[name]
        con_name = "bounds on $name"
        bounds_con = BoundsConstraint(ts, js, bound, traj.dim; name=con_name)
        push!(cons, bounds_con)
    end

    # add initial equality constraints
    for (name, val) ∈ traj.initial
        ts = [1]
        js = traj.components[name]
        con_name = "initial value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; name=con_name)
        push!(cons, eq_con)
    end

    # add final equality constraints
    for (name, val) ∈ traj.final
        ts = [traj.T]
        js = traj.components[name]
        con_name = "final value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; name=con_name)
        push!(cons, eq_con)
    end

    return cons
end




struct EqualityConstraint <: AbstractConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{R} where R
    vardim::Int
    name::String
end

function EqualityConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    name="unnamed equality constraint"
) where R

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, j must be an array of integers"

    @assert isa(val, R) ||
        (isa(val, Vector{R}) && isa(j, AbstractArray{Int})) &&
        length(val) == length(j) """
    if j and val are both arrays, dimensions must match:
        length(j)   = $(length(j))
        length(val) = $(length(val))
    """

    if isa(val, R) && isa(j, AbstractArray{Int})
        val = fill(val, length(j))
    end

    return EqualityConstraint(
        [t...],
        [j...],
        [val...],
        vardim,
        name
    )
end


function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for t in con.ts
        for (j, val) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.EqualTo(val)
            )
        end
    end
end

struct BoundsConstraint <: AbstractConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{Tuple{R, R}} where R <: Real
    vardim::Int
    name::String
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{Tuple{R, R}, Vector{Tuple{R, R}}},
    vardim::Int;
    name="unnamed bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{Tuple{R, R}}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, Tuple{R,R}) && isa(j, AbstractArray{Int})

        val = fill(val, length(j))

    elseif isa(val, Tuple{R, R}) && isa(j, Int)

        val = [val]
        j = [j]

    end

    @assert *([v[1] <= v[2] for v in val]...) "lower bound must be less than upper bound"

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        name
    )
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    name="unnamed bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, R) && isa(j, AbstractArray{Int})

        bounds = (-abs(val), abs(val))
        val = fill(bounds, length(j))

    elseif isa(val, R) && isa(j, Int)

        bounds = (-abs(val), abs(val))
        val = [bounds]
        j = [j]

    elseif isa(val, Vector{R})

        val = [(-abs(v), abs(v)) for v in val]

    end

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        name
    )
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for t in con.ts
        for (j, (lb, ub)) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.GreaterThan(lb)
            )
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.LessThan(ub)
            )
        end
    end
end

struct TimeStepBoundsConstraint <: AbstractConstraint
    bounds::Tuple{R, R} where R <: Real
    Δt_indices::AbstractVector{Int}
    name::String

    function TimeStepBoundsConstraint(
        bounds::Tuple{R, R} where R <: Real,
        Δt_indices::AbstractVector{Int},
        T::Int;
        name="unnamed time step bounds constraint"
    )
        @assert bounds[1] < bounds[2] "lower bound must be less than upper bound"
        return new(bounds, Δt_indices, name)
    end
end

function (con::TimeStepBoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for i ∈ con.Δt_indices
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.GreaterThan(con.bounds[1])
        )
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.LessThan(con.bounds[2])
        )
    end
end

struct TimeStepEqualityConstraint <: AbstractConstraint
    val::R where R <: Real
    Δt_indices::AbstractVector{Int}
    name::String

    function TimeStepEqualityConstraint(
        val::R where R <: Real,
        Δt_indices::AbstractVector{Int};
        name="unnamed time step equality constraint"
    )
        return new(val, Δt_indices, name)
    end
end

function (con::TimeStepEqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for i ∈ con.Δt_indices
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.EqualTo(con.val)
        )
    end
end

struct TimeStepsAllEqualConstraint <: AbstractConstraint
    Δt_indices::AbstractVector{Int}
    name::String

    function TimeStepsAllEqualConstraint(
        Δt_indices::AbstractVector{Int};
        name="unnamed time step all equal constraint"
    )
        return new(Δt_indices, name)
    end
end

function (con::TimeStepsAllEqualConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    N = length(con.Δt_indices)
    for i = 1:N-1
        Δtᵢ = MOI.ScalarAffineTerm(1.0, vars[con.Δt_indices[i]])
        minusΔt̄ = MOI.ScalarAffineTerm(-1.0, vars[con.Δt_indices[end]])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([Δtᵢ, minusΔt̄], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end

struct L1SlackConstraint <: AbstractConstraint
    s1_indices::AbstractArray{Int}
    s2_indices::AbstractArray{Int}
    x_indices::AbstractArray{Int}
    name::String

    function L1SlackConstraint(
        s1_indices::AbstractArray{Int},
        s2_indices::AbstractArray{Int},
        x_indices::AbstractArray{Int};
        name="unmamed L1 slack constraint"
    )
        @assert length(s1_indices) == length(s2_indices) == length(x_indices)
        return new(s1_indices, s2_indices, x_indices, name)
    end
end

function (con::L1SlackConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for (s1, s2, x) in zip(
        con.s1_indices,
        con.s2_indices,
        con.x_indices
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
