export trajectory_constraints

export EqualityConstraint
export BoundsConstraint
export TimeStepBoundsConstraint
export TimeStepEqualityConstraint
export TimeStepsAllEqualConstraint

"""
    trajectory_constraints(traj::NamedTrajectory)

Implements the initial and final value constraints and bounds constraints on the controls
and states as specified by traj.

"""
function trajectory_constraints(traj::NamedTrajectory)
    cons = AbstractConstraint[]

    init_names = []

    # add initial equality constraints
    for (name, val) ∈ pairs(traj.initial)
        ts = [1]
        js = traj.components[name]
        con_label = "initial value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; label=con_label)
        push!(cons, eq_con)
        push!(init_names, name)
    end

    final_names = []

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        ts = [traj.T]
        js = traj.components[name]
        con_label = "final value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; label=con_label)
        push!(cons, eq_con)
        push!(final_names, name)
    end

    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ init_names && name ∈ final_names
            ts = 2:traj.T-1
        elseif name ∈ init_names && !(name ∈ final_names)
            ts = 2:traj.T
        elseif name ∈ final_names && !(name ∈ init_names)
            ts = 1:traj.T-1
        else
            ts = 1:traj.T
        end
        js = traj.components[name]
        con_label = "bounds on $name"
        bounds = collect(zip(bound[1], bound[2]))
        bounds_con = BoundsConstraint(ts, js, bounds, traj.dim; label=con_label)
        push!(cons, bounds_con)
    end

    return cons
end

### 
### EqualityConstraint
###

"""
    struct EqualityConstraint

Represents a linear equality constraint.

# Fields
- `ts::AbstractArray{Int}`: the time steps at which the constraint is applied
- `js::AbstractArray{Int}`: the components of the trajectory at which the constraint is applied
- `vals::Vector{R}`: the values of the constraint
- `vardim::Int`: the dimension of a single time step of the trajectory
- `label::String`: a label for the constraint

"""
struct EqualityConstraint <: LinearConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{R} where R
    vardim::Int
    label::String
end

function EqualityConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    label="unlabeled equality constraint"
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
        label
    )
end


function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
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

### 
### BoundsConstraint
###

struct BoundsConstraint <: LinearConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{Tuple{R, R}} where R <: Real
    vardim::Int
    label::String
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{Tuple{R, R}, Vector{Tuple{R, R}}},
    vardim::Int;
    label="unlabeled bounds constraint"
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
        label
    )
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    label="unlabeled bounds constraint"
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
        label
    )
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
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

### 
### TimeStepBoundsConstraint
###

struct TimeStepBoundsConstraint <: LinearConstraint
    bounds::Tuple{R, R} where R <: Real
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepBoundsConstraint(
        bounds::Tuple{R, R} where R <: Real,
        Δt_indices::AbstractVector{Int},
        T::Int;
        label="time step bounds constraint"
    )
        @assert bounds[1] < bounds[2] "lower bound must be less than upper bound"
        return new(bounds, Δt_indices, label)
    end
end

function (con::TimeStepBoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
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

### 
### TimeStepEqualityConstraint
###

struct TimeStepEqualityConstraint <: LinearConstraint
    val::R where R <: Real
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepEqualityConstraint(
        val::R where R <: Real,
        Δt_indices::AbstractVector{Int};
        label="unlabeled time step equality constraint"
    )
        return new(val, Δt_indices, label)
    end
end

function (con::TimeStepEqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for i ∈ con.Δt_indices
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.EqualTo(con.val)
        )
    end
end

struct TimeStepsAllEqualConstraint <: LinearConstraint
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepsAllEqualConstraint(
        Δt_indices::AbstractVector{Int};
        label="time step all equal constraint"
    )
        return new(Δt_indices, label)
    end

    function TimeStepsAllEqualConstraint(
        Δt_symb::Symbol,
        traj::NamedTrajectory;
        label="time step all equal constraint"
    )
        Δt_comp = traj.components[Δt_symb][1]
        Δt_indices = [index(t, Δt_comp, traj.dim) for t = 1:traj.T]
        return new(Δt_indices, label)
    end
end

function (con::TimeStepsAllEqualConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
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