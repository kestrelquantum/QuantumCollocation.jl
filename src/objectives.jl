module Objectives

export Objective

export QuantumObjective
export QuantumStateObjective
export QuantumUnitaryObjective
export UnitaryInfidelityObjective

export MinimumTimeObjective

export QuadraticRegularizer
export QuadraticSmoothnessRegularizer
export L1Regularizer

using TrajectoryIndexingUtils
using ..QuantumSystems
using ..Losses
using ..Constraints

using NamedTrajectories
using LinearAlgebra
using SparseArrays
using Symbolics

#
# objective functions
#

"""
    Objective

A structure for defining objective functions.

Fields:
    `L`: the objective function
    `∇L`: the gradient of the objective function
    `∂²L`: the Hessian of the objective function
    `∂²L_structure`: the structure of the Hessian of the objective function
    `terms`: a vector of dictionaries containing the terms of the objective function
"""
struct Objective
	L::Function
	∇L::Function
	∂²L::Union{Function, Nothing}
	∂²L_structure::Union{Function, Nothing}
    terms::Vector{Dict}
end

function Base.:+(obj1::Objective, obj2::Objective)
	L = (Z⃗, Z) -> obj1.L(Z⃗, Z) + obj2.L(Z⃗, Z)
	∇L = (Z⃗, Z) -> obj1.∇L(Z⃗, Z) + obj2.∇L(Z⃗, Z)
	if isnothing(obj1.∂²L) && isnothing(obj2.∂²L)
		∂²L = Nothing
		∂²L_structure = Nothing
	elseif isnothing(obj1.∂²L)
		∂²L = (Z⃗, Z) -> obj2.∂²L(Z⃗, Z)
		∂²L_structure = obj2.∂²L_structure
	elseif isnothing(obj2.∂²L)
		∂²L = (Z⃗, Z) -> obj1.∂²L(Z⃗, Z)
		∂²L_structure = obj1.∂²L_structure
	else
		∂²L = (Z⃗, Z) -> vcat(obj1.∂²L(Z⃗, Z), obj2.∂²L(Z⃗, Z))
		∂²L_structure = Z -> vcat(obj1.∂²L_structure(Z), obj2.∂²L_structure(Z))
	end
    terms = vcat(obj1.terms, obj2.terms)
	return Objective(L, ∇L, ∂²L, ∂²L_structure, terms)
end

Base.:+(obj::Objective, ::Nothing) = obj

function Objective(terms::Vector{Dict})
    return +(Objective.(terms)...)
end

function Objective(term::Dict)
    return eval(term[:type])(; delete!(term, :type)...)
end

# function to convert sparse matrix to tuple of vector of nonzero indices and vector of nonzero values
function sparse_to_moi(A::SparseMatrixCSC)
    inds = collect(zip(findnz(A)...))
    vals = [A[i,j] for (i,j) ∈ inds]
    return (inds, vals)
end
"""
    QuantumObjective


"""
function QuantumObjective(;
    names::Union{Nothing,Tuple{Vararg{Symbol}}}=nothing,
    name::Union{Nothing,Symbol}=nothing,
    goals::Union{Nothing,AbstractVector{<:Real},Tuple{Vararg{AbstractVector{<:Real}}}}=nothing,
	loss::Symbol=:InfidelityLoss,
	Q::Union{Float64, Vector{Float64}}=100.0,
	eval_hessian::Bool=true
)
    @assert !(isnothing(names) && isnothing(name)) "name or names must be specified"
    @assert !isnothing(goals) "goals corresponding to names must be specified"

    if isnothing(names)
        names = (name,)
    end

    if goals isa AbstractVector
        goals = (goals,)
    end

    if Q isa Float64
        Q = ones(length(names)) * Q
    else
        @assert length(Q) == length(names)
    end

    params = Dict(
        :type => :QuantumObjective,
        :names => names,
        :goals => goals,
        :loss => loss,
        :Q => Q,
        :eval_hessian => eval_hessian,
    )

    losses = [eval(loss)(name, goal) for (name, goal) ∈ zip(names, goals)]

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        loss = 0.0
        for (Qᵢ, lᵢ, name) ∈ zip(Q, losses, names)
            name_slice = slice(Z.T, Z.components[name], Z.dim)
            loss += Qᵢ * lᵢ(Z⃗[name_slice])
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
        for (Qᵢ, lᵢ, name) ∈ zip(Q, losses, names)
            name_slice = slice(Z.T, Z.components[name], Z.dim)
            ∇[name_slice] = Qᵢ * lᵢ(Z⃗[name_slice]; gradient=true)
        end
        return ∇
    end

    function ∂²L_structure(Z::NamedTrajectory)
        structure = []
        final_time_offset = index(Z.T, 0, Z.dim)
        for (name, loss) ∈ zip(names, losses)
            comp_start_offset = Z.components[name][1] - 1
            comp_hessian_structure = [
                ij .+ (final_time_offset + comp_start_offset)
                    for ij ∈ loss.∇²l_structure
            ]
            append!(structure, comp_hessian_structure)
        end
        return structure
    end


    @views function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory; return_moi_vals=true)
        H = spzeros(Z.dim * Z.T, Z.dim * Z.T)
        for (Qᵢ, name, lᵢ) ∈ zip(Q, names, losses)
            name_slice = slice(Z.T, Z.components[name], Z.dim)
            H[name_slice, name_slice] =
                Qᵢ * lᵢ(Z⃗[name_slice]; hessian=true)
        end
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end



"""
    UnitaryInfidelityObjective


"""
function UnitaryInfidelityObjective(;
    name::Union{Nothing,Symbol}=nothing,
    goal::Union{Nothing,AbstractVector{<:Real}}=nothing,
	Q::Float64=100.0,
	eval_hessian::Bool=true,
    subspace=nothing
)
    @assert !isnothing(goal) "unitary goal name must be specified"

    loss = :UnitaryInfidelityLoss
    l = eval(loss)(name, goal; subspace=subspace)

    params = Dict(
        :type => :UnitaryInfidelityObjective,
        :name => name,
        :goal => goal,
        :Q => Q,
        :eval_hessian => eval_hessian,
        :subspace => subspace
    )

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        return Q * l(Z⃗[slice(Z.T, Z.components[name], Z.dim)])
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
        Ũ⃗_slice = slice(Z.T, Z.components[name], Z.dim)
        Ũ⃗ = Z⃗[Ũ⃗_slice]
        ∇l = l(Ũ⃗; gradient=true)
        ∇[Ũ⃗_slice] .= Q * ∇l
        return ∇
    end

    function ∂²L_structure(Z::NamedTrajectory)
        final_time_offset = index(Z.T, 0, Z.dim)
        comp_start_offset = Z.components[name][1] - 1
        structure = [
            ij .+ (final_time_offset + comp_start_offset)
                for ij ∈ l.∇²l_structure
        ]
        return structure
    end


    @views function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory; return_moi_vals=true)
        H = spzeros(Z.dim * Z.T, Z.dim * Z.T)
        Ũ⃗_slice = slice(Z.T, Z.components[name], Z.dim)
        H[Ũ⃗_slice, Ũ⃗_slice] = Q * l(Z⃗[Ũ⃗_slice]; hessian=true)
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end


    # ∂²L_structure(Z::NamedTrajectory) = []

    # ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory) = []

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function QuantumObjective(
    name::Symbol,
    traj::NamedTrajectory,
    loss::Symbol,
    Q::Float64
)
    goal = traj.goal[name]
    return QuantumObjective(name=name, goals=goal, loss=loss, Q=Q)
end

function UnitaryInfidelityObjective(
    name::Symbol,
    traj::NamedTrajectory,
    Q::Float64;
    subspace=nothing
)
    return UnitaryInfidelityObjective(name=name, goal=traj.goal[name], Q=Q, subspace=subspace)
end

function QuantumObjective(
    names::Tuple{Vararg{Symbol}},
    traj::NamedTrajectory,
    loss::Symbol,
    Q::Float64
)
    goals = Tuple(traj.goal[name] for name in names)
    return QuantumObjective(names=names, goals=goals, loss=loss, Q=Q)
end

function QuantumUnitaryObjective(
    name::Symbol,
    traj::NamedTrajectory,
    Q::Float64
)
    return QuantumObjective(name, traj, :UnitaryInfidelityLoss, Q)
end

function QuantumStateObjective(
    name::Symbol,
    traj::NamedTrajectory,
    Q::Float64
)
    return QuantumObjective(name, traj, :InfidelityLoss, Q)
end

function QuadraticRegularizer(;
	name::Symbol=nothing,
	times::AbstractVector{Int}=1:traj.T,
    dim::Int=nothing,
	R::AbstractVector{<:Real}=ones(traj.dims[name]),
	eval_hessian=true
)

    @assert !isnothing(name) "name must be specified"
    @assert !isnothing(times) "times must be specified"
    @assert !isnothing(dim) "dim must be specified"

    params = Dict(
        :type => :QuadraticRegularizer,
        :name => name,
        :times => times,
        :dim => dim,
        :R => R,
        :eval_hessian => eval_hessian
    )

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
		J = 0.0
		for t ∈ times
            vₜ = Z⃗[slice(t, Z.components[name], Z.dim)]
            J += 0.5 * vₜ' *  (R .* vₜ)
		end
		return J
	end

	@views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
		∇ = zeros(Z.dim * Z.T)
		Threads.@threads for t ∈ times
			vₜ_slice = slice(t, Z.components[name], Z.dim)
            vₜ = Z⃗[vₜ_slice]
			∇[vₜ_slice] = R .* vₜ
		end
		return ∇
	end

	∂²L = nothing
	∂²L_structure = nothing

	if eval_hessian

		∂²L_structure = Z -> begin
            structure = []
            # vₜ Hessian structure (eq. 17)
            for t ∈ times
                vₜ_slice = slice(t, Z.components[name], Z.dim)
                diag_inds = collect(zip(vₜ_slice, vₜ_slice))
                append!(structure, diag_inds)
            end
            return structure
        end

		∂²L = (Z⃗, Z) -> vcat(fill(R, length(times))...)
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    kwargs...
)
    return QuadraticRegularizer(;
        name=name,
        times=1:traj.T,
        dim=traj.dim,
        R=R,
        kwargs...
    )
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Real;
    kwargs...
)
    return QuadraticRegularizer(;
        name=name,
        times=1:traj.T,
        dim=traj.dim,
        R=R * ones(traj.dims[name]),
        kwargs...
    )
end


function QuadraticSmoothnessRegularizer(;
	name::Symbol=nothing,
    times::AbstractVector{Int}=1:traj.T,
	R::AbstractVector{<:Real}=ones(traj.dims[name]),
	eval_hessian=true
)
    @assert !isnothing(name) "name must be specified"
    @assert !isnothing(times) "times must be specified"

    params = Dict(
        :type => :QuadraticSmoothnessRegularizer,
        :name => name,
        :times => times,
        :R => R,
        :eval_hessian => eval_hessian
    )

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
		∑Δv² = 0.0
		for t ∈ times[1:end-1]
			vₜ₊₁ = Z⃗[slice(t + 1, Z.components[name], Z.dim)]
			vₜ = Z⃗[slice(t, Z.components[name], Z.dim)]
			Δv = vₜ₊₁ - vₜ
			∑Δv² += 0.5 * Δv' * (R .* Δv)
		end
		return ∑Δv²
	end

	@views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
		Threads.@threads for t ∈ times[1:end-1]

			vₜ_slice = slice(t, Z.components[name], Z.dim)
			vₜ₊₁_slice = slice(t + 1, Z.components[name], Z.dim)

			vₜ = Z⃗[vₜ_slice]
			vₜ₊₁ = Z⃗[vₜ₊₁_slice]

			Δv = vₜ₊₁ - vₜ

			∇[vₜ_slice] += -R .* Δv
			∇[vₜ₊₁_slice] += R .* Δv
		end
		return ∇
	end
    ∂²L = nothing
	∂²L_structure = nothing

	if eval_hessian

		∂²L_structure = Z -> begin
            structure = []
		# u smoothness regularizer Hessian main diagonal structure

            for t ∈ times

                vₜ_slice = slice(t, Z.components[name], Z.dim)

                # main diagonal (2 if t != 1 or T-1) * Rₛ I
                # components: ∂²vₜSₜ

                append!(
                    structure,
                    collect(
                        zip(
                            vₜ_slice,
                            vₜ_slice
                        )
                    )
                )
            end


            # u smoothness regularizer Hessian off diagonal structure

            for t ∈ times[1:end-1]

                vₜ_slice = slice(t, Z.components[name], Z.dim)
                vₜ₊₁_slice = slice(t + 1, Z.components[name], Z.dim)

                # off diagonal -Rₛ I components: ∂vₜ₊₁∂vₜSₜ

                append!(
                    structure,
                    collect(
                        zip(
                            vₜ_slice,
                            vₜ₊₁_slice
                        )
                    )
                )
            end
            return structure
        end

		∂²L = (Z⃗, Z) -> begin

			H = []

			# u smoothness regularizer Hessian main diagonal values

			append!(H, R)

			for t in times[2:end-1]
				append!(H, 2 * R)
			end

			append!(H, R)


			# u smoothness regularizer Hessian off diagonal values

			for t in times[1:end-1]
				append!(H, -R)
			end

			return H
		end
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function QuadraticSmoothnessRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    kwargs...
)
    return QuadraticSmoothnessRegularizer(;
        name=name,
        times=1:traj.T,
        R=R,
        kwargs...
    )
end

function QuadraticSmoothnessRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Real;
    kwargs...
)
    return QuadraticSmoothnessRegularizer(;
        name=name,
        times=1:traj.T,
        R=R * ones(traj.dims[name]),
        kwargs...
    )
end

function L1Regularizer(;
    name=nothing,
    R::Vector{Float64}=nothing,
    times=nothing,
    eval_hessian=true
)
    @assert !isnothing(name) "name must be specified"
    @assert !isnothing(R) "R must be specified"
    @assert !isnothing(times) "times must be specified"

    s1_name = Symbol("s1_$name")
    s2_name = Symbol("s2_$name")

    params = Dict(
        :type => :L1Regularizer,
        :name => name,
        :R => R,
        :eval_hessian => eval_hessian,
        :times => times,
    )

    L = (Z⃗, Z) -> sum(
        dot(
            R,
            Z⃗[slice(t, Z.components[s1_name], Z.dim)] +
            Z⃗[slice(t, Z.components[s2_name], Z.dim)]
        ) for t ∈ times
    )

    ∇L = (Z⃗, Z) -> begin
        ∇ = zeros(typeof(Z⃗[1]), length(Z⃗))
        Threads.@threads for t ∈ times
            ∇[slice(t, Z.components[s1_name], Z.dim)] += R
            ∇[slice(t, Z.components[s2_name], Z.dim)] += R
        end
        return ∇
    end

    if eval_hessian
        ∂²L = (_, _)  -> []
        ∂²L_structure = _ -> []
    else
        ∂²L = nothing
        ∂²L_structure = nothing
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

@doc raw"""
    L1Regularizer(
        name::Symbol;
        R_value::Float64=10.0,
        R::Vector{Float64}=fill(R_value, length(indices)),
        eval_hessian=true
    )

Create an L1 regularizer for the trajectory component `name` with regularization
strength `R`. The regularizer is defined as

```math
J_{L1}(u) = \sum_t \abs{R \cdot u_t}
```
"""
function L1Regularizer(
    name::Symbol,
    traj::NamedTrajectory;
    indices::AbstractVector{Int}=1:traj.dims[name],
    times=(name ∈ keys(traj.initial) ? 2 : 1):traj.T,
    R_value::Float64=10.0,
    R::Vector{Float64}=fill(R_value, length(indices)),
    eval_hessian=true
)
    J = L1Regularizer(;
        name=name,
        R=R,
        times=times,
        eval_hessian=eval_hessian
    )

    slack_con = L1SlackConstraint(name, traj; indices=indices, times=times)

    return J, slack_con
end


function MinimumTimeObjective(;
    D::Float64=1.0,
    Δt_indices::AbstractVector{Int}=nothing,
    eval_hessian::Bool=true
)
    @assert !isnothing(Δt_indices) "Δt_indices must be specified"

    params = Dict(
        :type => :MinimumTimeObjective,
        :D => D,
        :Δt_indices => Δt_indices,
        :eval_hessian => eval_hessian
    )

    # TODO: amend this for case of no TimeStepsAllEqualConstraint
	L(Z⃗::AbstractVector, Z::NamedTrajectory) = D * sum(Z⃗[Δt_indices])

	∇L = (Z⃗::AbstractVector, Z::NamedTrajectory) -> begin
		∇ = zeros(typeof(Z⃗[1]), length(Z⃗))
		∇[Δt_indices] .= D
		return ∇
	end

	if eval_hessian
		∂²L = (Z⃗, Z) -> []
		∂²L_structure = Z -> []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function MinimumTimeObjective(traj::NamedTrajectory; D=1.0)
    @assert traj.timestep isa Symbol "trajectory does not have a dynamical timestep"
    Δt_indices = [index(t, traj.components[traj.timestep][1], traj.dim) for t = 1:traj.T]
    return MinimumTimeObjective(; D=D, Δt_indices=Δt_indices)
end

end
