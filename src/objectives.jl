module Objectives

export Objective

export NullObjective
export QuantumObjective
export QuantumStateObjective
export QuantumUnitaryObjective
export UnitaryInfidelityObjective

export MinimumTimeObjective
export InfidelityRobustnessObjective

export QuadraticRegularizer
export PairwiseQuadraticRegularizer
export QuadraticSmoothnessRegularizer
export L1Regularizer

using TrajectoryIndexingUtils
using ..QuantumUtils
using ..QuantumSystems
using ..EmbeddedOperators
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

function NullObjective()
    params = Dict(:type => :NullObjective)
	L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory) = 0.0
    ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory) = zeros(Z.dim * Z.T)
    ∂²L_structure(Z::NamedTrajectory) = []
    function ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory; return_moi_vals=true)
        return return_moi_vals ? [] : spzeros(Z.dim * Z.T, Z.dim * Z.T)
    end
	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
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
        ∇[Ũ⃗_slice] = Q * ∇l
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
    subspace=nothing,
	eval_hessian::Bool=true
)
    return UnitaryInfidelityObjective(name=name, goal=traj.goal[name], Q=Q, subspace=subspace, eval_hessian=eval_hessian)
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
	name::Union{Nothing, Symbol}=nothing,
	times::Union{Nothing, AbstractVector{Int}}=nothing,
    dim::Union{Nothing, Int}=nothing,
	R::Union{Nothing, AbstractVector{<:Real}}=nothing,
    baseline::Union{Nothing, AbstractArray{<:Real}}=nothing,
	eval_hessian::Bool=true,
    timestep_symbol::Symbol=:Δt
)

    @assert !isnothing(name) "name must be specified"
    @assert !isnothing(times) "times must be specified"
    @assert !isnothing(dim) "dim must be specified"
    @assert !isnothing(R) "R must be specified"
    if isnothing(baseline)
        baseline = zeros(length(R), length(times))
    else
        if size(baseline) != (length(R), length(times))
            throw(ArgumentError("size(baseline)=$(size(baseline)) must match $(length(R)) x $(length(times))"))
        end
    end
    
    params = Dict(
        :type => :QuadraticRegularizer,
        :name => name,
        :times => times,
        :dim => dim,
        :baseline => baseline,
        :R => R,
        :eval_hessian => eval_hessian
    )

    @views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        J = 0.0
        for t ∈ times
            if Z.timestep isa Symbol
                Δt = Z⃗[slice(t, Z.components[timestep_symbol], Z.dim)]
            else
                Δt = Z.timestep
            end
            
            vₜ = Z⃗[slice(t, Z.components[name], Z.dim)]
            Δv = vₜ .- baseline[:, t]

            rₜ = Δt .* Δv
            J += 0.5 * rₜ' * (R .* rₜ)
        end
        return J
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)        
        Threads.@threads for t ∈ times
            vₜ_slice = slice(t, Z.components[name], Z.dim)
            Δv = Z⃗[vₜ_slice] .- baseline[:, t]

            if Z.timestep isa Symbol
                Δt_slice = slice(t, Z.components[timestep_symbol], Z.dim)
                Δt = Z⃗[Δt_slice]
                ∇[Δt_slice] .= Δv' * (R .* (Δt .* Δv))
            else
                Δt = Z.timestep
            end

            ∇[vₜ_slice] .= R .* (Δt.^2 .* Δv)
        end
        return ∇
    end

    ∂²L = nothing
    ∂²L_structure = nothing

    if eval_hessian

        ∂²L_structure = Z -> begin
            structure = []
            # Hessian structure (eq. 17)
            for t ∈ times
                vₜ_slice = slice(t, Z.components[name], Z.dim)
                vₜ_vₜ_inds = collect(zip(vₜ_slice, vₜ_slice))
                append!(structure, vₜ_vₜ_inds)

                if Z.timestep isa Symbol
                    Δt_slice = slice(t, Z.components[timestep_symbol], Z.dim)
                    # ∂²_vₜ_Δt
                    vₜ_Δt_inds = [(i, j) for i ∈ vₜ_slice for j ∈ Δt_slice]
                    append!(structure, vₜ_Δt_inds)
                    # ∂²_Δt_vₜ
                    Δt_vₜ_inds = [(i, j) for i ∈ Δt_slice for j ∈ vₜ_slice]
                    append!(structure, Δt_vₜ_inds)
                    # ∂²_Δt_Δt
                    Δt_Δt_inds = collect(zip(Δt_slice, Δt_slice))
                    append!(structure, Δt_Δt_inds)
                end
            end
            return structure
        end

        ∂²L = (Z⃗, Z) -> begin
            values = []
            # Match Hessian structure indices
            for t ∈ times
                if Z.timestep isa Symbol
                    Δt = Z⃗[slice(t, Z.components[timestep_symbol], Z.dim)]
                    append!(values, R .* Δt.^2)
                    # ∂²_vₜ_Δt, ∂²_Δt_vₜ
                    vₜ = Z⃗[slice(t, Z.components[name], Z.dim)]
                    Δv = vₜ .- baseline[:, t]
                    append!(values, 2 * (R .* (Δt .* Δv)))
                    append!(values, 2 * (R .* (Δt .* Δv)))
                    # ∂²_Δt_Δt
                    append!(values, Δv' * (R .* Δv))
                else
                    Δt = Z.timestep
                    append!(values, R .* Δt.^2)
                end
            end
            return values
        end
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

@doc raw"""
    PairwiseQuadraticRegularizer(
        Q::AbstractVector{<:Real},
        times::AbstractVector{Int},
        name1::Symbol,
        name2::Symbol;
        timestep_symbol::Symbol=:Δt,
        eval_hessian::Bool=false,
    )

Create a pairwise quadratic regularizer for the trajectory component `name` with
regularization strength `Q`. The regularizer is defined as

```math
    J_{Ũ⃗}(u) = \sum_t \frac{1}{2} \Delta t_t^2 (Ũ⃗_{1,t} - Ũ⃗_{2,t})^T Q (Ũ⃗_{1,t} - Ũ⃗_{2,t})
```

where $Ũ⃗_{1}$ and $Ũ⃗_{2}$ are selected by `name1` and `name2`. The
indices specify the appropriate block diagonal components of the direct sum 
unitary vector `Ũ⃗`.

TODO: Hessian not implemented
"""
function PairwiseQuadraticRegularizer(
    Q::AbstractVector{<:Real},
    times::AbstractVector{Int},
    name1::Symbol,
    name2::Symbol;
    timestep_symbol::Symbol=:Δt,
    eval_hessian::Bool=false,
)
    params = Dict(
        :type => :PairwiseQuadraticRegularizer,
        :times => times,
        :name => (name1, name2),
        :Q => Q,
        :eval_hessian => eval_hessian,
    )

    @views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        J = 0.0
        for t ∈ times
            if Z.timestep isa Symbol
                Δt = Z⃗[slice(t, Z.components[timestep_symbol], Z.dim)]
            else
                Δt = Z.timestep
            end
            z1_t = Z⃗[slice(t, Z.components[name1], Z.dim)]
            z2_t = Z⃗[slice(t, Z.components[name1], Z.dim)]
            r_t = Δt * (z1_t .- z2_t)
            J += 0.5 * r_t' * (Q .* r_t)
        end
        return J
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)        
        Threads.@threads for t ∈ times
            z1_t_slice = slice(t, Z.components[name1], Z.dim)
            z2_t_slice = slice(t, Z.components[name2], Z.dim)
            z1_t = Z⃗[z1_t_slice]
            z2_t = Z⃗[z2_t_slice]
            
            if Z.timestep isa Symbol
                Δt_slice = slice(t, Z.components[timestep_symbol], Z.dim)
                Δt = Z⃗[Δt_slice]
                ∇[Δt_slice] .= (z1_t .- z2_t)' * (Q .* (Δt .* (z1_t .- z2_t)))
            else
                Δt = Z.timestep
            end

            ∇[z1_t_slice] .= Q .* (Δt^2 * (z1_t .- z2_t))
            ∇[z2_t_slice] .= Q .* (Δt^2 * (z2_t .- z1_t))
        end
        return ∇
    end
    
    # TODO: Hessian not implemented
    ∂²L = nothing
    ∂²L_structure = nothing

    if eval_hessian
        throw(ErrorException("Hessian not implemented"))
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

@doc raw"""
PairwiseQuadraticRegularizer(
        Qs::AbstractVector{<:Real},
        graph::AbstractVector{<:AbstractVector{Symbol}},
        num_systems::Int;
        name::Symbol=:Ũ⃗,
        kwargs...
    )

A convenience constructor for creating a PairwiseQuadraticRegularizer
for the trajectory component `name` with regularization strength `Qs` over the
graph `graph`. 

The regularizer is defined as

```math
J_{Ũ⃗}(u) = \sum_{(i,j) \in E} \frac{1}{2} d(Ũ⃗_{i}, Ũ⃗_{j}; Q_{ij})
```

where $d(Ũ⃗_{i}, Ũ⃗_{j}; Q_{ij})$ is the pairwise distance between the unitaries with 
weight $Q_{ij}$.
"""
function PairwiseQuadraticRegularizer(
    traj::NamedTrajectory,
    Qs::Union{Float64, AbstractVector{<:Float64}},
    graph::AbstractVector{<:Tuple{Symbol, Symbol}};
    kwargs...
)       
    if isa(Qs, Float64)
        Qs = Qs * ones(length(graph))
    end
    @assert all(length(graph) == length(Qs)) "Graph and Qs must have same length"

    J = NullObjective()
    for (Qᵢⱼ, (symb1, symb2)) ∈ zip(Qs, graph)
        # Symbols should be the same size
        dim = size(traj[symb1], 1)
        J += PairwiseQuadraticRegularizer(
            Qᵢⱼ * ones(dim), 
            1:traj.T,
            symb1,
            symb2,
            kwargs...
        )  
    end

    return J
end

function PairwiseQuadraticRegularizer(
    traj::NamedTrajectory,
    Q::Float64,
    name1::Symbol,
    name2::Symbol;
    kwargs...
)
    return PairwiseQuadraticRegularizer(
        traj, Q, [(name1, name2)];
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

@doc raw"""
InfidelityRobustnessObjective(
    H_error::AbstractMatrix{<:Number},
    Z::NamedTrajectory;
    eval_hessian::Bool=false,
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing
)

Create a control objective which penalizes the sensitivity of the infidelity
to the provided operator defined in the subspace of the control dynamics,
thereby realizing robust control.

The control dynamics are
```math
U_C(a)= \prod_t \exp{-i H_C(a_t)}
```

In the control frame, the H_error operator is (proportional to)
```math
R_{Robust}(a) = \frac{1}{T \norm{H_e}_2} \sum_t U_C(a_t)^\dag H_e U_C(a_t) \Delta t
```
where we have adjusted to a unitless expression of the operator.

The robustness objective is
```math
R_{Robust}(a) = \frac{1}{N} \norm{R}^2_F
```
where N is the dimension of the Hilbert space.
"""
function InfidelityRobustnessObjective(
    H_error::AbstractMatrix{<:Number},
    Z::NamedTrajectory;
    eval_hessian::Bool=false,
    subspace::AbstractVector{<:Integer}=collect(1:size(H_error, 1)),
    state_symb::Symbol=:Ũ⃗
)
    # Indices of all non-zero subspace components for iso_vec_operators
    function iso_vec_subspace(subspace::AbstractVector{<:Integer}, Z::NamedTrajectory)
        d = isqrt(Z.dims[state_symb] ÷ 2)
        A = zeros(Complex, d, d)
        A[subspace, subspace] .= 1 + im
        # Return any index where there is a 1.
        return [j for (s, j) ∈ zip(operator_to_iso_vec(A), Z.components[state_symb]) if convert(Bool, s)]
    end
    ivs = iso_vec_subspace(subspace, Z)

    @views function get_timesteps(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        return map(1:Z.T) do t
            if Z.timestep isa Symbol
                Z⃗[slice(t, Z.components[Z.timestep], Z.dim)][1]
            else
                Z.timestep
            end
        end
    end

    # Control frame
    @views function rotate(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        Δts = get_timesteps(Z⃗, Z)
        T = sum(Δts)
        R = sum(
            map(1:Z.T) do t
                Uₜ = iso_vec_to_operator(Z⃗[slice(t, ivs, Z.dim)])
                Uₜ'H_error*Uₜ .* Δts[t]
            end
        ) / norm(H_error) / T
        return R
    end

    function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        R = rotate(Z⃗, Z)
        return real(tr(R'R)) / size(R, 1)
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
        R = rotate(Z⃗, Z)
        Δts = get_timesteps(Z⃗, Z)
        T = sum(Δts)
        units = 1 / norm(H_error) / T
        Threads.@threads for t ∈ 1:Z.T
            # State
            Uₜ_slice = slice(t, ivs, Z.dim)
            Uₜ = iso_vec_to_operator(Z⃗[Uₜ_slice])

            # State gradient
            ∇[Uₜ_slice] .= operator_to_iso_vec(2 * H_error * Uₜ * R * Δts[t]) * units

            # Time gradient
            if Z.timestep isa Symbol
                ∂R = Uₜ'H_error*Uₜ
                ∇[slice(t, Z.components[Z.timestep], Z.dim)] .= tr(∂R*R + R*∂R) * units
            end
        end
        return ∇ / size(R, 1)
    end

    # Hessian is dense (Control frame R contains sum over all unitaries).
    if eval_hessian
        # TODO
		∂²L = (Z⃗, Z) -> []
		∂²L_structure = Z -> []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

    params = Dict(
        :type => :QuantumRobustnessObjective,
        :error => H_error,
        :eval_hessian => eval_hessian
    )

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function InfidelityRobustnessObjective(
    H1_error::AbstractMatrix{<:Number},
    H2_error::AbstractMatrix{<:Number},
    state1_symb::Symbol=:Ũ⃗1,
    state2_symb::Symbol=:Ũ⃗2;
    eval_hessian::Bool=false,
    # subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing
)
    @views function get_timesteps(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        return map(1:Z.T) do t
            if Z.timestep isa Symbol
                Z⃗[slice(t, Z.components[Z.timestep], Z.dim)][1]
            else
                Z.timestep
            end
        end
    end

    function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        Δts = get_timesteps(Z⃗, Z)
        T = sum(Δts)
        R = 0.0
        for (i₁, Δt₁) ∈ enumerate(Δts)
            for (i₂, Δt₂) ∈ enumerate(Δts)
                # States
                U1ₜ₁ = iso_vec_to_operator(Z⃗[slice(i₁, Z.components[state1_symb], Z.dim)])
                U1ₜ₂ = iso_vec_to_operator(Z⃗[slice(i₂, Z.components[state1_symb], Z.dim)])
                U2ₜ₁ = iso_vec_to_operator(Z⃗[slice(i₁, Z.components[state2_symb], Z.dim)])
                U2ₜ₂ = iso_vec_to_operator(Z⃗[slice(i₂, Z.components[state2_symb], Z.dim)])

                # Rotating frame
                rH1ₜ₁ = U1ₜ₁'H1_error*U1ₜ₁
                rH1ₜ₂ = U1ₜ₂'H1_error*U1ₜ₂
                rH2ₜ₁ = U2ₜ₁'H2_error*U2ₜ₁
                rH2ₜ₂ = U2ₜ₂'H2_error*U2ₜ₂

                # Robustness
                units = 1 / T^2 / norm(H1_error)^2 / norm(H2_error)^2
                R += real(tr(rH1ₜ₁'rH1ₜ₂) * tr(rH2ₜ₁'rH2ₜ₂) * Δt₁ * Δt₂ * units)
            end
        end
        return R / size(H1_error, 1) / size(H2_error, 1)
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
        Δts = get_timesteps(Z⃗, Z)
        T = sum(Δts)
        Threads.@threads for (i₁, i₂) ∈ vec(collect(Iterators.product(1:Z.T, 1:Z.T)))
            # Times
            Δt₁ = Δts[i₁]
            Δt₂ = Δts[i₂]

            # States
            U1ₜ₁_slice = slice(i₁, Z.components[state1_symb], Z.dim)
            U1ₜ₂_slice = slice(i₂, Z.components[state1_symb], Z.dim)
            U2ₜ₁_slice = slice(i₁, Z.components[state2_symb], Z.dim)
            U2ₜ₂_slice = slice(i₂, Z.components[state2_symb], Z.dim)
            U1ₜ₁ = iso_vec_to_operator(Z⃗[U1ₜ₁_slice])
            U1ₜ₂ = iso_vec_to_operator(Z⃗[U1ₜ₂_slice])
            U2ₜ₁ = iso_vec_to_operator(Z⃗[U2ₜ₁_slice])
            U2ₜ₂ = iso_vec_to_operator(Z⃗[U2ₜ₂_slice])

            # Rotating frame
            rH1ₜ₁ = U1ₜ₁'H1_error*U1ₜ₁
            rH1ₜ₂ = U1ₜ₂'H1_error*U1ₜ₂
            rH2ₜ₁ = U2ₜ₁'H2_error*U2ₜ₁
            rH2ₜ₂ = U2ₜ₂'H2_error*U2ₜ₂
            
            # ∇Uiₜⱼ (assume H's are Hermitian)
            units = 1 / T^2 / norm(H1_error)^2 / norm(H2_error)^2
            R1 = tr(rH1ₜ₁'rH1ₜ₂) * Δt₁ * Δt₂ * units
            R2 = tr(rH2ₜ₁'rH2ₜ₂) * Δt₁ * Δt₂ * units
            ∇[U1ₜ₁_slice] += operator_to_iso_vec(2 * H1_error * U1ₜ₁ * rH1ₜ₂) * R2 
            ∇[U1ₜ₂_slice] += operator_to_iso_vec(2 * H1_error * U1ₜ₂ * rH1ₜ₁) * R2 
            ∇[U2ₜ₁_slice] += operator_to_iso_vec(2 * H2_error * U2ₜ₁ * rH2ₜ₂) * R1 
            ∇[U2ₜ₂_slice] += operator_to_iso_vec(2 * H2_error * U2ₜ₂ * rH2ₜ₁) * R1

            # Time gradients
            if Z.timestep isa Symbol
                R = real(tr(rH1ₜ₁'rH1ₜ₂) * tr(rH2ₜ₁'rH2ₜ₂)) * units
                ∇[slice(i₁, Z.components[Z.timestep], Z.dim)] .= R * Δt₂
                ∇[slice(i₂, Z.components[Z.timestep], Z.dim)] .= R * Δt₁
            end
        end
        return ∇ / size(H1_error, 1) / size(H2_error, 1)
    end

    # Hessian is dense (Control frame R contains sum over all unitaries).
    if eval_hessian
        # TODO
		∂²L = (Z⃗, Z) -> []
		∂²L_structure = Z -> []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

    params = Dict(
        :type => :QuantumRobustnessObjective,
        :error => H1_error ⊗ H2_error,
        :eval_hessian => eval_hessian
    )

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


end
