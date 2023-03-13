module Objectives

export Objective

export QuantumObjective
export MinTimeObjective

export QuadraticRegularizer
export QuadraticSmoothnessRegularizer
export L1SlackRegularizer

using ..IndexingUtils
using ..QuantumSystems
using ..Costs

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
	∂²L_structure::Union{Vector{Tuple{Int,Int}}, Nothing}
    terms::Vector{Dict}
end

function Base.:+(obj1::Objective, obj2::Objective)
	L = Z -> obj1.L(Z) + obj2.L(Z)
	∇L = Z -> obj1.∇L(Z) + obj2.∇L(Z)
	if isnothing(obj1.∂²L) && isnothing(obj2.∂²L)
		∂²L = Nothing
		∂²L_structure = Nothing
	elseif isnothing(obj1.∂²L)
		∂²L = Z -> obj2.∂²L(Z)
		∂²L_structure = obj2.∂²L_structure
	elseif isnothing(obj2.∂²L)
		∂²L = Z -> obj1.∂²L(Z)
		∂²L_structure = obj1.∂²L_structure
	else
		∂²L = Z -> vcat(obj1.∂²L(Z), obj2.∂²L(Z))
		∂²L_structure = vcat(
			obj1.∂²L_structure,
			obj2.∂²L_structure
		)
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
    wfn_names::Tuple{Vararg{Symbol}}
	cost=:InfidelityCost,
	Q::Union{Float64, Vector{Float64}}=100.0,
	eval_hessian=true
)
    if Q isa Float64
        Q = ones(length(wfn_names)) * Q
    else
        @assert length(Q) == length(wfn_names)
    end

    params = Dict(
        :type => :QuantumObjective,
        :wfn_names => wfn_names,
        :cost => cost,
        :Q => Q,
        :eval_hessian => eval_hessian,
    )

	function L(Z::NamedTrajectory)
        return sum(Qᵢ * cost(Z, wfn_name) for (Qᵢ, wfn_name) ∈ zip(Q, wfn_names))
    end

    function ∇L(Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T)
        for (Qᵢ, wfn_name) ∈ zip(Q, wfn_names)
            ∇[slice(Z.T, Z.components[wfn_name])] =
                Qᵢ * cost(Z, wfn_name; gradient=true)
        end
        return ∇
    end

    structure = []
    for wfn_name ∈ wfn_names
        wfn_start_idx = Z.components[wfn_name][1]
        wfn_structure = [ij .+ (wfn_start_idx - 1) for ij ∈ cost.∇²l_structure]
        append!(structure, wfn_structure)
    end

    function ∂²L(Z::NamedTrajectory; return_moi_vals=true)
        H = spzeros(Z.dim * Z.T, Z.dim * Z.T)
        for (Qᵢ, wfn_name) ∈ zip(Q, wfn_names)
            H[slice(Z.T, Z.components[wfn_name]), slice(Z.T, Z.components[wfn_name])] =
                Qᵢ * cost(Z, wfn_name; hessian=true)
        end
        if return_moi_vals
            structure = []
            for wfn_name ∈ wfn_names
                wfn_start_idx = Z.components[wfn_name][1]
                wfn_structure = [ij .+ (wfn_start_idx - 1) for ij ∈ cost.∇²l_structure]
                append!(structure, wfn_structure)
            end
            Hs = [H[i,j] for (i, j) ∈ structure]
            return Hs
        else
            return H
        end
    end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end



function QuantumObjective(;
	system::QuantumSystem=nothing,
	cost=:infidelity_cost,
	T=nothing,
	Q=100.0,
	eval_hessian=true
)
    @assert !isnothing(system) "system must be specified"
    @assert !isnothing(T) "T must be specified"

    params = Dict(
        :type => :QuantumObjective,
        :system => system,
        :cost_fn => cost_fn,
        :T => T,
        :Q => Q,
        :eval_hessian => eval_hessian
    )

	cost = QuantumCost(system, cost_fn)

	@views function L(Z::AbstractVector{F}) where F
		ψ̃T = Z[slice(T, system.n_wfn_states, system.vardim)]
		return Q * cost(ψ̃T)
	end

	∇c = QuantumCostGradient(cost)

	@views function ∇L(Z::AbstractVector{F}) where F
		∇ = zeros(F, length(Z))
		ψ̃T_slice = slice(T, system.n_wfn_states, system.vardim)
		ψ̃T = Z[ψ̃T_slice]
		∇[ψ̃T_slice] = Q * ∇c(ψ̃T)
		return ∇
	end

	∂²L = nothing
	∂²L_structure = nothing

	if eval_hessian
		∇²c = QuantumCostHessian(cost)

		# ℓⁱs Hessian structure (eq. 17)
		∂²L_structure = structure(∇²c, T, system.vardim)

		∂²L = Z::AbstractVector -> begin
			ψ̃T = view(
				Z,
				slice(T, system.n_wfn_states, system.vardim)
			)
			return Q * ∇²c(ψ̃T)
		end
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function QuadraticRegularizer(;
	indices::AbstractVector=nothing,
	vardim::Int=nothing,
	times::AbstractVector{Int}=nothing,
	R=1.0,
	eval_hessian=true
)

    @assert !isnothing(indices) "indices must be specified"
    @assert !isnothing(vardim) "vardim must be specified"
    @assert !isnothing(times) "times must be specified"

    params = Dict(
        :type => :QuadraticRegularizer,
        :indices => indices,
        :vardim => vardim,
        :times => times,
        :R => R,
        :eval_hessian => eval_hessian
    )

	@views function L(Z::AbstractVector)
		cost = 0.0
		for t in times
			vₜ = Z[slice(t, indices, vardim)]
			cost += R / 2 * sum(vₜ.^2)
		end
		return cost
	end

	@views function ∇L(Z::AbstractVector{F}) where F
		∇ = zeros(F, length(Z))
		for t in times
			vₜ_slice = slice(t, indices, vardim)
			vₜ = Z[vₜ_slice]
			∇[vₜ_slice] = R * vₜ
		end
		return ∇
	end

	∂²L = nothing
	∂²L_structure = nothing

	if eval_hessian

		∂²L_structure = []

		# vₜ Hessian structure (eq. 17)
		for t in times
			vₜ_slice = slice(
				t,
				indices,
				vardim
			)
			append!(
				∂²L_structure,
				collect(zip(vₜ_slice, vₜ_slice))
			)
		end

		∂²L = Z -> fill(R, length(indices) * length(times))
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


function QuadraticSmoothnessRegularizer(;
	indices::AbstractVector=nothing,
	vardim::Int=nothing,
    times::AbstractVector{Int}=nothing,
	R=1.0,
	eval_hessian=true
)
    @assert !isnothing(indices) "indices must be specified"
    @assert !isnothing(vardim) "vardim must be specified"
    @assert !isnothing(times) "times must be specified"

    params = Dict(
        :type => :QuadraticSmoothnessRegularizer,
        :indices => indices,
        :vardim => vardim,
        :times => times,
        :R => R,
        :eval_hessian => eval_hessian
    )

	@views function L(Z::AbstractVector)
		∑Δv² = 0.0
		for t in times[1:end-1]
			vₜ₊₁ = Z[slice(t + 1, indices, vardim)]
			vₜ = Z[slice(t, indices, vardim)]
			Δv = vₜ₊₁ - vₜ
			∑Δv² += dot(Δv, Δv)
		end
		return 0.5 * R * ∑Δv²
	end

	∇L = (Z::AbstractVector) -> begin

		∇ = zeros(typeof(Z[1]), length(Z))

		for t in times[1:end-1]

			vₜ_slice = slice(t, indices, vardim)
			vₜ₊₁_slice = slice(t + 1, indices, vardim)

			vₜ = Z[vₜ_slice]
			vₜ₊₁ = Z[vₜ₊₁_slice]

			Δv = vₜ₊₁ - vₜ

			∇[vₜ_slice] += -R * Δv
			∇[vₜ₊₁_slice] += R * Δv
		end
		return ∇
	end

	if eval_hessian

		∂²L_structure = []

		# u smoothness regularizer Hessian main diagonal structure

		for t in times

			vₜ_slice = slice(t, indices, vardim)

			# main diagonal (2 if t != 1 or T-1) * Rₛ I
			# components: ∂²vₜSₜ

			append!(
				∂²L_structure,
				collect(
					zip(
						vₜ_slice,
						vₜ_slice
					)
				)
			)
		end


		# u smoothness regularizer Hessian off diagonal structure

		for t in times[1:end-1]

			vₜ_slice = slice(t, indices, vardim)
			vₜ₊₁_slice = slice(t + 1, indices, vardim)

			# off diagonal -Rₛ I components: ∂vₜ₊₁∂vₜSₜ

			append!(
				∂²L_structure,
				collect(
					zip(
						vₜ_slice,
						vₜ₊₁_slice
					)
				)
			)
		end


		∂²L = Z::AbstractVector -> begin

			H = []

			# u smoothness regularizer Hessian main diagonal values

			append!(H, R * ones(length(indices)))

			for t in times[2:end-1]
				append!(H, 2 * R * ones(length(indices)))
			end

			append!(H, R * ones(length(indices)))


			# u smoothness regularizer Hessian off diagonal values

			for t in times[1:end-1]
				append!(H, -R * ones(length(indices)))
			end

			return H
		end
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


function L1SlackRegularizer(;
    s1_indices::AbstractVector{Int}=nothing,
    s2_indices::AbstractVector{Int}=nothing,
    α::Vector{Float64}=fill(1.0, length(s1_indices)),
    eval_hessian=true
)

    @assert !isnothing(s1_indices) "s1_indices must be specified"
    @assert !isnothing(s2_indices) "s2_indices must be specified"

    params = Dict(
        :type => :L1SlackRegularizer,
        :s1_indices => s1_indices,
        :s2_indices => s2_indices,
        :α => α,
        :eval_hessian => eval_hessian
    )

    L(Z) = dot(α, Z[s1_indices] + Z[s2_indices])

    ∇L = Z -> begin
        ∇ = zeros(typeof(Z[1]), length(Z))
        ∇[s1_indices] += α
        ∇[s2_indices] += α
        return ∇
    end

    if eval_hessian
        ∂²L_structure = []
        ∂²L = Z -> []
    else
        ∂²L_structure = nothing
        ∂²L = nothing
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end


function MinTimeObjective(;
    Δt_indices::UnitRange{Int}=nothing,
    T::Int=nothing,
    eval_hessian::Bool=true
)
    @assert !isnothing(Δt_indices) "Δt_indices must be specified"
    @assert !isnothing(T) "T must be specified"

    params = Dict(
        :type => :MinTimeObjective,
        :Δt_indices => Δt_indices,
        :T => T,
        :eval_hessian => eval_hessian
    )

	L(Z::AbstractVector) = sum(Z[Δt_indices])

	∇L = (Z::AbstractVector) -> begin
		∇ = zeros(typeof(Z[1]), length(Z))
		∇[Δt_indices] .= 1.0
		return ∇
	end

	if eval_hessian
		∂²L = Z -> []
		∂²L_structure = []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

end
