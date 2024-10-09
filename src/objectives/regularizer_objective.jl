export QuadraticRegularizer
export QuadraticSmoothnessRegularizer
export L1Regularizer
export L1Regularizer!
export PairwiseQuadraticRegularizer


###
### Quadratic Regularizer
###

"""
    QuadraticRegularizer

A quadratic regularizer for a trajectory component.

Fields:
    `name`: the name of the trajectory component to regularize
    `times`: the times at which to evaluate the regularizer
    `dim`: the dimension of the trajectory component
    `R`: the regularization matrix
    `baseline`: the baseline values for the trajectory component
    `eval_hessian`: whether to evaluate the Hessian of the regularizer
    `timestep_name`: the symbol for the timestep variable
"""
function QuadraticRegularizer(;
	name::Union{Nothing, Symbol}=nothing,
	times::Union{Nothing, AbstractVector{Int}}=nothing,
    dim::Union{Nothing, Int}=nothing,
	R::Union{Nothing, AbstractVector{<:Real}}=nothing,
    baseline::Union{Nothing, AbstractArray{<:Real}}=nothing,
	eval_hessian::Bool=true,
    timestep_name::Symbol=:Δt
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
                Δt = Z⃗[slice(t, Z.components[timestep_name], Z.dim)]
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
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
        Threads.@threads for t ∈ times
            vₜ_slice = slice(t, Z.components[name], Z.dim)
            Δv = Z⃗[vₜ_slice] .- baseline[:, t]

            if Z.timestep isa Symbol
                Δt_slice = slice(t, Z.components[timestep_name], Z.dim)
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
                    Δt_slice = slice(t, Z.components[timestep_name], Z.dim)
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
                    Δt = Z⃗[slice(t, Z.components[timestep_name], Z.dim)]
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

###
### QuadraticSmoothnessRegularizer
###

"""
    QuadraticSmoothnessRegularizer

A quadratic smoothness regularizer for a trajectory component.

Fields:
    `name`: the name of the trajectory component to regularize
    `times`: the times at which to evaluate the regularizer
    `R`: the regularization matrix
    `eval_hessian`: whether to evaluate the Hessian of the regularizer
"""
function QuadraticSmoothnessRegularizer(;
	name::Symbol=nothing,
    times::Union{Nothing, AbstractVector{Int}}=nothing,
	R::Union{Nothing, AbstractVector{<:Real}}=nothing,
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
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
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
                    collect(zip(vₜ_slice, vₜ_slice))
                )
            end

            # u smoothness regularizer Hessian off diagonal structure
            for t ∈ times[1:end-1]
                vₜ_slice = slice(t, Z.components[name], Z.dim)
                vₜ₊₁_slice = slice(t + 1, Z.components[name], Z.dim)

                # off diagonal -Rₛ I components: ∂vₜ₊₁∂vₜSₜ
                append!(
                    structure,
                    collect(zip(vₜ_slice, vₜ₊₁_slice))
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

###
### L1Regularizer
###

@doc raw"""
    L1Regularizer

Create an L1 regularizer for the trajectory component. The regularizer is defined as

```math
J_{L1}(u) = \sum_t \abs{R \cdot u_t}
```

where \(R\) is the regularization matrix and \(u_t\) is the trajectory component at time \(t\).


"""
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

function L1Regularizer!(
    constraints::Vector{<:AbstractConstraint},
    name::Symbol,
    traj::NamedTrajectory;
    kwargs...
)
    J, slack_con = L1Regularizer(name, traj; kwargs...)
    push!(constraints, slack_con)
    return J
end

###
### PairwiseQuadraticRegularizer
###

@doc raw"""
    PairwiseQuadraticRegularizer

Create a pairwise quadratic regularizer for the trajectory component `name` with
regularization strength `R`. The regularizer is defined as

```math
    J_{v⃗}(u) = \sum_t \frac{1}{2} \Delta t_t^2 (v⃗_{1,t} - v⃗_{2,t})^T R (v⃗_{1,t} - v⃗_{2,t})
```

where $v⃗_{1}$ and $v⃗_{2}$ are selected by `name1` and `name2`. The indices specify the
appropriate block diagonal components of the direct sum vector `v⃗`.

TODO: Hessian not implemented


Fields:
    `R`: the regularization strength
    `times`: the time steps to apply the regularizer
    `name1`: the first name
    `name2`: the second name
    `timestep_name`: the symbol for the timestep
    `eval_hessian`: whether to evaluate the Hessian
"""
function PairwiseQuadraticRegularizer(
    R::AbstractVector{<:Real},
    times::AbstractVector{Int},
    name1::Symbol,
    name2::Symbol;
    timestep_name::Symbol=:Δt,
    eval_hessian::Bool=false,
)
    params = Dict(
        :type => :PairwiseQuadraticRegularizer,
        :times => times,
        :name => (name1, name2),
        :R => R,
        :eval_hessian => eval_hessian,
    )

    @views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        J = 0.0
        for t ∈ times
            if Z.timestep isa Symbol
                Δt = Z⃗[slice(t, Z.components[timestep_name], Z.dim)]
            else
                Δt = Z.timestep
            end
            z1_t = Z⃗[slice(t, Z.components[name1], Z.dim)]
            z2_t = Z⃗[slice(t, Z.components[name1], Z.dim)]
            r_t = Δt * (z1_t .- z2_t)
            J += 0.5 * r_t' * (R .* r_t)
        end
        return J
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
        Threads.@threads for t ∈ times
            z1_t_slice = slice(t, Z.components[name1], Z.dim)
            z2_t_slice = slice(t, Z.components[name2], Z.dim)
            z1_t = Z⃗[z1_t_slice]
            z2_t = Z⃗[z2_t_slice]

            if Z.timestep isa Symbol
                Δt_slice = slice(t, Z.components[timestep_name], Z.dim)
                Δt = Z⃗[Δt_slice]
                ∇[Δt_slice] .= (z1_t .- z2_t)' * (R .* (Δt .* (z1_t .- z2_t)))
            else
                Δt = Z.timestep
            end

            ∇[z1_t_slice] .= R .* (Δt^2 * (z1_t .- z2_t))
            ∇[z2_t_slice] .= R .* (Δt^2 * (z2_t .- z1_t))
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
    PairwiseQuadraticRegularizer

A convenience constructor for creating a PairwiseQuadraticRegularizer for the
trajectory component `name` with regularization strength `Rs` over the graph `graph`.
"""
function PairwiseQuadraticRegularizer(
    traj::NamedTrajectory,
    Rs::Union{Float64, AbstractVector{<:Float64}},
    graph::AbstractVector{<:Tuple{Symbol, Symbol}};
    kwargs...
)
    if isa(Rs, Float64)
        Rs = Rs * ones(length(graph))
    end
    @assert all(length(graph) == length(Rs)) "Graph and Qs must have same length"

    J = NullObjective()
    for (Qᵢⱼ, (symb1, symb2)) ∈ zip(Rs, graph)
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
    R::Float64,
    name1::Symbol,
    name2::Symbol;
    kwargs...
)
    return PairwiseQuadraticRegularizer(
        traj, R, [(name1, name2)];
        kwargs...
    )
end

# ============================================================================ #

@testitem "Quadratic Regularizer Objective" begin
    using LinearAlgebra
    using NamedTrajectories
    using ForwardDiff
    include("../../test/test_utils.jl")

    T = 10

    Z = NamedTrajectory(
        (ψ̃ = randn(4, T), u = randn(2, T)),
        controls=:u,
        timestep=0.1,
        goal=(ψ̃ = [1.0, 0.0, 0.0, 0.0],)
    )


    J = QuadraticRegularizer(:u, Z, [1., 1.])

    L = Z⃗ -> J.L(Z⃗, Z)
    ∇L = Z⃗ -> J.∇L(Z⃗, Z)
    ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
    ∂²L_structure = J.∂²L_structure(Z)

    # test objective function gradient

    @test all(ForwardDiff.gradient(L, Z.datavec) .≈ ∇L(Z.datavec))

    # test objective function hessian
    shape = (Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
    @test all(isapprox(
        ForwardDiff.hessian(L, Z.datavec),
        dense(∂²L(Z.datavec), ∂²L_structure, shape);
        atol=1e-7
    ))
end

@testitem "Quadratic Smoothness Regularizer Objective" begin
    using LinearAlgebra
    using NamedTrajectories
    using ForwardDiff
    include("../../test/test_utils.jl")

    T = 10

    Z = NamedTrajectory(
        (ψ̃ = randn(4, T), u = randn(2, T)),
        controls=:u,
        timestep=0.1,
        goal=(ψ̃ = [1.0, 0.0, 0.0, 0.0],)
    )


    J = QuadraticSmoothnessRegularizer(:u, Z, [1., 1.])

    L = Z⃗ -> J.L(Z⃗, Z)
    ∇L = Z⃗ -> J.∇L(Z⃗, Z)
    ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
    ∂²L_structure = J.∂²L_structure(Z)

    # test objective function gradient

    @test all(ForwardDiff.gradient(L, Z.datavec) .≈ ∇L(Z.datavec))

    # test objective function hessian
    shape = (Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
    @test all(isapprox(
        ForwardDiff.hessian(L, Z.datavec),
        dense(∂²L(Z.datavec), ∂²L_structure, shape);
        atol=1e-7
    ))
end
