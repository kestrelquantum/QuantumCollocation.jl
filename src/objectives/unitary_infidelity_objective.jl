export UnitaryInfidelityObjective
export UnitaryFreePhaseInfidelityObjective

###
### UnitaryInfidelityObjective
###

"""
    UnitaryInfidelityObjective

A type of objective that measures the infidelity of a unitary operator to a target unitary operator.

Fields:
    `name`: the name of the unitary operator in the trajectory
    `goal`: the target unitary operator
    `Q`: a scaling factor
    `eval_hessian`: whether to evaluate the Hessian
    `subspace`: the subspace in which to evaluate the objective

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
    if isnothing(subspace)
        l = eval(loss)(name, goal)
    else
        l = eval(loss)(name, goal; subspace=subspace)
    end

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
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
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
        H = spzeros(Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
        Ũ⃗_slice = slice(Z.T, Z.components[name], Z.dim)
        H[Ũ⃗_slice, Ũ⃗_slice] = Q * l(Z⃗[Ũ⃗_slice]; hessian=true)
        if return_moi_vals
            Hs = [H[i,j] for (i, j) ∈ ∂²L_structure(Z)]
            return Hs
        else
            return H
        end
    end

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function UnitaryInfidelityObjective(
    name::Symbol,
    traj::NamedTrajectory,
    Q::Float64;
    subspace=nothing,
	eval_hessian::Bool=true
)
    return UnitaryInfidelityObjective(
        name=name,
        goal=traj.goal[name],
        Q=Q,
        subspace=subspace,
        eval_hessian=eval_hessian
    )
end

###
### UnitaryFreePhaseInfidelityObjective
###

"""
    UnitaryFreePhaseInfidelityObjective

A type of objective that measures the infidelity of a unitary operator to a target unitary operator,
where the target unitary operator is allowed to have phases on qubit subspaces.

Fields:
    `name`: the name of the unitary operator in the trajectory
    `global_name`: the name of the global phase in the trajectory
    `goal`: the target unitary operator
    `Q`: a scaling factor
    `eval_hessian`: whether to evaluate the Hessian
    `subspace`: the subspace in which to evaluate the objective

"""
function UnitaryFreePhaseInfidelityObjective(;
    name::Union{Nothing,Symbol}=nothing,
    goal::Union{Nothing,AbstractVector{<:R}}=nothing,
    phase_name::Union{Nothing,Symbol}=nothing,
    phase_operators::Union{Nothing,AbstractVector{<:AbstractMatrix{<:Complex{R}}}}=nothing,
	Q::R=1.0,
	eval_hessian::Bool=false,
    subspace=nothing
) where R <: Real
    @assert !isnothing(goal) "unitary goal name must be specified"
    @assert !isnothing(name) "unitary name must be specified"
    @assert !isnothing(phase_name) "phase name must be specified"

    loss = :UnitaryFreePhaseInfidelityLoss
    l = eval(loss)(goal, phase_operators; subspace=subspace)

    params = Dict(
        :type => :UnitaryFreePhaseInfidelityObjective,
        :name => name,
        :phase_name => phase_name,
        :goal => goal,
        :phase_operators => phase_operators,
        :Q => Q,
        :eval_hessian => eval_hessian,
        :subspace => subspace
    )

	@views function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        Ũ⃗ = Z⃗[slice(Z.T, Z.components[name], Z.dim)]
        ϕ⃗ = Z⃗[Z.global_components[phase_name]]
        return Q * l(Ũ⃗, ϕ⃗)
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
        Ũ⃗_slice = slice(Z.T, Z.components[name], Z.dim)
        Ũ⃗ = Z⃗[Ũ⃗_slice]
        ϕ⃗_slice = Z.global_components[phase_name]
        ϕ⃗ = Z⃗[ϕ⃗_slice]
        ∇l = l(Ũ⃗, ϕ⃗; gradient=true)
        ∇[Ũ⃗_slice] = Q * ∇l[1:length(Ũ⃗)]
        # WARNING: 2π periodic; using Q≠1 is not recommended
        ∇[ϕ⃗_slice] = Q * ∇l[length(Ũ⃗) .+ (1:length(ϕ⃗))]
        return ∇
    end

    ∂²L_structure(Z::NamedTrajectory) = []
    ∂²L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory) = []

	return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

function UnitaryFreePhaseInfidelityObjective(
    name::Symbol,
    phase_name::Symbol,
    phase_operators::AbstractVector{<:AbstractMatrix{<:Complex}},
    traj::NamedTrajectory,
    Q::Float64;
    subspace=nothing,
	eval_hessian::Bool=true
)
    return UnitaryFreePhaseInfidelityObjective(
        name=name,
        goal=traj.goal[name],
        phase_name=phase_name,
        phase_operators=phase_operators,
        Q=Q,
        subspace=subspace,
        eval_hessian=eval_hessian,
    )
end
