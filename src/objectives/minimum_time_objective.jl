export MinimumTimeObjective


"""
    MinimumTimeObjective

A type of objective that counts the time taken to complete a task.

Fields:
    `D`: a scaling factor
    `Δt_indices`: the indices of the time steps
    `eval_hessian`: whether to evaluate the Hessian

"""
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

function MinimumTimeObjective(
    traj::NamedTrajectory;
    D=1.0,
    kwargs...
)
    @assert traj.timestep isa Symbol "trajectory does not have a dynamical timestep"
    Δt_indices = [index(t, traj.components[traj.timestep][1], traj.dim) for t = 1:traj.T]
    return MinimumTimeObjective(; D=D, Δt_indices=Δt_indices, kwargs...)
end