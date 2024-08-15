export QuantumObjective
export QuantumUnitaryObjective
export QuantumStateObjective

###
### QuantumObjective
###

"""
    QuantumObjective

    A generic objective function for quantum trajectories that use a loss.

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
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
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
        H = spzeros(Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
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

function QuantumObjective(
    name::Symbol,
    traj::NamedTrajectory,
    loss::Symbol,
    Q::Float64
)
    goal = traj.goal[name]
    return QuantumObjective(name=name, goals=goal, loss=loss, Q=Q)
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

###
### Example: Default quantum objectives
###

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

# ============================================================================= # 

@testitem "Quantum State Objective" begin
    using LinearAlgebra
    using NamedTrajectories
    using ForwardDiff
    include("../../test/test_utils.jl")
    
    T = 10

    Z = NamedTrajectory(
        (ψ̃ = randn(4, T), u = randn(2, T)),
        controls=:u,
        timestep=0.1,
        goal=(ψ̃ = [1., 0., 0., 0.],)
    )

    loss = :InfidelityLoss
    Q = 100.0

    J = QuantumObjective(:ψ̃, Z, loss, Q)

    L = Z⃗ -> J.L(Z⃗, Z)
    ∇L = Z⃗ -> J.∇L(Z⃗, Z)
    ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
    ∂²L_structure = J.∂²L_structure(Z)

    # test objective function gradient
    @test ForwardDiff.gradient(L, Z.datavec) ≈ ∇L(Z.datavec)

    # test objective function hessian
    shape = (Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
    @test ForwardDiff.hessian(L, Z.datavec) ≈ dense(∂²L(Z.datavec), ∂²L_structure, shape)
end

@testitem "Unitary Objective" begin
    using LinearAlgebra
    using NamedTrajectories
    using ForwardDiff
    include("../../test/test_utils.jl")

    T = 10

    U_init = GATES[:I]
    U_goal = GATES[:X]

    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)

    Z = NamedTrajectory(
        (Ũ⃗ = randn(length(Ũ⃗_init), T), u = randn(2, T)),
        controls=:u,
        timestep=0.1,
        initial=(Ũ⃗ = Ũ⃗_init,),
        goal=(Ũ⃗ = Ũ⃗_goal,)
    )

    loss = :UnitaryInfidelityLoss
    Q = 100.0

    J = QuantumObjective(:Ũ⃗, Z, loss, Q)

    L = Z⃗ -> J.L(Z⃗, Z)
    ∇L = Z⃗ -> J.∇L(Z⃗, Z)
    ∂²L = Z⃗ -> J.∂²L(Z⃗, Z)
    ∂²L_structure = J.∂²L_structure(Z)

    # test objective function gradient
    @test all(ForwardDiff.gradient(L, Z.datavec) ≈ ∇L(Z.datavec))

    # test objective function hessian
    shape = (Z.dim * Z.T + Z.global_dim, Z.dim * Z.T + Z.global_dim)
    H = dense(∂²L(Z.datavec), ∂²L_structure, shape)
    H_forwarddiff = ForwardDiff.hessian(L, Z.datavec)
    @test all(H .≈ H_forwarddiff)
end