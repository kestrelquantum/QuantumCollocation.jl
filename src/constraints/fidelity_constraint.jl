export FinalFidelityConstraint
export FinalUnitaryFidelityConstraint
export FinalQuantumStateFidelityConstraint
export FinalUnitaryFreePhaseFidelityConstraint

###
### FinalFidelityConstraint
###

"""
    FinalFidelityConstraint(<keyword arguments>)


Returns a NonlinearInequalityConstraint representing a constraint on the
minimum allowed fidelity.

# Arguments
- `fidelity_function::Union{Function,Nothing}=nothing`: the fidelity function
- `value::Union{Float64,Nothing}=nothing`: the minimum fidelity value allowed
   by the constraint
- `comps::Union{AbstractVector{Int},Nothing}=nothing`: the components of the
   state to which the fidelity function is applied
- `goal::Union{AbstractVector{Float64},Nothing}=nothing`: the goal state
- `statedim::Union{Int,Nothing}=nothing`: the dimension of the state
- `zdim::Union{Int,Nothing}=nothing`: the dimension of a single time step of the trajectory
- `T::Union{Int,Nothing}=nothing`: the number of time steps
- `subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing`: the subspace indices of the fidelity

"""
function FinalFidelityConstraint(;
    fidelity_function::Union{Symbol,Function,Nothing}=nothing,
    value::Union{Float64,Nothing}=nothing,
    comps::Union{AbstractVector{Int},Nothing}=nothing,
    goal::Union{AbstractVector{Float64},Nothing}=nothing,
    statedim::Union{Int,Nothing}=nothing,
    zdim::Union{Int,Nothing}=nothing,
    T::Union{Int,Nothing}=nothing,
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    eval_hessian::Bool=true
)
    @assert !isnothing(fidelity_function) "must provide a fidelity function"
    @assert !isnothing(value) "must provide a fidelity value"
    @assert !isnothing(comps) "must provide a list of components"
    @assert !isnothing(goal) "must provide a goal state"
    @assert !isnothing(statedim) "must provide a state dimension"
    @assert !isnothing(zdim) "must provide a z dimension"
    @assert !isnothing(T) "must provide a T"

    if fidelity_function isa Symbol
        fidelity_function_symbol = fidelity_function
        fidelity_function = eval(fidelity_function)
    else
        fidelity_function_symbol = Symbol(fidelity_function)
    end

    if isnothing(subspace)
        fid = x -> fidelity_function(x, goal)
    else
        fid = x -> fidelity_function(x, goal; subspace=subspace)
    end

    @assert fid(randn(statedim)) isa Float64 "fidelity function must return a scalar"

    params = Dict{Symbol, Any}()

    if fidelity_function_symbol ∉ names(Losses)
        @warn "Fidelity function :$(string(fidelity_function_symbol)) is not exported. Unable to save this constraint."
        params[:type] = :FinalFidelityConstraint
        params[:fidelity_function] = :not_saveable
    else
        params[:type] = :FinalFidelityConstraint
        params[:fidelity_function] = fidelity_function_symbol
        params[:value] = value
        params[:comps] = comps
        params[:goal] = goal
        params[:statedim] = statedim
        params[:zdim] = zdim
        params[:T] = T
        params[:subspace] = subspace
        params[:eval_hessian] = eval_hessian
    end

    state_slice = slice(T, comps, zdim)

    ℱ(x) = [fid(x)]

    g(Z⃗) = ℱ(Z⃗[state_slice]) .- value

    ∂ℱ(x) = ForwardDiff.jacobian(ℱ, x)

    ∂ℱ_structure = jacobian_structure(∂ℱ, statedim)

    col_offset = index(T, comps[1] - 1, zdim)

    ∂g_structure = [(i, j + col_offset) for (i, j) in ∂ℱ_structure]

    @views function ∂g(Z⃗; ipopt=true)
        ∂ = spzeros(1, T * zdim)
        ∂ℱ_x = ∂ℱ(Z⃗[state_slice])
        for (i, j) ∈ ∂ℱ_structure
            ∂[i, j + col_offset] = ∂ℱ_x[i, j]
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    if eval_hessian
        ∂²ℱ(x) = ForwardDiff.hessian(fid, x)

        ∂²ℱ_structure = hessian_of_lagrangian_structure(∂²ℱ, statedim, 1)

        μ∂²g_structure = [ij .+ col_offset for ij in ∂²ℱ_structure]

        @views function μ∂²g(Z⃗, μ; ipopt=true)
            HoL = spzeros(T * zdim, T * zdim)
            μ∂²ℱ = μ[1] * ∂²ℱ(Z⃗[state_slice])
            for (i, j) ∈ ∂²ℱ_structure
                HoL[i + col_offset, j + col_offset] = μ∂²ℱ[i, j]
            end
            if ipopt
                return [HoL[i, j] for (i, j) in μ∂²g_structure]
            else
                return HoL
            end
        end

    else
        μ∂²g_structure = nothing
        μ∂²g = nothing
    end

    return NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        1,
        params
    )
end

###
### FinalUnitaryFidelityConstraint
###

"""
    FinalUnitaryFidelityConstraint(statesymb::Symbol, val::Float64, traj::NamedTrajectory)

Returns a FinalFidelityConstraint for the unitary fidelity function where statesymb
is the NamedTrajectory symbol representing the unitary.

"""
function FinalUnitaryFidelityConstraint(
    statesymb::Symbol,
    val::Float64,
    traj::NamedTrajectory;
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    eval_hessian::Bool=true
)
    return FinalFidelityConstraint(;
        fidelity_function=iso_vec_unitary_fidelity,
        value=val,
        comps=traj.components[statesymb],
        goal=traj.goal[statesymb],
        statedim=traj.dims[statesymb],
        zdim=traj.dim,
        T=traj.T,
        subspace=subspace,
        eval_hessian=eval_hessian
    )
end

###
### FinalQuantumStateFidelityConstraint
###

"""
    FinalQuantumStateFidelityConstraint(statesymb::Symbol, val::Float64, traj::NamedTrajectory)

Returns a FinalFidelityConstraint for the unitary fidelity function where statesymb
is the NamedTrajectory symbol representing the unitary.

"""
function FinalQuantumStateFidelityConstraint(
    statesymb::Symbol,
    val::Float64,
    traj::NamedTrajectory;
    kwargs...
)
    @assert statesymb ∈ traj.names
    return FinalFidelityConstraint(;
        fidelity_function=fidelity,
        value=val,
        comps=traj.components[statesymb],
        goal=traj.goal[statesymb],
        statedim=traj.dims[statesymb],
        zdim=traj.dim,
        T=traj.T,
        kwargs...
    )
end

###
### FinalUnitaryFreePhaseFidelityConstraint
###

"""
    FinalUnitaryFreePhaseFidelityConstraint

Returns a NonlinearInequalityConstraint representing a constraint on the minimum allowed fidelity
for a free phase unitary.

"""
function FinalUnitaryFreePhaseFidelityConstraint(;
    value::Union{Float64,Nothing}=nothing,
    state_slice::Union{AbstractVector{Int},Nothing}=nothing,
    phase_slice::Union{AbstractVector{Int},Nothing}=nothing,
    goal::Union{AbstractVector{Float64},Nothing}=nothing,
    phase_operators::Union{AbstractVector{<:AbstractMatrix{<:Complex}},Nothing}=nothing,
    zdim::Union{Int,Nothing}=nothing,
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    eval_hessian::Bool=false
)
    @assert !isnothing(value) "must provide a fidelity value"
    @assert !isnothing(state_slice) "must provide state_slice"
    @assert !isnothing(phase_slice) "must provide phase_slice"
    @assert !isnothing(goal) "must provide a goal state"
    @assert !isnothing(zdim) "must provide a z dimension"

    loss = :UnitaryFreePhaseInfidelityLoss
    ℱ = eval(loss)(goal, phase_operators; subspace=subspace)

    params = Dict(
        :type => :FinalUnitaryFreePhaseFidelityConstraint,
        :loss => loss,
        :value => value,
        :state_slice => state_slice,
        :phase_slice => phase_slice,
        :goal => goal,
        :phase_operators => phase_operators,
        :subspace => subspace,
        :eval_hessian => eval_hessian,
    )

    @views function g(Z⃗)
        Ũ⃗ = Z⃗[state_slice]
        ϕ⃗ = Z⃗[phase_slice]
        return [(1 - value) - ℱ(Ũ⃗, ϕ⃗)]
    end

    ∂g_structure = [(1, j) for j ∈ [state_slice; phase_slice]]

    @views function ∂g(Z⃗; ipopt=true)
        Ũ⃗ = Z⃗[state_slice]
        ϕ⃗ = Z⃗[phase_slice]
        ∂ = -ℱ(Ũ⃗, ϕ⃗; gradient=true)

        if ipopt
            return ∂
        else
            ∂_fill = spzeros(1, zdim)
            for (∂ᵢⱼ, (i, j)) in zip(∂, ∂g_structure)
                ∂_fill[i, j] = ∂ᵢⱼ
            end
            return ∂_fill
        end
    end

    # Hessian
    μ∂²g_structure = []
    μ∂²g = nothing

    return NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        1,
        params
    )
end

function FinalUnitaryFreePhaseFidelityConstraint(
    state_name::Symbol,
    phase_name::Symbol,
    phase_operators::AbstractVector{<:AbstractMatrix{<:Complex}},
    val::Float64,
    traj::NamedTrajectory;
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    eval_hessian::Bool=false
)
    return FinalUnitaryFreePhaseFidelityConstraint(;
        value=val,
        state_slice=slice(traj.T, traj.components[state_name], traj.dim),
        phase_slice=traj.global_components[phase_name],
        goal=traj.goal[state_name],
        phase_operators=phase_operators,
        zdim=length(traj),
        subspace=subspace,
        eval_hessian=eval_hessian
    )
end