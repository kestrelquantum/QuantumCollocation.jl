export AreaConstraint


function AreaConstraint(;
    areas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    comps::Union{AbstractVector{Int}, Nothing}=nothing,
    times::Union{AbstractVector{Int}, Nothing}=nothing,
    timestep::Union{Real, Int, Nothing}=nothing,
    freetime::Union{Bool, Nothing}=nothing,
    zdim::Union{Int, Nothing}=nothing,
    eval_hessian::Bool=false
)

    @assert !isnothing(areas) "must provide areas"
    @assert !isnothing(comps) "must provide components"
    @assert !isnothing(times) "must provide times"
    @assert !isnothing(timestep) "must provide a timestep"
    @assert !isnothing(freetime) "must provide freetime"
    @assert !isnothing(zdim) "must provide a trajectory dimension"

    params = Dict{Symbol, Any}()

    params[:type] = :AreaConstraint
    params[:areas] = areas
    params[:comps] = comps
    params[:times] = times
    params[:timestep] = timestep
    params[:freetime] = freetime
    params[:zdim] = zdim
    params[:eval_hessian] = eval_hessian

    @views function g(Z⃗)
        total = zeros(Real, length(comps))
        for t ∈ times[1:end-1]
            xₜ = Z⃗[slice(t, comps, zdim)]
            xₜ₊₁ = Z⃗[slice(t + 1, comps, zdim)]
            if freetime
                Δt = Z⃗[slice(t, [timestep], zdim)][1]
            else
                Δt = timestep
            end
            total .+= (xₜ + xₜ₊₁) * Δt / 2
        end
        return total .- areas
    end

    ∂g_structure = []
    for (cᵢ, c) ∈ enumerate(comps)
        for t ∈ times
            push!(∂g_structure, (cᵢ, index(t, c, zdim)))
        end
    end
    if freetime
        for t ∈ times[1:end-1]
            for cᵢ ∈ 1:length(comps)
                push!(∂g_structure, (cᵢ, index(t, timestep, zdim)))
            end
        end
    end
    
    @views function ∂g(Z⃗; ipopt=true)
        ∂ = spzeros(Real, length(comps), length(Z⃗))
        for t ∈ times[1:end-1]
            if freetime
                xₜ = Z⃗[slice(t, comps, zdim)]
                xₜ₊₁ = Z⃗[slice(t + 1, comps, zdim)]
                Δt_slice = slice(t, [timestep], zdim)
                ∂[:, Δt_slice] .+= (xₜ + xₜ₊₁) / 2
                Δt = Z⃗[Δt_slice]
            else
                Δt = timestep
            end
            ∂[:, slice(t, comps, zdim)] .+= Δt / 2
            ∂[:, slice(t + 1, comps, zdim)] .+= Δt / 2
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    if eval_hessian
        # TODO
		μ∂²g = (Z⃗) -> []
		μ∂²g_structure = []
	else
        μ∂²g = nothing
        μ∂²g_structure = []
	end

    return NonlinearEqualityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        length(comps),
        params
    )
end

function AreaConstraint(
    name::Symbol,
    areas::AbstractVector{<:Real},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.T,
    name_comps::AbstractVector{Int}=1:traj.dims[name],
    kwargs...
)
    @assert name ∈ traj.names
    comps = traj.components[name][name_comps]
    freetime = traj.timestep isa Symbol
    if freetime
        timestep = traj.components[traj.timestep][1]
    else
        timestep = traj.timestep
    end
    return AreaConstraint(
        areas=areas,
        comps=comps,
        times=times,
        timestep=timestep,
        freetime=freetime,
        zdim=traj.dim,
    )
end

function AreaConstraint(
    names::AbstractVector{Symbol},
    areas::AbstractVector{<:Real},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.T,
    name_comps::AbstractVector{Int}=[1:traj.dims[n] for n ∈ names],
    kwargs...
)
    @assert all(name ∈ traj.names for name ∈ names)
    comps = vcat([traj.components[n][c] for (n, c) ∈ zip(names, name_comps)]...)
    @assert length(areas) == length(comps)
    if freetime
        timestep = traj.components[traj.timestep][1]
    else
        timestep = traj.timestep
    end
    return AreaConstraint(
        areas=areas,
        comps=comps,
        times=times,
        timestep=timestep,
        freetime=freetime,
        zdim=traj.dim,
    )
end
