export ComplexModulusContraint


"""
    ComplexModulusContraint(<keyword arguments>)

Returns a NonlinearInequalityConstraint on the complex modulus of a complex control

TODO: Changed zdim -> dim. Constraint should be tested for global params.

# Arguments
- `R::Union{Float64,Nothing}=nothing`: the maximum allowed complex modulus
- `comps::Union{AbstractVector{Int},Nothing}=nothing`: the components of the complex control,
   both the real and imaginary parts
- `times::Union{AbstractVector{Int},Nothing}=nothing`: the times at which the constraint is applied
- `dim::Union{Int,Nothing}=nothing`: the dimension of a single time step of the trajectory
- `T::Union{Int,Nothing}=nothing`: the number of time steps
"""
function ComplexModulusContraint(;
    R::Union{Float64, Nothing}=nothing,
    comps::Union{AbstractVector{Int}, Nothing}=nothing,
    times::Union{AbstractVector{Int}, Nothing}=nothing,
    zdim::Union{Int, Nothing}=nothing,
    T::Union{Int, Nothing}=nothing,
)
    @assert !isnothing(R) "must provide a value R, s.t. |z| <= R"
    @assert !isnothing(comps) "must provide components of the complex number"
    @assert !isnothing(times) "must provide times"
    @assert !isnothing(zdim) "must provide a trajectory dimension"
    @assert !isnothing(T) "must provide a T"

    @assert length(comps) == 2 "component must represent a complex number and have dimension 2"

    params = Dict{Symbol, Any}()

    params[:type] = :ComplexModulusContraint
    params[:R] = R
    params[:comps] = comps
    params[:times] = times
    params[:zdim] = zdim
    params[:T] = T

    gₜ(xₜ, yₜ) = [R^2 - xₜ^2 - yₜ^2]
    ∂gₜ(xₜ, yₜ) = [-2xₜ, -2yₜ]
    μₜ∂²gₜ(μₜ) = sparse([1, 2], [1, 2], [-2μₜ, -2μₜ])

    @views function g(Z⃗)
        r = zeros(length(times))
        for (i, t) ∈ enumerate(times)
            zₜ = Z⃗[slice(t, comps, zdim)]
            xₜ = zₜ[1]
            yₜ = zₜ[2]
            r[i] = gₜ(xₜ, yₜ)[1]
        end
        return r
    end

    ∂g_structure = []

    for (i, t) ∈ enumerate(times)
        push!(∂g_structure, (i, index(t, comps[1], zdim)))
        push!(∂g_structure, (i, index(t, comps[2], zdim)))
    end

    @views function ∂g(Z⃗; ipopt=true)
        ∂ = spzeros(length(times), length(Z⃗))
        for (i, t) ∈ enumerate(times)
            zₜ = Z⃗[slice(t, comps, zdim)]
            xₜ = zₜ[1]
            yₜ = zₜ[2]
            ∂[i, slice(t, comps, zdim)] = ∂gₜ(xₜ, yₜ)
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    μ∂²g_structure = []

    for t ∈ times
        push!(
            μ∂²g_structure,
            (
                index(t, comps[1], zdim),
                index(t, comps[1], zdim)
            )
        )
        push!(
            μ∂²g_structure,
            (
                index(t, comps[2], zdim),
                index(t, comps[2], zdim)
            )
        )
    end

    function μ∂²g(Z⃗, μ; ipopt=true)
        n = length(Z⃗)
        μ∂² = spzeros(n, n)
        for (i, t) ∈ enumerate(times)
            t_slice = slice(t, comps, zdim)
            μ∂²[t_slice, t_slice] = μₜ∂²gₜ(μ[i])
        end
        if ipopt
            return [μ∂²[i, j] for (i, j) in μ∂²g_structure]
        else
            return μ∂²
        end
    end

    return NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        length(times),
        params
    )
end

"""
    ComplexModulusContraint(symb::Symbol, R::Float64, traj::NamedTrajectory)

Returns a ComplexModulusContraint for the complex control NamedTrajector symbol
where R is the maximum allowed complex modulus.
"""
function ComplexModulusContraint(
    name::Symbol,
    R::Float64,
    traj::NamedTrajectory;
    times=1:traj.T,
    name_comps=1:traj.dims[name]
)
    @assert name ∈ traj.names
    comps = traj.components[name][name_comps]
    return ComplexModulusContraint(;
        R=R,
        comps=comps,
        times=times,
        zdim=traj.dim,
        T=traj.T
    )
end
