module DirectSums

export append_suffix
export get_suffix
export direct_sum

using ..Integrators
using ..Problems
using ..QuantumSystems
using ..QuantumUtils
using ..Objectives

using NamedTrajectories
using SparseArrays


"""
    direct_sum(A::AbstractMatrix, B::AbstractMatrix)

Returns the direct sum of two matrices.
"""
function direct_sum(A::AbstractMatrix, B::AbstractMatrix)
    return [A spzeros((size(A, 1), size(B, 2))); spzeros((size(B, 1), size(A, 2))) B]
end

"""
    direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)

Returns the direct sum of two sparse matrices.
"""
function direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return blockdiag(A, B)
end

"""
    direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)

Returns the direct sum of two iso_vec operators.
"""
function direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)
    return operator_to_iso_vec(direct_sum(iso_vec_to_operator(Ã⃗), iso_vec_to_operator(B̃⃗)))
end

"""
    direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)

Returns the direct sum of two `QuantumSystem` objects.
"""
function direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)
    H_drift = direct_sum(sys1.H_drift, sys2.H_drift)
    H1_zero = spzeros(size(sys1.H_drift))
    H2_zero = spzeros(size(sys2.H_drift))
    H_drives = [
        [direct_sum(H, H2_zero) for H ∈ sys1.H_drives]...,
        [direct_sum(H1_zero, H) for H ∈ sys2.H_drives]...
    ]
    return QuantumSystem(
        H_drift,
        H_drives,
        params=merge_outer(sys1.params, sys2.params)
    )
end

"""
    direct_sum(traj1::NamedTrajectory, traj2::NamedTrajectory)

Returns the direct sum of two `NamedTrajectory` objects. 

The `NamedTrajectory` objects must have the same timestep. However, a direct sum 
can return a free time problem by passing the keyword argument  `free_time=true`. 
In this case, the timestep symbol must be provided. If a free time problem with more
than two trajectories is desired, the `reduce` function has been written to handle calls 
to direct sums of `NamedTrajectory` objects; simply pass the keyword argument `free_time=true`
to the `reduce` function.

# Arguments
- `traj1::NamedTrajectory`: The first `NamedTrajectory` object.
- `traj2::NamedTrajectory`: The second `NamedTrajectory` object.
- `free_time::Bool=false`: Whether to construct a free time problem.
- `timestep_symbol::Symbol=:Δt`: The timestep symbol to use for free time problems.
"""
function direct_sum(
    traj1::NamedTrajectory, 
    traj2::NamedTrajectory;
    free_time::Bool=false,
    timestep_symbol::Symbol=:Δt,
)
    if traj1.timestep isa Symbol || traj2.timestep isa Symbol
        throw(ArgumentError("Provided trajectories must have fixed timesteps"))
    end
    
    if traj1.timestep != traj2.timestep
        throw(ArgumentError("Fixed timesteps must be equal"))
    end

    # collect component data
    component_names1 = vcat(traj1.state_names..., traj1.control_names...)
    component_names2 = vcat(traj2.state_names..., traj2.control_names...)

    components = merge_outer(
        get_components(component_names1, traj1),
        get_components(component_names2, traj2)
    )
    
    # Add timestep to components
    if free_time
        components = merge_outer(components, NamedTuple{(timestep_symbol,)}([get_timesteps(traj1)]))
    end
    
    return NamedTrajectory(
        components,
        controls=merge_outer(traj1.control_names, traj2.control_names),
        timestep=free_time ? timestep_symbol : traj1.timestep,
        bounds=merge_outer(traj1.bounds, traj2.bounds),
        initial=merge_outer(traj1.initial, traj2.initial),
        final=merge_outer(traj1.final, traj2.final),
        goal=merge_outer(traj1.goal, traj2.goal)
    )
end

function Base.reduce(f::typeof(direct_sum), args::AbstractVector{<:NamedTrajectory}; free_time::Bool=false)
    # init unimplemented for NamedTrajectory to avoid issues with free_time
    if length(args) > 2
        return f(reduce(f, args[1:end-1], free_time=false), args[end], free_time=free_time)
    elseif length(args) == 2
        return f(args[1], args[2], free_time=free_time)
    elseif length(args) == 1
        return args[1]
    else
        throw(DomainError(args, "reducing over an empty collection is not allowed"))
    end
end

function get_components(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory)
    symbs = Tuple(c for c in components)
    vals = [traj[name] for name ∈ components]
    return NamedTuple{symbs}(vals)
end

Base.endswith(symb::Symbol, suffix::AbstractString) = endswith(String(symb), suffix)
Base.endswith(integrator::UnitaryPadeIntegrator, suffix::String) = endswith(integrator.unitary_symb, suffix)
Base.endswith(integrator::DerivativeIntegrator, suffix::String) = endswith(integrator.variable, suffix)
Base.startswith(symb::Symbol, prefix::AbstractString) = startswith(String(symb), prefix)
Base.startswith(symb::Symbol, prefix::Symbol) = startswith(String(symb), String(prefix))

# Append suffix utilities
# -----------------------

append_suffix(symb::Symbol, suffix::String) = Symbol(string(symb, suffix))
append_suffix(symbs::Tuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) = 
    Tuple(s ∈ exclude ? s : append_suffix(s, suffix) for s ∈ symbs)
append_suffix(symbs::AbstractVector, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) = 
    [s ∈ exclude ? s : append_suffix(s, suffix) for s ∈ symbs]
append_suffix(d::Dict{Symbol, Any}, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    typeof(d)(k ∈ exclude ? k : append_suffix(k, suffix) => v for (k, v) ∈ d)

function append_suffix(nt::NamedTuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[])
    symbs = Tuple(k ∈ exclude ? k : append_suffix(k, suffix) for k ∈ keys(nt))
    return NamedTuple{symbs}(values(nt))
end

function append_suffix(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory, suffix::String)
    return append_suffix(get_components(components, traj), suffix)
end

function append_suffix(traj::NamedTrajectory, suffix::String)
    # Timesteps are appended because of bounds and initial/final constraints.
    component_names = vcat(traj.state_names..., traj.control_names...)
    components = append_suffix(component_names, traj, suffix)
    controls = append_suffix(traj.control_names, suffix)
    return NamedTrajectory(
        components;
        controls=controls,
        timestep=traj.timestep isa Symbol ? append_suffix(traj.timestep, suffix) : traj.timestep,
        bounds=append_suffix(traj.bounds, suffix),
        initial=append_suffix(traj.initial, suffix),
        final=append_suffix(traj.final, suffix),
        goal=append_suffix(traj.goal, suffix)
    )
end

append_suffix(integrator::AbstractIntegrator, suffix::String) = 
    modify_integrator_suffix(integrator, suffix, append_suffix)

append_suffix(integrators::AbstractVector{<:AbstractIntegrator}, suffix::String) =
    [append_suffix(integrator, suffix) for integrator ∈ integrators]

function append_suffix(sys::QuantumSystem, suffix::String)
    return QuantumSystem(
        sys.H_drift,
        sys.H_drives,
        params=append_suffix(sys.params, suffix)
    )
end

# Special integrator routines
# ---------------------------

function modify_integrator_suffix(
    integrator::UnitaryPadeIntegrator, 
    suffix::String,
    modifier::Function
)   
    # Just need the matrices
    sys = QuantumSystem(
        QuantumSystems.H(integrator.G_drift), 
        QuantumSystems.H.(integrator.G_drives)
    )
    return UnitaryPadeIntegrator(
        sys,
        modifier(integrator.unitary_symb, suffix),
        modifier(integrator.drive_symb, suffix),
        order=integrator.order,
        autodiff=integrator.autodiff,
        G=integrator.G
    )
end

function modify_integrator_suffix(
    integrator::DerivativeIntegrator, 
    suffix::String,
    modifier::Function
)
    return DerivativeIntegrator(
        modifier(integrator.variable, suffix),
        modifier(integrator.derivative, suffix),
        integrator.dim
    )
end

# remove suffix utilities
# -----------------------

function remove_suffix(s::String, suffix::String)
    if endswith(s, suffix)
        return chop(s, tail=length(suffix))
    else
        error("Suffix '$suffix' not found at the end of '$s'")
    end
end

remove_suffix(symb::Symbol, suffix::String) = Symbol(remove_suffix(String(symb), suffix))
remove_suffix(symbs::Tuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    Tuple(s ∈ exclude ? s : remove_suffix(s, suffix) for s ∈ symbs)
remove_suffix(symbs::AbstractVector, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    [s ∈ exclude ? s : remove_suffix(s, suffix) for s ∈ symbs]
remove_suffix(d::Dict{Symbol, Any}, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    typeof(d)(k ∈ exclude ? k : remove_suffix(k, suffix) => v for (k, v) ∈ d)

function remove_suffix(nt::NamedTuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[])
    symbs = Tuple(k ∈ exclude ? k : remove_suffix(k, suffix) for k ∈ keys(nt))
    return NamedTuple{symbs}(values(nt))
end

remove_suffix(integrator::AbstractIntegrator, suffix::String) =
    modify_integrator_suffix(integrator, suffix, remove_suffix)

remove_suffix(integrators::AbstractVector{<:AbstractIntegrator}, suffix::String) =
    [remove_suffix(integrator, suffix) for integrator ∈ integrators]

# Merge utilities
# ---------------

function merge_outer(nt1::NamedTuple, nt2::NamedTuple)
    common_keys = intersect(keys(nt1), keys(nt2))
    if !isempty(common_keys)
        error("Key collision detected: ", common_keys)
    end
    return merge(nt1, nt2)
end

function merge_outer(d1::Dict{Symbol, <:Any}, d2::Dict{Symbol, <:Any})
    common_keys = intersect(keys(d1), keys(d2))
    if !isempty(common_keys)
        error("Key collision detected: ", common_keys)
    end
    return merge(d1, d2)
end

function merge_outer(s1::AbstractVector, s2::AbstractVector)
    common_keys = intersect(s1, s2)
    if !isempty(common_keys)
        error("Key collision detected: ", common_keys)
    end
    return vcat(s1, s2)
end

function merge_outer(t1::Tuple, t2::Tuple)
    m = merge_outer([tᵢ for tᵢ in t1], [tⱼ for tⱼ in t2])
    return Tuple(mᵢ for mᵢ in m)
end

# Get suffix utilities
# --------------------

function get_suffix(nt::NamedTuple, suffix::String; remove::Bool=false)
    names = Tuple(remove ? remove_suffix(k, suffix) : k for (k, v) ∈ pairs(nt) if endswith(k, suffix))
    values = [v for (k, v) ∈ pairs(nt) if endswith(k, suffix)]
    return NamedTuple{names}(values)
end

function get_suffix(d::Dict{<:Symbol, <:Any}, suffix::String; remove::Bool=false)
    return Dict(remove ? remove_suffix(k, suffix) : k => v for (k, v) ∈ d if endswith(k, suffix))
end

function get_suffix(traj::NamedTrajectory, suffix::String; remove::Bool=false)
    state_names = Tuple(s for s ∈ traj.state_names if endswith(s, suffix))

    # control names
    if traj.timestep isa Symbol
        if endswith(traj.timestep, suffix)
            control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix))
            timestep = remove ? remove_suffix(traj.timestep, suffix) : traj.timestep
            exclude = Symbol[]
        else
            # extract the shared timestep
            control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix) || s == traj.timestep)
            timestep = traj.timestep
            exclude = [timestep]
        end
    else
        control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix))
        timestep = traj.timestep
        exclude = Symbol[]
    end

    component_names = Tuple(vcat(state_names..., control_names...))
    components = get_components(component_names, traj)
    if remove
        components = remove_suffix(components, suffix; exclude=exclude)
    end

    return NamedTrajectory(
        components,
        controls=remove ? remove_suffix(control_names, suffix; exclude=exclude) : control_names,
        timestep=timestep,
        bounds=get_suffix(traj.bounds, suffix, remove=remove),
        initial=get_suffix(traj.initial, suffix, remove=remove),
        final=get_suffix(traj.final, suffix, remove=remove),
        goal=get_suffix(traj.goal, suffix, remove=remove)
    )
end

function get_suffix(integrators::AbstractVector{<:AbstractIntegrator}, suffix::String; remove::Bool=false)
    found = AbstractIntegrator[]
    for integrator ∈ integrators
        if endswith(integrator, suffix)
            push!(found, remove ? remove_suffix(integrator, suffix) : deepcopy(integrator))
        end
    end
    return found
end

function get_suffix(
    prob::QuantumControlProblem,
    suffix::String;
    unitary_prefix::Symbol=:Ũ⃗,
    remove::Bool=false,
)
    # Extract the trajectory
    traj = get_suffix(prob.trajectory, suffix, remove=remove)

    # Extract the integrators
    integrators = get_suffix(prob.integrators, suffix, remove=remove)
    
    # Get direct sum indices
    # TODO: doesn't exclude more than one match
    i₀ = 0
    indices = Int[]
    for (k, d) ∈ pairs(traj.dims)
        if startswith(k, unitary_prefix)
            if endswith(k, suffix)
                # isovec: undo real/imag, undo vectorization
                append!(indices, i₀+1:i₀+isqrt(d ÷ 2))
            else
                i₀ += isqrt(d ÷ 2)
            end
        end
    end

    # Extract the system
    system = QuantumSystem(
        copy(prob.system.H_drift[indices, indices]),
        [copy(H[indices, indices]) for H in prob.system.H_drives if !iszero(H[indices, indices])],
        # params=get_suffix(prob.system.params, suffix)
    )

    # Null objective function
    # TODO: Should we extract past objectives?
    J = NullObjective()

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators
    )
end

end # module