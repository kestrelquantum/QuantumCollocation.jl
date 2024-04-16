@doc """
    UnitaryDirectSumProblem(probs, final_fidelity; kwargs...)

Construct a `QuantumControlProblem` as a direct sum of unitary gate problems. The 
purpose is to find solutions that are as close as possible with respect to one of 
their components. In particular, this is useful for finding interpolatable control solutions.

A graph of edges (specified by problem labels) will enforce a `PairwiseQuadraticRegularizer` between
the component trajectories of the problem in `probs` corresponding to the names of the edge in `edges`
with corresponding edge weight `Q`.

Boundary values can be included to enforce a `QuadraticRegularizer` on edges where one of the nodes is 
not optimized. The boundary values are specified as a dictionary with keys corresponding to the edge
labels and values corresponding to the boundary values.

The default behavior is to use a 1D chain for the graph, i.e., enforce a `PairwiseQuadraticRegularizer`
between each neighbor of the provided `probs`.

# Arguments
- `probs::AbstractVector{<:QuantumControlProblem}`: the problems to combine
- `final_fidelity::Real`: the fidelity to enforce between the component final unitaries and the component goal unitaries

# Keyword Arguments
- `prob_labels::AbstractVector{<:String}}`: the labels for the problems
-  graph::Union{Nothing, AbstractVector{<:Tuple{String, String}}, AbstractVector{<:Tuple{Symbol, Symbol}}}`: the graph of edges to enforce
- `boundary_values::Union{Nothing, AbstractDict{<:String, <:AbstractArray}, AbstractDict{<:Symbol, <:AbstractVector}}=nothing`: the boundary values for the problems
- `Q::Union{Float64, Vector{Float64}}=100.0`: the weights on the pairwise regularizers
- `Q_symb::Symbol=:Ũ⃗`: the symbol to use for the regularizer
- `R::Float64=1e-2`: the shared weight on all control terms (:a, :da, :dda is assumed)
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `R_b::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the boundary values
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `drive_reset_ratio::Float64=0.1`: amount of random noise to add to the control data (can help avoid hitting restoration if provided problems are converged)
- `subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing`: the subspace to use for the fidelity of each problem
- `max_iter::Int=1000`: the maximum number of iterations for the Ipopt solver
- `linear_solver::String="mumps"`: the linear solver to use in Ipopt
- `hessian_approximation=true`: whether or not to use L-BFGS hessian approximation in Ipopt
- `jacobian_structure=true`: whether or not to use the jacobian structure in Ipopt
- `blas_multithreading=true`: whether or not to use multithreading in BLAS
- `ipopt_options::Options=Options()`: the options for the Ipopt solver

"""
function UnitaryDirectSumProblem(
    probs::AbstractVector{<:QuantumControlProblem},
    final_fidelity::Real;
    prob_labels::AbstractVector{<:String}=[string(i) for i ∈ 1:length(probs)],
    graph::Union{Nothing, AbstractVector{<:Tuple{String, String}}, AbstractVector{<:Tuple{Symbol, Symbol}}}=nothing,
    boundary_values::Union{AbstractDict{<:String, <:AbstractArray}, AbstractDict{<:Symbol, <:AbstractArray}}=Dict{String, Array}(),
    Q::Union{Float64, Vector{Float64}}=100.0,
    Q_symb::Symbol=:dda,
    R::Float64=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    R_b::Union{Float64, Vector{Float64}}=R,
    drive_derivative_σ::Float64=0.01,
    drive_reset_ratio::Float64=0.50,
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    verbose::Bool=false,
    hessian_approximation=true,
    jacobian_structure=true,
    blas_multithreading=true,
    ipopt_options=Options(),
    kwargs...
)
    N = length(probs)
    if length(prob_labels) != N
        throw(ArgumentError("Length of prob_labels must match length of probs"))
    end

    if N < 2
        throw(ArgumentError("At least two problems are required"))
    end

    if drive_reset_ratio < 0 || drive_reset_ratio > 1
        throw(ArgumentError("drive_reset_σ must be in [0, 1]"))
    end

    if !isempty(intersect(keys(boundary_values), prob_labels))
        throw(ArgumentError("Boundary value keys cannot be in prob_labels"))
    end

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    # Default chain graph and boundary
    boundary = Tuple{Symbol, Array}[]
    if isnothing(graph)
        graph = [
            (apply_suffix(Q_symb, i), apply_suffix(Q_symb, j))
            for (i, j) ∈ zip(prob_labels[1:N-1], prob_labels[2:N])
        ]
    else
        # Check that String edge labels are in prob_labels or boundary, and make Symbols
        if eltype(eltype(graph)) == String
            graph_symbols = Tuple{Symbol, Symbol}[]
            for edge in graph
                if edge[1] in prob_labels && edge[2] in prob_labels
                    push!(graph_symbols, (apply_suffix(Q_symb, edge[1]), apply_suffix(Q_symb, edge[2])))
                elseif edge[1] in keys(boundary_values) && edge[2] in prob_labels
                    push!(boundary, (apply_suffix(Q_symb, edge[2]), boundary_values[edge[1]]))
                elseif edge[1] in prob_labels && edge[2] in keys(boundary_values)
                    push!(boundary, (apply_suffix(Q_symb, edge[1]), boundary_values[edge[2]]))
                else
                    throw(ArgumentError("Edge labels must be in prob_labels or boundary_values"))
                end
            end
            graph = graph_symbols
        end
    end

    # Build the direct sum system

    # merge suffix trajectories
    traj = reduce(
        merge_outer, 
        [apply_suffix(p.trajectory, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # add noise to control data (avoid restoration)
    if drive_reset_ratio > 0
        σs = repeat([drive_derivative_σ], 2)
        for ℓ in prob_labels
            a_symb = apply_suffix(:a, ℓ)
            da_symb = apply_suffix(:da, ℓ)
            dda_symb = apply_suffix(:dda, ℓ)
            a_bounds_lower, a_bounds_upper = traj.bounds[a_symb]
            a, da, dda = randomly_fill_drives(traj.T, a_bounds_lower, a_bounds_upper, σs)
            update!(traj, a_symb, (1 - drive_reset_ratio) * traj[a_symb] + drive_reset_ratio * a)
            update!(traj, da_symb, (1 - drive_reset_ratio) * traj[da_symb] + drive_reset_ratio * da)
            update!(traj, dda_symb, (1 - drive_reset_ratio) * traj[dda_symb] + drive_reset_ratio * dda)
        end
    end

    # concatenate suffix integrators
    integrators = vcat(
        [apply_suffix(p.integrators, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]...
    )

    # direct sum (used for problem saving, only)
    system = reduce(
        direct_sum, 
        [apply_suffix(p.system, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # Rebuild trajectory constraints
    build_trajectory_constraints = true
    constraints = AbstractConstraint[]

    # Add fidelity constraints for each problem
    for (p, ℓ) ∈ zip(probs, prob_labels)
        goal_symb, = keys(p.trajectory.goal)
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            apply_suffix(goal_symb, ℓ),
            final_fidelity,
            traj;
            subspace=subspace
        )
        push!(constraints, fidelity_constraint)
    end

    # Build the objective function
    J = PairwiseQuadraticRegularizer(traj, Q, graph)

    for (symb, s₀) ∈ boundary
        J += QuadraticRegularizer(symb, traj, R_b; baseline=s₀)
    end

    for ℓ ∈ prob_labels
        J += QuadraticRegularizer(apply_suffix(:a, ℓ), traj, R_a)
        J += QuadraticRegularizer(apply_suffix(:da, ℓ), traj, R_da)
        J += QuadraticRegularizer(apply_suffix(:dda, ℓ), traj, R_dda)
    end

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
        jacobian_structure=jacobian_structure,
        hessian_approximation=hessian_approximation,
        eval_hessian=!hessian_approximation,
        build_trajectory_constraints=build_trajectory_constraints
    )
end

# *************************************************************************** #

"""
    direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)

Returns the direct sum of two `QuantumSystem` objects.
"""
function QuantumUtils.direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)
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

apply_suffix(symb::Symbol, suffix::String) = Symbol(string(symb, suffix))
apply_suffix(symb::Tuple, suffix::String) = Tuple(apply_suffix(s, suffix) for s ∈ symb)
apply_suffix(symb::AbstractVector, suffix::String) = [apply_suffix(s, suffix) for s ∈ symb]
apply_suffix(d::Dict{Symbol, Any}, suffix::String) = typeof(d)(apply_suffix(k, suffix) => v for (k, v) ∈ d)

function apply_suffix(nt::NamedTuple, suffix::String)
    symbs = Tuple(apply_suffix(k, suffix) for k ∈ keys(nt))
    vals = [v for v ∈ values(nt)]
    return NamedTuple{symbs}(vals)
end

function get_components(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory)
    symbs = Tuple(c for c in components)
    vals = [traj[name] for name ∈ components]
    return NamedTuple{symbs}(vals)
end

function apply_suffix(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory, suffix::String)
    return apply_suffix(get_components(components, traj), suffix)
end

function apply_suffix(traj::NamedTrajectory, suffix::String)
    component_names = vcat(traj.state_names..., traj.control_names...)
    components = apply_suffix(component_names, traj, suffix)
    bounds = apply_suffix(traj.bounds, suffix)
    initial = apply_suffix(traj.initial, suffix)
    final = apply_suffix(traj.final, suffix)
    goal = apply_suffix(traj.goal, suffix)
    controls = apply_suffix(traj.control_names, suffix)
    if traj.timestep isa Symbol
        timestep = apply_suffix(traj.timestep, suffix)
    else
        timestep = traj.timestep
    end
    return NamedTrajectory(
        components;
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal
    )
end

function apply_suffix(
    integrator::UnitaryPadeIntegrator, 
    suffix::String
)   
    # TODO: Generalize to QuantumIntegrator?
    # Just need the matrices
    sys = QuantumSystem(
        QuantumSystems.H(integrator.G_drift), 
        QuantumSystems.H.(integrator.G_drives)
    )
    return UnitaryPadeIntegrator(
        sys,
        apply_suffix(integrator.unitary_symb, suffix),
        apply_suffix(integrator.drive_symb, suffix),
        order=integrator.order,
        autodiff=integrator.autodiff,
        G=integrator.G
    )
end

function apply_suffix(integrator::DerivativeIntegrator, suffix::String)
    return DerivativeIntegrator(
        apply_suffix(integrator.variable, suffix),
        apply_suffix(integrator.derivative, suffix),
        integrator.dim
    )
end

apply_suffix(integrators::AbstractVector{<:AbstractIntegrator}, suffix::String) =
    [apply_suffix(integrator, suffix) for integrator ∈ integrators]

function apply_suffix(sys::QuantumSystem, suffix::String)
    return QuantumSystem(
        sys.H_drift,
        sys.H_drives,
        params=apply_suffix(sys.params, suffix)
    )
end

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

function merge_outer(traj1::NamedTrajectory, traj2::NamedTrajectory)   
    # TODO: Free time problem
    if traj1.timestep isa Symbol || traj2.timestep isa Symbol
        throw(ArgumentError("Free time problems not supported"))
    end

    # TODO: Integrators can use different timesteps. What about NamedTrajectories?
    if traj1.timestep != traj2.timestep
        throw(ArgumentError("Timesteps must be equal"))
    end

    # collect component data
    component_names1 = vcat(traj1.state_names..., traj1.control_names...)
    component_names2 = vcat(traj2.state_names..., traj2.control_names...)
    components = merge_outer(
        get_components(component_names1, traj1),
        get_components(component_names2, traj2)
    )

    return NamedTrajectory(
        components,
        controls=merge_outer(traj1.control_names, traj2.control_names),
        timestep=traj1.timestep,
        bounds=merge_outer(traj1.bounds, traj2.bounds),
        initial=merge_outer(traj1.initial, traj2.initial),
        final=merge_outer(traj1.final, traj2.final),
        goal=merge_outer(traj1.goal, traj2.goal)
    )
end

function merge_outer(
    integrators1::AbstractVector{<:AbstractIntegrator},
    integrators2::AbstractVector{<:AbstractIntegrator}
)   
    return vcat(integrators1, integrators2)

end

# TODO: overload or new name?
Base.endswith(symb::Symbol, suffix::AbstractString) = endswith(String(symb), suffix)
Base.endswith(integrator::UnitaryPadeIntegrator, suffix::String) = endswith(integrator.unitary_symb, suffix)
Base.endswith(integrator::DerivativeIntegrator, suffix::String) = endswith(integrator.variable, suffix)
Base.startswith(symb::Symbol, prefix::AbstractString) = startswith(String(symb), prefix)
Base.startswith(symb::Symbol, prefix::Symbol) = startswith(String(symb), String(prefix))


function get_suffix(nt::NamedTuple, suffix::String)
    names = Tuple(k for (k, v) ∈ pairs(nt) if endswith(k, suffix))
    values = [v for (k, v) ∈ pairs(nt) if endswith(k, suffix)]
    return NamedTuple{names}(values)
end

function get_suffix(d::Dict{<:Symbol, <:Any}, suffix::String)
    return Dict(k => v for (k, v) ∈ d if endswith(k, suffix))
end

function get_suffix(traj::NamedTrajectory, suffix::String)
    state_names = Tuple(s for s ∈ traj.state_names if endswith(s, suffix))
    control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix))
    component_names = Tuple(vcat(state_names..., control_names...))
    components = NamedTuple{component_names}([traj[name] for name ∈ component_names])
    return NamedTrajectory(
        components,
        controls=control_names,
        timestep=traj.timestep,
        bounds=get_suffix(traj.bounds, suffix),
        initial=get_suffix(traj.initial, suffix),
        final=get_suffix(traj.final, suffix),
        goal=get_suffix(traj.goal, suffix)
    )
end

function get_suffix(integrators::AbstractVector{<:AbstractIntegrator}, suffix::String)
    return [deepcopy(integrator) for integrator ∈ integrators if endswith(integrator, suffix)]
end

function get_suffix(
    prob::QuantumControlProblem,
    suffix::String;
    unitary_prefix::Symbol=:Ũ⃗,
    kwargs...
)
    # Extract the trajectory
    traj = get_suffix(prob.trajectory, suffix)

    # Extract the integrators
    integrators = get_suffix(prob.integrators, suffix)
    
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
    J = NullObjective()

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators
    )
end

# *************************************************************************** #

@testitem "Apply suffix to trajectories" begin
    using NamedTrajectories
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    suffix = "_new"
    new_traj = ProblemTemplates.apply_suffix(traj, suffix)
    
    @test new_traj.state_names == ProblemTemplates.apply_suffix(traj.state_names, suffix)
    @test new_traj.control_names == ProblemTemplates.apply_suffix(traj.control_names, suffix)

    same_traj = ProblemTemplates.apply_suffix(traj, "")
    @test traj == same_traj
end

@testitem "Merge trajectories" begin
    using NamedTrajectories
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    
    # apply suffix
    pf_traj1 = ProblemTemplates.apply_suffix(traj, "_1")
    pf_traj2 = ProblemTemplates.apply_suffix(traj, "_2")

    # merge
    new_traj = ProblemTemplates.merge_outer(pf_traj1, pf_traj2)

    @test issetequal(new_traj.state_names, vcat(pf_traj1.state_names..., pf_traj2.state_names...))
    @test issetequal(new_traj.control_names, vcat(pf_traj1.control_names..., pf_traj2.control_names...))
end

@testitem "Merge systems" begin
    using NamedTrajectories
    include("../../test/test_utils.jl")

    H_drift = 0.01 * GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    T = 50
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T))
        
    # apply suffix and sum
    sys2 = direct_sum(
        ProblemTemplates.apply_suffix(sys, "_1"),
         ProblemTemplates.apply_suffix(sys, "_2")
    )

    @test length(sys2.H_drives) == 4
    @test sys2.params[:T_1] == T
    @test sys2.params[:T_2] == T

    # add another system
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T, :S=>2T))
    sys3 = direct_sum(sys2, ProblemTemplates.apply_suffix(sys, "_3"))
    @test length(sys3.H_drives) == 6
    @test sys3.params[:T_3] == T
    @test sys3.params[:S_3] == 2T
end

@testitem "Construct direct sum problem" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal1 = GATES[:X]
    U_ε = haar_identity(2, 0.33)
    U_goal2 = U_ε'GATES[:X]*U_ε
    T = 50
    Δt = 0.2
    ops = Options(print_level=1)
    prob1 = UnitarySmoothPulseProblem(sys, U_goal1, T, Δt, free_time=false, ipopt_options=ops)
    prob2 = UnitarySmoothPulseProblem(sys, U_goal2, T, Δt, free_time=false, ipopt_options=ops)

    # Test default
    direct_sum_prob1 = UnitaryDirectSumProblem([prob1, prob2], 0.99, ipopt_options=ops)
    state_names = vcat(
        ProblemTemplates.apply_suffix(prob1.trajectory.state_names, "1")...,
        ProblemTemplates.apply_suffix(prob2.trajectory.state_names, "2")...
    )
    control_names = vcat(
        ProblemTemplates.apply_suffix(prob1.trajectory.control_names, "1")...,
        ProblemTemplates.apply_suffix(prob2.trajectory.control_names, "2")...
    )
    @test issetequal(direct_sum_prob1.trajectory.state_names, state_names)
    @test issetequal(direct_sum_prob1.trajectory.control_names, control_names)

    # Test label graph
    direct_sum_prob2 = UnitaryDirectSumProblem(
        [prob1, prob2], 
        0.99, 
        prob_labels=["a", "b"],
        graph=[("a", "b")],
        ipopt_options=ops)
    state_names_ab = vcat(
        ProblemTemplates.apply_suffix(prob1.trajectory.state_names, "a")...,
        ProblemTemplates.apply_suffix(prob2.trajectory.state_names, "b")...
    )
    control_names_ab = vcat(
        ProblemTemplates.apply_suffix(prob1.trajectory.control_names, "a")...,
        ProblemTemplates.apply_suffix(prob2.trajectory.control_names, "b")...
    )
    @test issetequal(direct_sum_prob2.trajectory.state_names, state_names_ab)
    @test issetequal(direct_sum_prob2.trajectory.control_names, control_names_ab)

    # Test bad graph
    @test_throws ArgumentError UnitaryDirectSumProblem(
        [prob1, prob2], 
        0.99, 
        prob_labels=["a", "b"],
        graph=[("x", "b")],
        ipopt_options=ops
    )

    # Test symbol graph
    direct_sum_prob3 = UnitaryDirectSumProblem(
        [prob1, prob2], 
        0.99, 
        graph=[(:a1, :a2)],
        ipopt_options=ops
    )
    @test issetequal(direct_sum_prob3.trajectory.state_names, state_names)
    @test issetequal(direct_sum_prob3.trajectory.control_names, control_names)

    # Test triple
    direct_sum_prob4 = UnitaryDirectSumProblem([prob1, prob2, prob1], 0.99, ipopt_options=ops)
    state_names_triple = vcat(state_names..., ProblemTemplates.apply_suffix(prob1.trajectory.state_names, "3")...)
    control_names_triple = vcat(control_names..., ProblemTemplates.apply_suffix(prob1.trajectory.control_names, "3")...)
    @test issetequal(direct_sum_prob4.trajectory.state_names, state_names_triple)
    @test issetequal(direct_sum_prob4.trajectory.control_names, control_names_triple)

    # Test boundary values
    direct_sum_prob5 = UnitaryDirectSumProblem(
        [prob1, prob2], 
        0.99,
        graph=[("x", "1"), ("1", "2")],
        R_b=1e3,
        Q_symb=:dda,
        boundary_values=Dict("x"=>copy(prob1.trajectory[:dda])),
        ipopt_options=ops
    )
    # # TODO: Check for objectives?
    
end

@testitem "Get suffix" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    T = 50
    Δt = 0.2
    ops = Options(print_level=1)
    prob1 = UnitarySmoothPulseProblem(sys, GATES[:X], T, Δt, free_time=false, ipopt_options=ops)
    prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, free_time=false, ipopt_options=ops)
    
    # Direct sum problem with suffix extraction
    # Note: Turn off control reset
    direct_sum_prob = UnitaryDirectSumProblem([prob1, prob2], 0.99, drive_reset_ratio=0.0, ipopt_options=ops)
    prob1_got = ProblemTemplates.get_suffix(direct_sum_prob, "1")
    @test prob1_got.trajectory == ProblemTemplates.apply_suffix(prob1.trajectory, "1")
end
