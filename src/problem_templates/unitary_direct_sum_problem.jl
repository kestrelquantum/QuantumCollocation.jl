@doc """
    UnitaryDirectSumProblem(probs, final_fidelity; kwargs...)

Construct a `QuantumControlProblem` as a direct sum of unitary gate problems. The 
purpose is to find solutions that are as close as possible with respect to one of 
their components. In particular, this is useful for finding interpolatable control solutions.

A graph of edges (specified by problem labels) will enforce a `PairwiseQuadraticRegularizer` between
the component trajectories of the problem in `probs` corresponding to the names of the edge in `edges`
with corresponding edge weight `Q`.

The default behavior is to use a 1D chain for the graph, i.e., enforce a `PairwiseQuadraticRegularizer`
between each neighbor of the provided `probs`.

# Arguments
- `probs::AbstractVector{<:QuantumControlProblem}`: the problems to combine
- `final_fidelity::Real`: the fidelity to enforce between the component final unitaries and the component goal unitaries

# Keyword Arguments
- `prob_labels::AbstractVector{<:String}}`: the labels for the problems
-  graph::Union{Nothing, AbstractVector{<:Tuple{String, String}}, AbstractVector{<:Tuple{Symbol, Symbol}}}`: the graph of edges to enforce
- `Q::Union{Float64, Vector{Float64}}=100.0`: the weights on the pairwise regularizers
- `Q_symb::Symbol=:Ũ⃗`: the symbol to use for the regularizer
- `R::Float64=1e-2`: the shared weight on all control terms (:a, :da, :dda is assumed)
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
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
    Q::Union{Float64, Vector{Float64}}=100.0,
    Q_symb::Symbol=:Ũ⃗,
    R::Float64=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
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

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    # Default chain graph 
    if isnothing(graph)
        graph = [
            (apply_postfix(Q_symb, i), apply_postfix(Q_symb, j))
            for (i, j) ∈ zip(prob_labels[1:N-1], prob_labels[2:N])
        ]
    else
        # Check that String edge labels are in prob_labels, and make Symbols
        if eltype(eltype(graph)) == String
            graph_symb = Tuple{Symbol, Symbol}[]
            for edge in graph
                if !(edge[1] in prob_labels && edge[2] in prob_labels)
                    throw(ArgumentError("Edge labels must be in prob_labels"))
                end
                push!(graph_symb, (apply_postfix(Q_symb, edge[1]), apply_postfix(Q_symb, edge[2])))
            end
            graph = graph_symb
        end
    end

    # Build the direct sum system

    # merge postfix trajectories
    traj = reduce(
        merge_outer, 
        [apply_postfix(p.trajectory, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # add noise to control data (avoid restoration)
    if drive_reset_ratio > 0
        σs = repeat([drive_derivative_σ], 2)
        for ℓ in prob_labels
            a_symb = apply_postfix(:a, ℓ)
            da_symb = apply_postfix(:da, ℓ)
            dda_symb = apply_postfix(:dda, ℓ)
            a_bounds_lower, a_bounds_upper = traj.bounds[a_symb]
            a, da, dda = randomly_fill_drives(traj.T, a_bounds_lower, a_bounds_upper, σs)
            update!(traj, a_symb, (1 - drive_reset_ratio) * traj[a_symb] + drive_reset_ratio * a)
            update!(traj, da_symb, (1 - drive_reset_ratio) * traj[da_symb] + drive_reset_ratio * da)
            update!(traj, dda_symb, (1 - drive_reset_ratio) * traj[dda_symb] + drive_reset_ratio * dda)
        end
    end

    # concatenate postfix integrators
    integrators = vcat(
        [apply_postfix(p.integrators, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]...
    )

    # direct sum (used for problem saving, only)
    system = reduce(
        direct_sum, 
        [apply_postfix(p.system, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # Rebuild trajectory constraints
    build_trajectory_constraints = true
    constraints = AbstractConstraint[]

    # Add fidelity constraints for each problem
    for (p, ℓ) ∈ zip(probs, prob_labels)
        goal_symb, = keys(p.trajectory.goal)
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            apply_postfix(goal_symb, ℓ),
            final_fidelity,
            traj;
            subspace=subspace
        )
        push!(constraints, fidelity_constraint)
    end

    # Build the objective function
    J = PairwiseQuadraticRegularizer(traj, Q, graph)

    for (p, ℓ) ∈ zip(probs, prob_labels)
        J += QuadraticRegularizer(apply_postfix(:a, ℓ), traj, R_a)
        J += QuadraticRegularizer(apply_postfix(:da, ℓ), traj, R_da)
        J += QuadraticRegularizer(apply_postfix(:dda, ℓ), traj, R_dda)
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

apply_postfix(symb::Symbol, postfix::String) = Symbol(string(symb, postfix))
apply_postfix(symb::Tuple, postfix::String) = Tuple(apply_postfix(s, postfix) for s ∈ symb)
apply_postfix(symb::AbstractVector, postfix::String) = [apply_postfix(s, postfix) for s ∈ symb]
apply_postfix(d::Dict{Symbol, Any}, postfix::String) = typeof(d)(apply_postfix(k, postfix) => v for (k, v) ∈ d)

function apply_postfix(nt::NamedTuple, postfix::String)
    symbs = Tuple(apply_postfix(k, postfix) for k ∈ keys(nt))
    vals = [v for v ∈ values(nt)]
    return NamedTuple{symbs}(vals)
end

function get_components(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory)
    symbs = Tuple(c for c in components)
    vals = [traj[name] for name ∈ components]
    return NamedTuple{symbs}(vals)
end

function apply_postfix(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory, postfix::String)
    return apply_postfix(get_components(components, traj), postfix)
end

function apply_postfix(traj::NamedTrajectory, postfix::String)
    component_names = vcat(traj.state_names..., traj.control_names...)
    components = apply_postfix(component_names, traj, postfix)
    bounds = apply_postfix(traj.bounds, postfix)
    initial = apply_postfix(traj.initial, postfix)
    final = apply_postfix(traj.final, postfix)
    goal = apply_postfix(traj.goal, postfix)
    controls = apply_postfix(traj.control_names, postfix)
    if traj.timestep isa Symbol
        timestep = apply_postfix(traj.timestep, postfix)
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

function apply_postfix(
    integrator::UnitaryPadeIntegrator, 
    postfix::String
)
    # Just need the matrices
    sys = QuantumSystem(
        QuantumSystems.H(integrator.G_drift), 
        QuantumSystems.H.(integrator.G_drives)
    )
    return UnitaryPadeIntegrator(
        sys,
        apply_postfix(integrator.unitary_symb, postfix),
        apply_postfix(integrator.drive_symb, postfix),
        order=integrator.order,
        autodiff=integrator.autodiff,
        G=integrator.G
    )
end

function apply_postfix(integrator::DerivativeIntegrator, postfix::String)
    return DerivativeIntegrator(
        apply_postfix(integrator.variable, postfix),
        apply_postfix(integrator.derivative, postfix),
        integrator.dim
    )
end

apply_postfix(integrators::AbstractVector{<:AbstractIntegrator}, postfix::String) =
    [apply_postfix(integrator, postfix) for integrator ∈ integrators]

function apply_postfix(sys::QuantumSystem, postfix::String)
    return QuantumSystem(
        sys.H_drift,
        sys.H_drives,
        params=apply_postfix(sys.params, postfix)
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

# *************************************************************************** #

@testitem "Apply postfix to trajectories" begin
    using NamedTrajectories
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    postfix = "_new"
    new_traj = ProblemTemplates.apply_postfix(traj, postfix)
    
    @test new_traj.state_names == ProblemTemplates.apply_postfix(traj.state_names, postfix)
    @test new_traj.control_names == ProblemTemplates.apply_postfix(traj.control_names, postfix)

    same_traj = ProblemTemplates.apply_postfix(traj, "")
    @test traj == same_traj
end

@testitem "Merge trajectories" begin
    using NamedTrajectories
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    
    # apply postfix
    pf_traj1 = ProblemTemplates.apply_postfix(traj, "_1")
    pf_traj2 = ProblemTemplates.apply_postfix(traj, "_2")

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
        
    # apply postfix and sum
    sys2 = direct_sum(
        ProblemTemplates.apply_postfix(sys, "_1"),
         ProblemTemplates.apply_postfix(sys, "_2")
    )

    @test length(sys2.H_drives) == 4
    @test sys2.params[:T_1] == T
    @test sys2.params[:T_2] == T

    # add another system
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T, :S=>2T))
    sys3 = direct_sum(sys2, ProblemTemplates.apply_postfix(sys, "_3"))
    @test length(sys3.H_drives) == 6
    @test sys3.params[:T_3] == T
    @test sys3.params[:S_3] == 2T
end

@testitem "Construct problem" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal1 = GATES[:X]
    U_ε = haar_identity(2, 0.33)
    U_goal2 = U_ε'U_goal1*U_ε
    T = 50
    Δt = 0.2
    ops = Options(recalc_y="yes", recalc_y_feas_tol=0.01, print_level=1)
    prob1 = UnitarySmoothPulseProblem(sys, U_goal1, T, Δt, free_time=false, ipopt_options=ops)
    prob2 = UnitarySmoothPulseProblem(sys, U_goal2, T, Δt, free_time=false, ipopt_options=ops)

    # Test default
    direct_sum_prob1 = UnitaryDirectSumProblem([prob1, prob2], 0.99, ipopt_options=ops)
    state_names = vcat(
        ProblemTemplates.apply_postfix(prob1.trajectory.state_names, "1")...,
        ProblemTemplates.apply_postfix(prob2.trajectory.state_names, "2")...
    )
    control_names = vcat(
        ProblemTemplates.apply_postfix(prob1.trajectory.control_names, "1")...,
        ProblemTemplates.apply_postfix(prob2.trajectory.control_names, "2")...
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
        ProblemTemplates.apply_postfix(prob1.trajectory.state_names, "a")...,
        ProblemTemplates.apply_postfix(prob2.trajectory.state_names, "b")...
    )
    control_names_ab = vcat(
        ProblemTemplates.apply_postfix(prob1.trajectory.control_names, "a")...,
        ProblemTemplates.apply_postfix(prob2.trajectory.control_names, "b")...
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
    state_names_triple = vcat(state_names..., ProblemTemplates.apply_postfix(prob1.trajectory.state_names, "3")...)
    control_names_triple = vcat(control_names..., ProblemTemplates.apply_postfix(prob1.trajectory.control_names, "3")...)
    @test issetequal(direct_sum_prob4.trajectory.state_names, state_names_triple)
    @test issetequal(direct_sum_prob4.trajectory.control_names, control_names_triple)
end