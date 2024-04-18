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
            (append_suffix(Q_symb, i), append_suffix(Q_symb, j))
            for (i, j) ∈ zip(prob_labels[1:N-1], prob_labels[2:N])
        ]
    else
        # Check that String edge labels are in prob_labels or boundary, and make Symbols
        if eltype(eltype(graph)) == String
            graph_symbols = Tuple{Symbol, Symbol}[]
            for edge in graph
                if edge[1] in prob_labels && edge[2] in prob_labels
                    push!(graph_symbols, (append_suffix(Q_symb, edge[1]), append_suffix(Q_symb, edge[2])))
                elseif edge[1] in keys(boundary_values) && edge[2] in prob_labels
                    push!(boundary, (append_suffix(Q_symb, edge[2]), boundary_values[edge[1]]))
                elseif edge[1] in prob_labels && edge[2] in keys(boundary_values)
                    push!(boundary, (append_suffix(Q_symb, edge[1]), boundary_values[edge[2]]))
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
        direct_sum, 
        [append_suffix(p.trajectory, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # add noise to control data (avoid restoration)
    if drive_reset_ratio > 0
        σs = repeat([drive_derivative_σ], 2)
        for ℓ in prob_labels
            a_symb = append_suffix(:a, ℓ)
            da_symb = append_suffix(:da, ℓ)
            dda_symb = append_suffix(:dda, ℓ)
            a_bounds_lower, a_bounds_upper = traj.bounds[a_symb]
            a, da, dda = randomly_fill_drives(traj.T, a_bounds_lower, a_bounds_upper, σs)
            update!(traj, a_symb, (1 - drive_reset_ratio) * traj[a_symb] + drive_reset_ratio * a)
            update!(traj, da_symb, (1 - drive_reset_ratio) * traj[da_symb] + drive_reset_ratio * da)
            update!(traj, dda_symb, (1 - drive_reset_ratio) * traj[dda_symb] + drive_reset_ratio * dda)
        end
    end

    # concatenate suffix integrators
    integrators = vcat(
        [append_suffix(p.integrators, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]...
    )

    # direct sum (used for problem saving, only)
    system = reduce(
        direct_sum, 
        [append_suffix(p.system, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

    # Rebuild trajectory constraints
    build_trajectory_constraints = true
    constraints = AbstractConstraint[]

    # Add fidelity constraints for each problem
    for (p, ℓ) ∈ zip(probs, prob_labels)
        goal_symb, = keys(p.trajectory.goal)
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            append_suffix(goal_symb, ℓ),
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
        J += QuadraticRegularizer(append_suffix(:a, ℓ), traj, R_a)
        J += QuadraticRegularizer(append_suffix(:da, ℓ), traj, R_da)
        J += QuadraticRegularizer(append_suffix(:dda, ℓ), traj, R_dda)
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

# *******************************************************************************

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
        append_suffix(prob1.trajectory.state_names, "1")...,
        append_suffix(prob2.trajectory.state_names, "2")...
    )
    control_names = vcat(
        append_suffix(prob1.trajectory.control_names, "1")...,
        append_suffix(prob2.trajectory.control_names, "2")...
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
        append_suffix(prob1.trajectory.state_names, "a")...,
        append_suffix(prob2.trajectory.state_names, "b")...
    )
    control_names_ab = vcat(
        append_suffix(prob1.trajectory.control_names, "a")...,
        append_suffix(prob2.trajectory.control_names, "b")...
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
    state_names_triple = vcat(state_names..., append_suffix(prob1.trajectory.state_names, "3")...)
    control_names_triple = vcat(control_names..., append_suffix(prob1.trajectory.control_names, "3")...)
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
