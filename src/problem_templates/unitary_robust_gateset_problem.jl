@doc """
    UnitaryRobustGatesetProblem(
        probs::AbstractVector{<:QuantumControlProblem},
        final_fidelity::Real;
        prob_labels::AbstractVector{<:String}=[string(i) for i ∈ 1:length(probs)],
        crosstalk_graph::Union{
            AbstractVector{<:Tuple{String, String}}, 
            AbstractVector{<:Tuple{Symbol, Symbol}}
        }=Tuple{Symbol, Symbol}[],
        crosstalk_operators::Union{
            Tuple{<:AbstractMatrix, <:AbstractMatrix}, 
            AbstractVector{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}
        }=Matrix{ComplexF64}[],
        local_operators::Union{AbstractDict{<:String, <:AbstractArray}, AbstractDict{<:Symbol, <:AbstractArray}}=Dict{String, Array}(),
        Q_symb::Symbol=:Ũ⃗,
        R::Float64=1e-2,
        R_a::Union{Float64, Vector{Float64}}=R,
        R_da::Union{Float64, Vector{Float64}}=R,
        R_dda::Union{Float64, Vector{Float64}}=R,
        max_iter::Int=1000,
        linear_solver::String="mumps",
        verbose::Bool=false,
        jacobian_structure=true,
        blas_multithreading=true,
        ipopt_options=Options(),
        kwargs...
    )

    Construct a robust gateset problem from a set of quantum control problems, `probs`. The objective function is the sum of the
    infidelity robustness objectives with respect to the `crosstalk_operators` for each pair of problems in the `crosstalk_graph`, 
    as well as any `local_operators` on individual problems. The `crosstalk_operators` are provided as a pair of operators, for
    example (Z, Z) for ZZ crosstalk. See `UnitaryRobustnessProblem` for more details on this type of objective function.
    
    The fidelity constraints are the final unitary fidelity constraints for each problem in the set.

    # Arguments
    - `probs::AbstractVector{<:QuantumControlProblem}`: A list of quantum control problems.
    - `final_fidelity::Real`: The shared final fidelity for each problem in the set.
    - `prob_labels::AbstractVector{<:String}`: The labels for each problem in the set.
    - `crosstalk_graph::Union{AbstractVector{<:Tuple{String, String}}, AbstractVector{<:Tuple{Symbol, Symbol}}}`: 
        A list of edges between problems in the set.
    - `crosstalk_operators::Union{Tuple{<:AbstractMatrix, <:AbstractMatrix}, AbstractVector{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}}`:
        A list of crosstalk operators (pairs) for each edge in the `crosstalk_graph`.
    - `local_operators::Union{AbstractDict{<:String, <:AbstractArray}, AbstractDict{<:Symbol, <:AbstractArray}}`:
        A dictionary of local operators for each problem in the set.
    - `Q_symb::Symbol`: The symbol for the unitary trajectory.
    - `Q::Float64`: The regularization parameter for the infidelity cost, if using.
    - `R::Float64`: The regularization parameter for the quadratic regularizer.
    - `R_a::Union{Float64, Vector{Float64}}`: The regularization parameter for the quadratic regularizer on the amplitudes.
    - `R_da::Union{Float64, Vector{Float64}}`: The regularization parameter for the quadratic regularizer on the derivatives of the amplitudes.
    - `R_dda::Union{Float64, Vector{Float64}}`: The regularization parameter for the quadratic regularizer on the second derivatives of the amplitudes.
    - `fidelity_cost::Bool`: Whether to include the fidelity cost in the objective function.
    - `max_iter::Int`: The maximum number of iterations for the optimization.
    - `linear_solver::String`: The linear solver to use for the optimization.
    - `verbose::Bool`: Whether to print verbose output.
    - `jacobian_structure::Bool`: Whether to use the jacobian structure.
    - `blas_multithreading::Bool`: Whether to use BLAS multithreading.
    - `ipopt_options`: The options for the Ipopt solver.
    - `kwargs...`: Additional keyword arguments.
"""
function UnitaryRobustGatesetProblem(
    probs::AbstractVector{<:QuantumControlProblem},
    final_fidelity::Real;
    prob_labels::AbstractVector{<:String}=[string(i) for i ∈ 1:length(probs)],
    crosstalk_graph::Union{
        AbstractVector{<:Tuple{String, String}}, 
        AbstractVector{<:Tuple{Symbol, Symbol}}}=Tuple{Symbol, Symbol}[],
    crosstalk_operators::Union{
        Tuple{<:AbstractMatrix, <:AbstractMatrix}, 
        AbstractVector{<:Tuple{<:AbstractMatrix, <:AbstractMatrix}}}=Matrix{ComplexF64}[],
    local_operators::Union{AbstractDict{<:String, <:AbstractArray}, AbstractDict{<:Symbol, <:AbstractArray}}=Dict{String, Array}(),
    Q_symb::Symbol=:Ũ⃗,
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    fidelity_cost::Bool=false,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    verbose::Bool=false,
    jacobian_structure=true,
    blas_multithreading=true,
    ipopt_options=Options(),
    kwargs...
)
    # TODO: Need to implement subspace operators

    # Must use this option
    hessian_approximation=true

    N = length(probs)
    if length(prob_labels) != N
        throw(ArgumentError("Length of prob_labels must match length of probs"))
    end

    if N < 2
        throw(ArgumentError("At least two problems are required"))
    end

    # Broadcast single operator to all edges
    if eltype(crosstalk_operators) <: AbstractMatrix
        crosstalk_operators = repeat([crosstalk_operators], length(crosstalk_graph))
    else
        if length(crosstalk_graph) != length(crosstalk_operators)
            throw(ArgumentError("Length of crosstalk_graph must match length of crosstalk_operators"))
        end
    end

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    # Check that String edge labels are in prob_labels or boundary, and make Symbols
    if eltype(eltype(crosstalk_graph)) == String
        graph_symbols = Tuple{Symbol, Symbol}[]
        for (e1, e2) ∈ crosstalk_graph
            if e1 ∈ prob_labels && e2 ∈ prob_labels
                push!(graph_symbols, (add_suffix(Q_symb, e1), add_suffix(Q_symb, e2)))
            else
                throw(ArgumentError("Edge labels must be in prob_labels"))
            end
        end
        crosstalk_graph = graph_symbols
    end

    # Build the direct sum system

    # merge suffix trajectories
    traj = direct_sum([add_suffix(p.trajectory, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)])

    # concatenate suffix integrators
    integrators = vcat(
        [add_suffix(p.integrators, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]...
    )

    # direct sum (used for problem saving, only)
    system = direct_sum([add_suffix(p.system, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)])

    # Rebuild trajectory constraints
    build_trajectory_constraints = true
    constraints = AbstractConstraint[]

    # Add fidelity constraints for each problem
    for (p, ℓ) ∈ zip(probs, prob_labels)
        goal_symb, = keys(p.trajectory.goal)
        fidelity_constraint = FinalUnitaryFidelityConstraint(
            add_suffix(goal_symb, ℓ),
            final_fidelity,
            traj,
            hessian=!hessian_approximation
        )
        push!(constraints, fidelity_constraint)
    end

    # Build the objective function
    J = NullObjective()
    for ((s1, s2), (H1, H2)) ∈ zip(crosstalk_graph, crosstalk_operators)
        J += InfidelityRobustnessObjective(H1, H2, s1, s2)
    end

    # Add (optional) fidelity cost
    if fidelity_cost
        for ℓ ∈ prob_labels
            J += UnitaryInfidelityObjective(
                add_suffix(:Ũ⃗, ℓ), traj, Q,
                # subspace=subspace, 
                eval_hessian=!hessian_approximation
            )
        end
    end

    for (ℓ, H) ∈ local_operators
        J += InfidelityRobustnessObjective(H, traj, state_symb=add_suffix(Q_symb, ℓ))
    end

    for ℓ ∈ prob_labels
        J += QuadraticRegularizer(add_suffix(:a, ℓ), traj, R_a)
        J += QuadraticRegularizer(add_suffix(:da, ℓ), traj, R_da)
        J += QuadraticRegularizer(add_suffix(:dda, ℓ), traj, R_dda)
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

@testitem "Construct gateset robustness problem" begin
    sys = QuantumSystem([GATES[:X] / 2])
    U_goal = GATES[:X]
    H_crosstalk = (GATES[:Z], GATES[:Z])
    T = 50
    Δt = 0.2
    
    prob1 = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt, a_guess=π / T * ones(1, T),
        free_time=false, ipopt_options=Options(print_level=1)
    )
    prob2 = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt, a_guess=3π / T * ones(1, T),
        free_time=false, ipopt_options=Options(print_level=1)
    )
    
    solve!(prob1, max_iter=5)
    solve!(prob2, max_iter=5)
    
    # Join the two problems
    prob = UnitaryRobustGatesetProblem(
        [prob1, prob2], 
        1-1e-5, 
        crosstalk_graph=[("1", "2")], 
        crosstalk_operators=H_crosstalk,
        ipopt_options=Options(print_level=1)
    )
    
    solve!(prob; max_iter=10)
end
