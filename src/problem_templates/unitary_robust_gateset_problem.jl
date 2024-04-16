@doc """

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
                push!(graph_symbols, (apply_suffix(Q_symb, e1), apply_suffix(Q_symb, e2)))
            else
                throw(ArgumentError("Edge labels must be in prob_labels"))
            end
        end
        crosstalk_graph = graph_symbols
    end

    # Build the direct sum system

    # merge suffix trajectories
    traj = reduce(
        merge_outer, 
        [apply_suffix(p.trajectory, ℓ) for (p, ℓ) ∈ zip(probs, prob_labels)]
    )

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
            traj
        )
        push!(constraints, fidelity_constraint)
    end

    # Build the objective function
    J = NullObjective()
    for ((s1, s2), (H1, H2)) ∈ zip(crosstalk_graph, crosstalk_operators)
        J += InfidelityRobustnessObjective(H1, H2, s1, s2)
    end

    for (ℓ, H) ∈ local_operators
        J += InfidelityRobustnessObjective(H, traj, state_symb=apply_suffix(Q_symb, ℓ))
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

@testitem "Construct gateset robustness problem" begin
    sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal1 = GATES[:X]
    U_goal2 = GATES[:X]
    H_crosstalk = GATES[:Z] ⊗ GATES[:Z]
    T = 50
    Δt = 0.2
    ops = Options(print_level=1)
    prob1 = UnitarySmoothPulseProblem(sys, U_goal1, T, Δt, free_time=false, ipopt_options=ops)
    prob2 = UnitarySmoothPulseProblem(sys, U_goal2, T, Δt, free_time=false, ipopt_options=ops)

    prob = UnitaryRobustGatesetProblem(
        [prob1, prob2], 
        1e-3, 
        crosstalk_graph=[("1", "2")], 
        crosstalk_operators=H_crosstalk
    )

    solve!(prob; max_iter=10)
end