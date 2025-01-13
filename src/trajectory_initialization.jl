module TrajectoryInitialization

export unitary_geodesic
export linear_interpolation
export unitary_linear_interpolation
export initialize_trajectory

using NamedTrajectories

using Distributions
using ExponentialAction
using LinearAlgebra
using TestItemRunner

using ..Isomorphisms
using ..QuantumSystems
using ..Rollouts
using ..EmbeddedOperators
using ..DirectSums


# ----------------------------------------------------------------------------- #
#                           Initial states                                      #
# ----------------------------------------------------------------------------- #

"""
    unitary_linear_interpolation(
        U_init::AbstractMatrix,
        U_goal::AbstractMatrix,
        samples::Int
    )

Compute a linear interpolation of unitary operators with `samples` samples.
"""
function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    Ũ⃗s = [Ũ⃗_init + (Ũ⃗_goal - Ũ⃗_init) * t for t ∈ range(0, 1, length=samples)]
    Ũ⃗ = hcat(Ũ⃗s...)
    return Ũ⃗
end

function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int
)
    return unitary_linear_interpolation(U_init, U_goal.operator, samples)
end

"""
    unitary_geodesic(
        operator::EmbeddedOperator,
        samples::Int;
        kwargs...
    )

    unitary_geodesic(
        U_goal::AbstractMatrix{<:Number},
        samples::Int;
        kwargs...
    )

    unitary_geodesic(
        U₀::AbstractMatrix{<:Number},
        U₁::AbstractMatrix{<:Number},
        samples::Number;
        kwargs...
    )

    unitary_geodesic(
        U₀::AbstractMatrix{<:Number},
        U₁::AbstractMatrix{<:Number},
        timesteps::AbstractVector{<:Number};
        return_generator=false
    )

Compute a geodesic connecting two unitary operators.
"""
function unitary_geodesic end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int;
    kwargs...
)
    U1 = unembed(U_init, U_goal)
    U2 = unembed(U_goal)
    Ũ⃗ = unitary_geodesic(U1, U2, samples; kwargs...)
    return hcat([
        operator_to_iso_vec(embed(iso_vec_to_operator(Ũ⃗ₜ), U_goal))
        for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    return unitary_geodesic(U_init, U_goal, range(0, 1, samples); kwargs...)
end

function unitary_geodesic(
    U_goal::AbstractPiccoloOperator,
    samples::Int;
    kwargs...
)
    if U_goal isa EmbeddedOperator
        U_goal = U_goal.operator
    end
    return unitary_geodesic(
        Matrix{ComplexF64}(I(size(U_goal, 1))),
        U_goal,
        samples;
        kwargs...
    )
end

"""
    unitary_geodesic(U_init, U_goal, times; kwargs...)

Compute the geodesic connecting U_init and U_goal at the specified times. Allows for the possibility of unequal times and ranges outside [0,1].

# Arguments
- `U_init::AbstractMatrix{<:Number}`: The initial unitary operator.
- `U_goal::AbstractMatrix{<:Number}`: The goal unitary operator.
- `times::AbstractVector{<:Number}`: The times at which to evaluate the geodesic.

# Keyword Arguments
- `return_unitary_isos::Bool=true`: If true returns a matrix where each column is a unitary isovec, i.e. vec(vcat(real(U), imag(U))). If false, returns a vector of unitary matrices.
- `return_generator::Bool=false`: If true, returns the effective Hamiltonian generating the geodesic.
"""
function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    times::AbstractVector{<:Number};
    return_unitary_isos=true,
    return_generator=false
)
    t₀ = times[1]
    T = times[end] - t₀
    H = im * log(U_goal * U_init') / T
    # -im prefactor is not included in H
    U_geo = [exp(-im * H * (t - t₀)) * U_init for t ∈ times]
    if !return_unitary_isos
        if return_generator
            return U_geo, H
        else
            return U_geo
        end
    else
        Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
        if return_generator
            return Ũ⃗_geo, H
        else
            return Ũ⃗_geo
        end
    end
end

linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) =
    hcat(range(x, y, n)...)

# ============================================================================= #

const VectorBound = Union{AbstractVector{R}, Tuple{AbstractVector{R}, AbstractVector{R}}} where R <: Real
const ScalarBound = Union{R, Tuple{R, R}} where R <: Real

function initialize_unitary_trajectory(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractPiccoloOperator,
    T::Int;
    geodesic=true
)
    if geodesic
        Ũ⃗ = unitary_geodesic(U_init, U_goal, T)
    else
        Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, T)
    end
    return Ũ⃗
end

# ----------------------------------------------------------------------------- #
#                           Initial controls                                    #
# ----------------------------------------------------------------------------- #

function initialize_control_trajectory(
    n_drives::Int,
    n_derivatives::Int,
    T::Int,
    bounds::VectorBound,
    drive_derivative_σ::Float64,
)
    if bounds isa AbstractVector
        a_dists = [Uniform(-bounds[i], bounds[i]) for i = 1:n_drives]
    elseif bounds isa Tuple
        a_dists = [Uniform(aᵢ_lb, aᵢ_ub) for (aᵢ_lb, aᵢ_ub) ∈ zip(bounds...)]
    else
        error("bounds must be a Vector or Tuple")
    end

    controls = Matrix{Float64}[]

    a = hcat([
        zeros(n_drives),
        vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
        zeros(n_drives)
    ]...)
    push!(controls, a)

    for _ in 1:n_derivatives
        push!(controls, randn(n_drives, T) * drive_derivative_σ)
    end

    return controls
end

function initialize_control_trajectory(
    a::AbstractMatrix,
    Δt::AbstractVecOrMat,
    n_derivatives::Int
)
    controls = Matrix{Float64}[a]

    for n in 1:n_derivatives
        # next derivative
        push!(controls,  derivative(controls[end], Δt))

        # to avoid constraint violation error at initial iteration for da, dda, ...
        if n > 1
            controls[end-1][:, end] =
                controls[end-1][:, end-1] + Δt[end-1] * controls[end][:, end-1]
        end
    end
    return controls
end

initialize_control_trajectory(a::AbstractMatrix, Δt::Real, n_derivatives::Int) =
    initialize_control_trajectory(a, fill(Δt, size(a, 2)), n_derivatives)

# ----------------------------------------------------------------------------- #
#                           Trajectory initialization                           #
# ----------------------------------------------------------------------------- #

"""
    initialize_trajectory


Initialize a trajectory for a control problem. The trajectory is initialized with
data that should be consistently the same type (in this case, Float64).

"""
function initialize_trajectory(
    state_data::Vector{<:AbstractMatrix{Float64}},
    state_inits::Vector{<:AbstractVector{Float64}},
    state_goals::Vector{<:AbstractVector{Float64}},
    state_names::AbstractVector{Symbol},
    T::Int,
    Δt::Union{Float64, AbstractVecOrMat{<:Float64}},
    n_drives::Int,
    control_bounds::Tuple{Vararg{VectorBound}};
    bound_state=false,
    free_time=false,
    control_name=:a,
    n_control_derivatives::Int=length(control_bounds) - 1,
    timestep_name=:Δt,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    phase_name::Symbol=:ϕ,
    phase_data::Union{AbstractVector{<:Real}, Nothing}=nothing,
    verbose=false,
)
    @assert length(state_data) == length(state_names) == length(state_inits) == length(state_goals) "state_data, state_names, state_inits, and state_goals must have the same length"
    @assert length(control_bounds) == n_control_derivatives + 1 "control_bounds must have $n_control_derivatives + 1 elements"

    # assert that state names are unique
    @assert length(state_names) == length(Set(state_names)) "state_names must be unique"

    # Control data
    control_derivative_names = [
        Symbol("d"^i * string(control_name)) for i = 1:n_control_derivatives
    ]
    if verbose
        println("control_derivative_names: $control_derivative_names")
    end

    control_names = (control_name, control_derivative_names...)

    control_bounds = NamedTuple{control_names}(control_bounds)

    # Timestep data
    if free_time
        if Δt isa Real
            Δt = fill(Δt, 1, T)
        elseif Δt isa AbstractVector
            Δt = reshape(Δt, 1, :)
        else
            @assert size(Δt) == (1, T) "Δt must be a Real, AbstractVector, or 1x$(T) AbstractMatrix"
        end
        timestep = timestep_name
    else
        @assert Δt isa Real "Δt must be a Real if free_time is false"
        timestep = Δt
    end

    # Constraints
    initial = (;
        (state_names .=> state_inits)...,
        control_name => zeros(n_drives),
    )

    final = (;
        control_name => zeros(n_drives),
    )

    goal = (; (state_names .=> state_goals)...)

    # Bounds
    bounds = control_bounds

    # Put unit box bounds on the state if bound_state is true
    if bound_state
        state_dim = length(state_inits[1])
        state_bounds = repeat([(-ones(state_dim), ones(state_dim))], length(state_names))
        bounds = merge(bounds, (; (state_names .=> state_bounds)...))
    end

    # Trajectory
    if isnothing(a_guess)
        # Randomly sample controls
        a_values = initialize_control_trajectory(
            n_drives,
            n_control_derivatives,
            T,
            bounds[control_name],
            drive_derivative_σ
        )
    else
        # Use provided controls and take derivatives
        a_values = initialize_control_trajectory(a_guess, Δt, n_control_derivatives)
    end

    names = [state_names..., control_names...]
    values = [state_data..., a_values...]

    if free_time
        push!(names, timestep_name)
        push!(values, Δt)
        controls = (control_names[end], timestep_name)
        bounds = merge(bounds, (; timestep_name => Δt_bounds,))
    else
        controls = (control_names[end],)
    end

    # Construct global data for free phases
    global_data = isnothing(phase_data) ? (;) : (; phase_name => phase_data)

    return NamedTrajectory(
        (; (names .=> values)...);
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal,
        global_data=global_data
    )
end
    
"""
    initialize_trajectory

Trajectory initialization of unitaries.
"""
function initialize_trajectory(
    U_goal::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Real, AbstractVecOrMat{<:Real}},
    args...;
    state_name::Symbol=:Ũ⃗,
    U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(U_goal, 1))),
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, Nothing}=nothing,
    rollout_integrator::Function=expv,
    geodesic=true,
    phase_operators::Union{AbstractVector{<:AbstractMatrix}, Nothing}=nothing,
    kwargs...
)
    # Construct timesteps
    if Δt isa AbstractMatrix
        timesteps = vec(Δt)
    elseif Δt isa Float64
        timesteps = fill(Δt, T)
    else
        timesteps = Δt
    end

    # Initial state and goal
    Ũ⃗_init = operator_to_iso_vec(U_init)

    if U_goal isa EmbeddedOperator
        Ũ⃗_goal = operator_to_iso_vec(U_goal.operator)
    else
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
    end
    
    # Construct state data
    if isnothing(a_guess)
        Ũ⃗_traj = initialize_unitary_trajectory(U_init, U_goal, T; geodesic=geodesic)
    else
        @assert !isnothing(system) "System must be provided if a_guess is provided."
        Ũ⃗_traj = unitary_rollout(Ũ⃗_init, a_guess, timesteps, system; integrator=rollout_integrator)
    end

    # Construct phase data
    phase_data = isnothing(phase_operators) ? nothing : π * randn(length(phase_operators))

    return initialize_trajectory(
        [Ũ⃗_traj],
        [Ũ⃗_init],
        [Ũ⃗_goal],
        [state_name],
        T,
        Δt,
        args...;
        phase_data=phase_data,
        a_guess=a_guess,
        kwargs...
    )
end

"""
    initialize_trajectory
    
Trajectory initialization of quantum states.
"""
function initialize_trajectory(
    ψ_goals::AbstractVector{<:AbstractVector{ComplexF64}},
    ψ_inits::AbstractVector{<:AbstractVector{ComplexF64}},
    T::Int,
    Δt::Union{Real, AbstractVector{<:Real}},
    args...;
    state_name=:ψ̃,
    state_names::AbstractVector{<:Symbol}=length(ψ_goals) == 1 ?
        [state_name] :
        [Symbol(string(state_name) * "$i") for i = 1:length(ψ_goals)],
    a_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, Nothing}=nothing,
    rollout_integrator::Function=expv,
    kwargs...
)
    @assert length(ψ_inits) == length(ψ_goals) "ψ_inits and ψ_goals must have the same length"
    @assert length(state_names) == length(ψ_goals) "state_names and ψ_goals must have the same length"

    ψ̃_goals = ket_to_iso.(ψ_goals)
    ψ̃_inits = ket_to_iso.(ψ_inits)

    # Construct timesteps
    if Δt isa AbstractMatrix
        timesteps = vec(Δt)
    elseif Δt isa Float64
        timesteps = fill(Δt, T)
    else
        timesteps = Δt
    end

    # Construct state data
    ψ̃_trajs = Matrix{Float64}[]
    if isnothing(a_guess)
        for (ψ̃_init, ψ̃_goal) ∈ zip(ψ̃_inits, ψ̃_goals)
            ψ̃_traj = linear_interpolation(ψ̃_init, ψ̃_goal, T)
            push!(ψ̃_trajs, ψ̃_traj)
        end
        if system isa AbstractVector
            ψ̃_trajs = repeat(ψ̃_trajs, length(system))
        end
    else
        for ψ̃_init ∈ ψ̃_inits
            ψ̃_traj = rollout(ψ̃_init, a_guess, timesteps, system; integrator=rollout_integrator)
            push!(ψ̃_trajs, ψ̃_traj)
        end
    end

    return initialize_trajectory(
        ψ̃_trajs,
        ψ̃_inits,
        ψ̃_goals,
        state_names,
        T,
        Δt,
        args...;
        a_guess=a_guess,
        kwargs...
    )
end


# ============================================================================= #

@testitem "Random drive initialization" begin
    T = 10
    n_drives = 2
    n_derivates = 2
    drive_bounds = [1.0, 2.0]
    drive_derivative_σ = 0.01

    a, da, dda = TrajectoryInitialization.initialize_control_trajectory(n_drives, n_derivates, T, drive_bounds, drive_derivative_σ)

    @test size(a) == (n_drives, T)
    @test size(da) == (n_drives, T)
    @test size(dda) == (n_drives, T)
    @test all([-drive_bounds[i] < minimum(a[i, :]) < drive_bounds[i] for i in 1:n_drives])
end

@testitem "Geodesic" begin
    using LinearAlgebra

    ## Group 1: identity to X (π rotation)

    # Test π rotation
    U_α = GATES[:I]
    U_ω = GATES[:X]
    Us, H = unitary_geodesic(
        U_α, U_ω, range(0, 1, 4), return_generator=true
    )

    @test size(Us, 2) == 4
    @test Us[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H' - H ≈ zeros(2, 2)
    @test norm(H) ≈ π

    # Test modified timesteps (10x)
    Us10, H10 = unitary_geodesic(
        U_α, U_ω, range(-5, 5, 4), return_generator=true
    )

    @test size(Us10, 2) == 4
    @test Us10[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us10[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H10' - H10 ≈ zeros(2, 2)
    @test norm(H10) ≈ π/10

    # Test wrapped call
    Us_wrap, H_wrap = unitary_geodesic(U_ω, 10, return_generator=true)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)
    rotation = [exp(-im * H_wrap * t) for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us_wrap, Us_test)


    ## Group 2: √X to X (π/2 rotation)

    # Test geodesic not at identity
    U₀ = sqrt(GATES[:X])
    U₁ = GATES[:X]
    Us, H = unitary_geodesic(U₀, U₁, 10, return_generator=true)
    @test Us[:, 1] ≈ operator_to_iso_vec(U₀)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)

    rotation = [exp(-im * H * t) * U₀ for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us, Us_test)
    Us_wrap = unitary_geodesic(U_ω, 4)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)

end

@testitem "unitary trajectory initialization" begin
    using NamedTrajectories
    U_goal = GATES[:X]
    T = 10
    Δt = 0.1
    n_drives = 2
    a_bounds = ([1.0, 1.0],)

    traj = initialize_trajectory(
        U_goal, T, Δt, n_drives, a_bounds
    )

    @test traj isa NamedTrajectory
end

@testitem "quantum state trajectory initialization" begin
    using NamedTrajectories

    ψ_init = Vector{ComplexF64}([0.0, 1.0])
    ψ_goal = Vector{ComplexF64}([1.0, 0.0])

    T = 10
    Δt = 0.1
    n_drives = 2
    all_a_bounds = ([1.0, 1.0],)

    traj = initialize_trajectory(
        [ψ_goal], [ψ_init], T, Δt, n_drives, all_a_bounds
    )

    @test traj isa NamedTrajectory
end


end
