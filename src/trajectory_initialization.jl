module TrajectoryInitialization

export unitary_linear_interpolation
export unitary_geodesic

using LinearAlgebra

using ..QuantumUtils
using ..EmbeddedOperators

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
    operator::EmbeddedOperator,
    samples::Int;
    kwargs...
)
    U_goal = unembed(operator)
    U_init = Matrix{ComplexF64}(I(size(U_goal, 1)))
    Ũ⃗ = unitary_geodesic(U_init, U_goal, samples; kwargs...)
    return hcat([
        operator_to_iso_vec(EmbeddedOperators.embed(iso_vec_to_operator(Ũ⃗ₜ), operator))
            for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    N = size(U_goal, 1)
    U₀ = Matrix{ComplexF64}(I(N))
    return unitary_geodesic(U₀, U_goal, samples; kwargs...)
end

function unitary_geodesic(
    U₀::AbstractMatrix{<:Number},
    U₁::AbstractMatrix{<:Number},
    samples::Number;
    kwargs...
)
    return unitary_geodesic(U₀, U₁, range(0, 1, samples); kwargs...)
end

function unitary_geodesic(
    U₀::AbstractMatrix{<:Number},
    U₁::AbstractMatrix{<:Number},
    timesteps::AbstractVector{<:Number};
    return_generator=false
)
    """
    Compute the effective generator of the geodesic connecting U₀ and U₁.
        U₁ = exp(-im * H * T) U₀
        log(U₁ * U₀') = -im * H * T

    Allow for the possibiltiy of unequal timesteps and ranges outside [0,1].

    Returns the geodesic.
    Optionally returns the effective Hamiltonian generating the geodesic.
    """
    t₀ = timesteps[1]
    T = timesteps[end] - t₀
    H = im * log(U₁ * U₀') / T
    # -im prefactor is not included in H
    U_geo = [exp(-im * H * (t - t₀)) * U₀ for t ∈ timesteps]
    Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
    if return_generator
        return Ũ⃗_geo, H
    else
        return Ũ⃗_geo
    end
end

end
