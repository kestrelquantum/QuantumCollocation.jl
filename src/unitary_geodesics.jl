module UnitaryGeodesics

export unitary_geodesic

using LinearAlgebra

using ..QuantumUtils
using ..EmbeddedOperators

function skew_symmetric(v::AbstractVector, n::Int)
    M = zeros(eltype(v), n, n)
    k = 1
    for j = 1:n
        for i = 1:j-1
            vᵢⱼ = v[k]
            M[i, j] = vᵢⱼ
            M[j, i] = -vᵢⱼ
            k += 1
        end
    end
    return M
end

function skew_symmetric_vec(M::AbstractMatrix)
    n = size(M, 1)
    v = zeros(eltype(M), n * (n - 1) ÷ 2)
    k = 1
    for j = 1:n
        for i = 1:j-1
            v[k] = M[i, j]
            k += 1
        end
    end
    return v
end

function unitary_geodesic(
    operator::EmbeddedOperator,
    samples::Int
)
    U_goal = unembed(operator)
    U_init = Matrix{ComplexF64}(I(size(U_goal, 1)))
    Ũ⃗ = unitary_geodesic(U_init, U_goal, samples)
    return hcat([
        operator_to_iso_vec(EmbeddedOperators.embed(iso_vec_to_operator(Ũ⃗ₜ), operator))
            for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(U_goal, samples; kwargs...)
    N = size(U_goal, 1)
    U₀ = Matrix{ComplexF64}(I(N))
    return unitary_geodesic(U₀, U_goal, samples; kwargs...)
end

unitary_geodesic(
    U₀::AbstractMatrix{<:Number},
    U₁::AbstractMatrix{<:Number},
    samples::Number;
    kwargs...
) = unitary_geodesic(U₀, U₁, range(0, 1, samples); kwargs...)

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
