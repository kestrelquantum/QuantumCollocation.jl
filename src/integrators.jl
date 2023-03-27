module Integrators

export AbstractIntegrator
export QuantumStateIntegrator

export Exponential

export SecondOrderPade
export second_order_pade

export FourthOrderPade
export fourth_order_pade

# jacobians
export ∂ψ̃ⁱₜ
export ∂ψ̃ⁱₜ₊₁
export ∂aₜ

# hessians of the lagrangian
export μₜ∂²aₜ
export μⁱₜ∂aₜ∂ψ̃ⁱₜ
export μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁

# min time jacobian and hessians of the lagrangian
export ∂Δtₜ
export μₜ∂²Δtₜ
export μⁱₜ∂Δtₜ∂ψ̃ⁱₜ
export μⁱₜ∂Δtₜ∂ψ̃ⁱₜ₊₁
export μₜ∂Δtₜ∂aₜ


using ..IndexingUtils
using ..QuantumSystems
using ..QuantumUtils

using LinearAlgebra
using SparseArrays


# G(a) helper function

function G(
    a::AbstractVector,
    G_drift::AbstractMatrix,
    G_drives::AbstractVector{<:AbstractMatrix}
)
    return G_drift + sum(a .* G_drives)
end

const Id2 = 1.0 * I(2)
const Im2 = 1.0 * [0 -1; 1 0]

function vecinv(
    x::AbstractVector{T};
    shape::Tuple{Int,Int}=(Int(sqrt(length(x))), Int(sqrt(length(x))))
)::Matrix{T} where T
    m, n = shape
    return reshape(x, m, n)
end

opr_re(x::AbstractVector) = Id2 ⊗ vecinv(x)
opr_im(x::AbstractVector) = Im2 ⊗ vecinv(x)

revec(x::AbstractVector) = vec(Id2 ⊗ vecinv(x))
imvec(x::AbstractVector) = vec(Im2 ⊗ vecinv(x))

function projector(N::Int)
    Pre = hcat(I(N), zeros(N, N))
    Pim = hcat(zeros(N, N), I(N))
    P̂ = Pre'Pre ⊗ Pre - Pim'Pim ⊗ Pre
    return P̂
end

function isovec(Ũ::AbstractMatrix)
    N = size(Ũ, 1) ÷ 2
    P̂ = projector(N)
    return P̂ * vec(Ũ)
end

function iso_operator(Ũ⃗::AbstractVector)
    N² = length(Ũ⃗) ÷ 2
    return opr_re(Ũ⃗[1:N²]) + opr_im(Ũ⃗[N²+1:end])
end

v(Ũ⃗::AbstractVector) = vec(iso_operator(Ũ⃗))

function ∂Ũ⃗v(Ũ⃗::AbstractVector)
    N² = length(Ũ⃗) ÷ 2
    E = I(N²)
    ∂v = Matrix{Float64}(undef, 4N², 2N²)
    for i = 1:N²
        eᵢ = E[:, i]
        ∂v[:, i] = revec(eᵢ)
        ∂v[:, N² + i] = imvec(eᵢ)
    end
    return ∂v
end




#
# integrator types
#

abstract type AbstractIntegrator end

abstract type QuantumStateIntegrator <: AbstractIntegrator end


# exponential

struct Exponential <: QuantumStateIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}

    Exponential(sys::QuantumSystem) =
        new(sys.G_drift, sys.G_drives)
end

function (integrator::Exponential)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
    return ψ̃ₜ₊₁ - exp(Gₜ * Δt) * ψ̃ₜ
end


# 2nd order Pade integrator

struct SecondOrderPade <: QuantumStateIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}
    nqstates::Int
    isodim::Int

    SecondOrderPade(sys::QuantumSystem) =
        new(sys.G_drift, sys.G_drives, sys.nqstates, sys.isodim)
end

function (integrator::SecondOrderPade)(
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
    # Id = I(size(Gₜ, 1))
    # return (Id - Δt / 2 * Gₜ) * ψ̃ₜ₊₁ -
    #        (Id + Δt / 2 * Gₜ) * ψ̃ₜ
    return ψ̃ₜ₊₁ - ψ̃ₜ - Δt / 2 * Gₜ * (ψ̃ₜ₊₁ + ψ̃ₜ)
end

function second_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    return inv(Id - 1 / 2 * Gₜ) *
        (Id + 1 / 2 * Gₜ)
end


# 4th order Pade integrator

anticom(A::AbstractMatrix, B::AbstractMatrix) = A * B + B * A

struct FourthOrderPade <: QuantumStateIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}
    G_drive_anticoms::Symmetric
    G_drift_anticoms::Vector{Matrix}
    P̂::Matrix

    function FourthOrderPade(sys::QuantumSystem)

        ncontrols = length(sys.G_drives)

        drive_anticoms = fill(
            zeros(size(sys.G_drift)),
            ncontrols,
            ncontrols
        )

        for j = 1:ncontrols
            for k = 1:j
                if k == j
                    drive_anticoms[k, k] = 2 * sys.G_drives[k]^2
                else
                    drive_anticoms[k, j] =
                        anticom(sys.G_drives[k], sys.G_drives[j])
                end
            end
        end

        drift_anticoms = [
            anticom(G_drive, sys.G_drift)
                for G_drive in sys.G_drives
        ]

        N = size(sys.G_drift, 1) ÷ 2

        P̂ = projector(N)

        return new(
            sys.G_drift,
            sys.G_drives,
            Symmetric(drive_anticoms),
            drift_anticoms,
            P̂
        )
    end
end

function (P::FourthOrderPade)(
    xₜ₊₁::AbstractVector{<:Real},
    xₜ::AbstractVector{<:Real},
    uₜ::AbstractVector{<:Real},
    Δt::Real;
    G_additional::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
    operator::Bool=false
)
    Gₜ = G(uₜ, P.G_drift, P.G_drives)
    if !isnothing(G_additional)
        Gₜ += G_additional
    end
    Id = I(size(Gₜ, 1))
    if operator
        Gₜ² = Gₜ^2
        # vₜ = v(xₜ)
        # vₜ₊₁ = v(xₜ₊₁)
        # δŨ⃗ = (Id ⊗ (Id + Δt^2 / 9 * Gₜ²)) * (vₜ₊₁ - vₜ) -
            # (Id ⊗ (Δt / 2 * Gₜ)) * (vₜ₊₁ + vₜ)
        Ũₜ₊₁ = iso_operator(xₜ₊₁)
        Ũₜ = iso_operator(xₜ)
        δŨ = (Id + Δt^2 / 9 * Gₜ²) * (Ũₜ₊₁ - Ũₜ) -
            Δt / 2 * Gₜ * (Ũₜ₊₁ + Ũₜ)
        return P.P̂ * vec(δŨ)
        # Ũₜ₊₁ = iso_vec_to_iso_operator(xₜ₊₁)
        # Ũₜ = iso_vec_to_iso_operator(xₜ)
        # δŨ = (Id + Δt^2 / 9 * Gₜ²) * (Ũₜ₊₁ - Ũₜ) -
        #     Δt / 2 * Gₜ * (Ũₜ₊₁ + Ũₜ)
        # δŨ⃗ = iso_operator_to_iso_vec(δŨ)
        # return δŨ⃗
    else
        δx = (Id + Δt^2 / 9 * Gₜ^2) * (xₜ₊₁ - xₜ) -
            Δt / 2 * Gₜ * (xₜ₊₁ + xₜ)
        return δx
    end
end

function (P::FourthOrderPade)(
    xₜ₊₁::AbstractVector{<:Real},
    xₜ::AbstractVector{<:Real},
    Δt::Real;
    kwargs...
)
    return P(xₜ₊₁, xₜ, zeros(eltype(xₜ), length(P.G_drives)), Δt; kwargs...)
end


# function (integrator::FourthOrderPade)(
#     ψ̃ₜ₊₁::AbstractVector,
#     ψ̃ₜ::AbstractVector,
#     aₜ::AbstractVector,
#     Δt::Real
# )
#     Gₜ = G(aₜ, integrator.G_drift, integrator.G_drives)
#     Id = I(size(Gₜ, 1))
#     # return (Id - Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2) * ψ̃ₜ₊₁ -
#     #        (Id + Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2) * ψ̃ₜ
#     return (Id + Δt^2 / 9 * Gₜ^2) * (ψ̃ₜ₊₁ - ψ̃ₜ) -
#         Δt / 2 * Gₜ * (ψ̃ₜ₊₁ + ψ̃ₜ)
# end

function fourth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    Gₜ² = Gₜ^2
    return inv(Id - 1 / 2 * Gₜ + 1 / 9 * Gₜ²) *
        (Id + 1 / 2 * Gₜ + 1 / 9 * Gₜ²)
end


#
# Jacobians
#


function ∂ψ̃ⁱₜ(
    P::SecondOrderPade,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    Id = I(size(Gₜ, 1))
    return -(Id + Δt / 2 * Gₜ)
end

function ∂ψ̃ⁱₜ(
    P::FourthOrderPade,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    Id = I(size(Gₜ, 1))
    return -(Id + Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2)
end


function ∂ψ̃ⁱₜ₊₁(
    P::SecondOrderPade,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    Id = I(size(Gₜ, 1))
    return Id - Δt / 2 * Gₜ
end

function ∂ψ̃ⁱₜ₊₁(
    P::FourthOrderPade,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    Id = I(size(Gₜ, 1))
    return Id - Δt / 2 * Gₜ + Δt^2 / 9 * Gₜ^2
end


function ∂aₜ(
    P::SecondOrderPade,
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    ∂aₜPⁱₜ = zeros(length(ψ̃ⁱₜ), length(aₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        ∂aₜPⁱₜ[:, j] = -Δt / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ)
    end
    return ∂aₜPⁱₜ
end

function ∂aₜ(
    P::FourthOrderPade,
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    ∂aₜPⁱₜ = zeros(length(ψ̃ⁱₜ), length(aₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        Gʲ_anticom_Gₜ =
            G(aₜ, P.G_drift_anticoms[j], P.G_drive_anticoms[:, j])
        ∂aₜPⁱₜ[:, j] =
            -Δt / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ) +
            Δt^2 / 9 * Gʲ_anticom_Gₜ * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
    end
    return ∂aₜPⁱₜ
end


function ∂Δtₜ(
    P::SecondOrderPade,
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real # not used, but kept to match 4th order method signature
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    return - 1 / 2 * Gₜ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ)
end

function ∂Δtₜ(
    P::FourthOrderPade,
    ψ̃ⁱₜ₊₁::AbstractVector,
    ψ̃ⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    return - 1 / 2 * Gₜ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ) +
        2 / 9 * Δtₜ * Gₜ^2 * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
end


#
# Hessians of the Lagrangian
#



# for dispatching purposes
function μₜ∂²aₜ(P::SecondOrderPade, args...)
    ncontrols = length(P.G_drives)
    return zeros(ncontrols, ncontrols)
end

# TODO: test and add multithreading

function μₜ∂²aₜ(
    P::FourthOrderPade,
    μₜ::AbstractVector,
    Ψ̃ₜ₊₁::AbstractVector,
    Ψ̃ₜ::AbstractVector,
    Δt::Real
)
    ncontrols = length(P.G_drives)
    μₜ∂²aₜPₜ = zeros(ncontrols, ncontrols)
    for i = 1:P.nqstates
        ψ̃ⁱ_slice = slice(i, P.isodim)
        ψ̃ⁱₜ₊₁ = Ψ̃ₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = Ψ̃ₜ[ψ̃ⁱ_slice]
        μⁱₜ = μₜ[ψ̃ⁱ_slice]
        for j = 1:ncontrols # jth column
            for k = 1:j     # kth row
                ∂aᵏₜ∂aʲₜPⁱₜ = Δt^2 / 9 * P.G_drive_anticoms[k, j] * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
                μₜ∂²aₜPₜ[k, j] += dot(μⁱₜ, ∂aᵏₜ∂aʲₜPⁱₜ)
            end
        end
    end
    return Symmetric(μₜ∂²aₜPₜ)
end


function μⁱₜ∂aₜ∂ψ̃ⁱₜ(
    P::SecondOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ = zeros(length(μⁱₜ), length(aₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        ∂aₜ∂ψ̃ⁱₜPⁱₜ = -Δt / 2 * Gʲ
        μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ[:, j] = (∂aₜ∂ψ̃ⁱₜPⁱₜ)' * μⁱₜ
    end
    return μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ
end

function μⁱₜ∂aₜ∂ψ̃ⁱₜ(
    P::FourthOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ = zeros(length(μⁱₜ), length(aₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        Ĝʲ = G(aₜ, P.G_drift_anticoms[j], P.G_drive_anticoms[:, j])
        ∂aₜ∂ψ̃ⁱₜPⁱₜ = -(Δt / 2 * Gʲ + Δt^2 / 9 * Ĝʲ)
        μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ[:, j] = (∂aₜ∂ψ̃ⁱₜPⁱₜ)' * μⁱₜ
    end
    return μⁱₜ∂aₜ∂ψ̃ⁱₜPⁱₜ
end


function μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁(
    P::SecondOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ = zeros(length(aₜ), length(μⁱₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ = -Δt / 2 * Gʲ
        μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ[j, :] = (μⁱₜ)' * ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ
    end
    return μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ
end

function μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁(
    P::FourthOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real,
)
    μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ = zeros(length(aₜ), length(μⁱₜ))
    for j = 1:length(aₜ)
        Gʲ = P.G_drives[j]
        Ĝʲ = G(aₜ, P.G_drift_anticoms[j], P.G_drive_anticoms[:, j])
        ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ = -Δt / 2 * Gʲ + Δt^2 / 9 * Ĝʲ
        μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ[j, :] = (μⁱₜ)' * ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ
    end
    return μⁱₜ∂aₜ∂ψ̃ⁱₜ₊₁Pⁱₜ
end



#
# min time problem hessians of the lagrangian
#


# for dispatching purposes
μₜ∂²Δtₜ(P::SecondOrderPade, args...) = 0.0

@views function μₜ∂²Δtₜ(
    P::FourthOrderPade,
    μₜ::AbstractVector,
    Ψ̃ₜ₊₁::AbstractVector,
    Ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    μₜ∂²ΔtₜPₜ = 0.0
    for i = 1:P.nqstates
        ψ̃ⁱ_slice = slice(i, P.isodim)
        ψ̃ⁱₜ₊₁ = Ψ̃ₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = Ψ̃ₜ[ψ̃ⁱ_slice]
        μⁱₜ = μₜ[ψ̃ⁱ_slice]
        ∂²ΔtₜPⁱₜ = 2 / 9 * Gₜ^2 * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
        μₜ∂²ΔtₜPₜ += dot(μⁱₜ, ∂²ΔtₜPⁱₜ)
    end
    return μₜ∂²ΔtₜPₜ
end


function μⁱₜ∂Δtₜ∂ψ̃ⁱₜ(
    P::SecondOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real # kept for dispatching purposes
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    ∂Δtₜ∂ψ̃ⁱₜPⁱₜ = -1 / 2 * Gₜ
    return (∂Δtₜ∂ψ̃ⁱₜPⁱₜ)' * μⁱₜ
end

function μⁱₜ∂Δtₜ∂ψ̃ⁱₜ(
    P::FourthOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    ∂Δtₜ∂ψ̃ⁱₜPⁱₜ = -(1 / 2 * Gₜ + 2 / 9 * Δtₜ * Gₜ^2)
    return (∂Δtₜ∂ψ̃ⁱₜPⁱₜ)' * μⁱₜ
end


μⁱₜ∂Δtₜ∂ψ̃ⁱₜ₊₁(P::SecondOrderPade, args...) = μⁱₜ∂Δtₜ∂ψ̃ⁱₜ(P, args...)

function μⁱₜ∂Δtₜ∂ψ̃ⁱₜ₊₁(
    P::FourthOrderPade,
    μⁱₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    ∂Δtₜ∂ψ̃ⁱₜ₊₁Pⁱₜ = -1 / 2 * Gₜ + 2 / 9 * Δtₜ * Gₜ^2
    return (∂Δtₜ∂ψ̃ⁱₜ₊₁Pⁱₜ)' * μⁱₜ
end

@views function μₜ∂Δtₜ∂aₜ(
    P::SecondOrderPade,
    μₜ::AbstractVector,
    Ψ̃ₜ₊₁::AbstractVector,
    Ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real # kept for dispatching purposes
)
    μₜ∂Δtₜ∂aₜPₜ = zeros(length(aₜ))
    for i = 1:P.nqstates
        ψ̃ⁱ_slice = slice(i, P.isodim)
        μⁱₜ = μₜ[ψ̃ⁱ_slice]
        ψ̃ⁱₜ₊₁ = Ψ̃ₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = Ψ̃ₜ[ψ̃ⁱ_slice]
        for j = 1:length(aₜ)
            Gʲ = P.G_drives[j]
            ∂Δtₜ∂aʲₜPⁱₜ = -1 / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ)
            μₜ∂Δtₜ∂aₜPₜ[j] += dot(μⁱₜ, ∂Δtₜ∂aʲₜPⁱₜ)
        end
    end
    return μₜ∂Δtₜ∂aₜPₜ
end

@views function μₜ∂Δtₜ∂aₜ(
    P::FourthOrderPade,
    μₜ::AbstractVector,
    Ψ̃ₜ₊₁::AbstractVector,
    Ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    μₜ∂Δtₜ∂aₜPₜ = zeros(length(aₜ))
    for i = 1:P.nqstates
        ψ̃ⁱ_slice = slice(i, P.isodim)
        μⁱₜ = μₜ[ψ̃ⁱ_slice]
        ψ̃ⁱₜ₊₁ = Ψ̃ₜ₊₁[ψ̃ⁱ_slice]
        ψ̃ⁱₜ = Ψ̃ₜ[ψ̃ⁱ_slice]
        for j = 1:length(aₜ)
            Gʲ = P.G_drives[j]
            Ĝʲ = G(aₜ, P.G_drift_anticoms[j], P.G_drive_anticoms[:, j])
            ∂Δtₜ∂aʲₜPⁱₜ =
                -1 / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ) +
                2 / 9 * Δtₜ * Ĝʲ * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
            μₜ∂Δtₜ∂aₜPₜ[j] += dot(μⁱₜ, ∂Δtₜ∂aʲₜPⁱₜ)
        end
    end
    return μₜ∂Δtₜ∂aₜPₜ
end

end
