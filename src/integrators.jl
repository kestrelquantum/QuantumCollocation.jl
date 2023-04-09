module Integrators

export AbstractIntegrator

export QuantumIntegrator
export QuantumStateIntegrator
export UnitaryFourthOrderPade

export jacobian
export hessian_of_the_lagrangian

export DerivativeIntegrator

export state
export controls
export timestep
export comps
export dim

export fourth_order_pade

export SixthOrderPade
export sixth_order_pade

export eighth_order_pade

export TenthOrderPade
export tenth_order_pade

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

using NamedTrajectories
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

function fourth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    Gₜ² = Gₜ^2
    return inv(Id - 1 / 2 * Gₜ + 1 / 9 * Gₜ²) *
        (Id + 1 / 2 * Gₜ + 1 / 9 * Gₜ²)
end


const Id2 = 1.0 * I(2)
const Im2 = 1.0 * [0 -1; 1 0]

anticomm(A::AbstractMatrix, B::AbstractMatrix) = A * B + B * A

function anticomm(A::AbstractMatrix, Bs::AbstractVector{<:AbstractMatrix})
    return [anticomm(A, B) for B in Bs]
end

function anticomm(As::AbstractVector{<:AbstractMatrix{R}}, Bs::AbstractVector{<:AbstractMatrix{R}}) where R
    @assert length(As) == length(Bs)
    n = length(As)
    anticomms = Matrix{Matrix{R}}(undef, n, n)
    for i = 1:n
        for j = 1:n
            anticomms[i, j] = anticomm(As[i], Bs[j])
        end
    end
    return anticomms
end


#
# integrator types
#

abstract type AbstractIntegrator end

abstract type DerivativeIntegrator <: AbstractIntegrator end

abstract type QuantumIntegrator <: AbstractIntegrator end

abstract type QuantumStateIntegrator <: QuantumIntegrator end

abstract type QuantumUnitaryIntegrator <: QuantumIntegrator end


function comps(P::AbstractIntegrator, traj::NamedTrajectory)
    state_comps = traj.components[state(P)]
    u = controls(P)
    if u isa Tuple
        control_comps = Tuple(traj.components[uᵢ] for uᵢ ∈ u)
    else
        control_comps = traj.components[u]
    end
    timestep_comp = traj.components[timestep(P)]
    return state_comps, control_comps, timestep_comp
end

dim(integrator::AbstractIntegrator) = integrator.dim
dim(integrators::AbstractVector{<:AbstractIntegrator}) = sum(dim, integrators)



"""
"""
struct UnitaryFourthOrderPade{R} <: QuantumUnitaryIntegrator
    I_2N::SparseMatrixCSC{R, Int}
    Ω_2N::SparseMatrixCSC{R, Int}
    H_drift_real::Matrix{R}
    H_drift_imag::Matrix{R}
    H_drives_real::Vector{Matrix{R}}
    H_drives_imag::Vector{Matrix{R}}
    H_drift_real_anticomm_H_drift_imag::Matrix{R}
    H_drift_real_squared::Matrix{R}
    H_drift_imag_squared::Matrix{R}
    H_drive_real_anticomms::Matrix{Matrix{R}}
    H_drive_imag_anticomms::Matrix{Matrix{R}}
    H_drift_real_anticomm_H_drives_real::Vector{Matrix{R}}
    H_drift_real_anticomm_H_drives_imag::Vector{Matrix{R}}
    H_drift_imag_anticomm_H_drives_real::Vector{Matrix{R}}
    H_drift_imag_anticomm_H_drives_imag::Vector{Matrix{R}}
    H_drives_real_anticomm_H_drives_imag::Matrix{Matrix{R}}
    unitary_symb::Union{Symbol,Nothing}
    drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}
    timestep_symb::Union{Symbol,Nothing}
    n_drives::Int
    N::Int
    dim::Int

    function UnitaryFourthOrderPade(
        sys::QuantumSystem{R},
        unitary_symb::Union{Symbol,Nothing}=nothing,
        drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing,
        timestep_symb::Union{Symbol,Nothing}=nothing
    ) where R <: Real
        n_drives = length(sys.H_drives_real)
        N = size(sys.H_drift_real, 1)
        dim = 2N^2

        I_2N = sparse(I(2N))
        Ω_2N = sparse(kron(Im2, I(N)))

        H_drift_real_anticomm_H_drift_imag = anticomm(sys.H_drift_real, sys.H_drift_imag)

        H_drift_real_squared = sys.H_drift_real^2
        H_drift_imag_squared = sys.H_drift_imag^2

        H_drive_real_anticomms = anticomm(sys.H_drives_real, sys.H_drives_real)
        H_drive_imag_anticomms = anticomm(sys.H_drives_imag, sys.H_drives_imag)

        H_drift_real_anticomm_H_drives_real =
            anticomm(sys.H_drift_real, sys.H_drives_real)

        H_drift_real_anticomm_H_drives_imag =
            anticomm(sys.H_drift_real, sys.H_drives_imag)

        H_drift_imag_anticomm_H_drives_real =
            anticomm(sys.H_drift_imag, sys.H_drives_real)

        H_drift_imag_anticomm_H_drives_imag =
            anticomm(sys.H_drift_imag, sys.H_drives_imag)

        H_drives_real_anticomm_H_drives_imag =
            anticomm(sys.H_drives_real, sys.H_drives_imag)

        return new{R}(
            I_2N,
            Ω_2N,
            sys.H_drift_real,
            sys.H_drift_imag,
            sys.H_drives_real,
            sys.H_drives_imag,
            H_drift_real_anticomm_H_drift_imag,
            H_drift_real_squared,
            H_drift_imag_squared,
            H_drive_real_anticomms,
            H_drive_imag_anticomms,
            H_drift_real_anticomm_H_drives_real,
            H_drift_real_anticomm_H_drives_imag,
            H_drift_imag_anticomm_H_drives_real,
            H_drift_imag_anticomm_H_drives_imag,
            H_drives_real_anticomm_H_drives_imag,
            unitary_symb,
            drive_symb,
            timestep_symb,
            n_drives,
            N,
            dim
        )
    end
end

state(P::UnitaryFourthOrderPade) = P.unitary_symb
controls(P::UnitaryFourthOrderPade) = P.drive_symb
timestep(P::UnitaryFourthOrderPade) = P.timestep_symb

@inline function squared_operator(
    a::AbstractVector{<:Real},
    A_drift_squared::Matrix{<:Real},
    A_drift_anticomm_A_drives::Vector{<:Matrix{<:Real}},
    A_drive_anticomms::Matrix{<:Matrix{<:Real}},
    n_drives::Int
)
    A² = A_drift_squared
    for i = 1:n_drives
        aⁱ = a[i]
        A² += aⁱ * A_drift_anticomm_A_drives[i]
        A² += aⁱ^2 * A_drive_anticomms[i, i] / 2
        for j = i+1:n_drives
            aʲ = a[j]
            A² += aⁱ * aʲ * A_drive_anticomms[i, j]
        end
    end
    return A²
end

@inline function operator(
    a::AbstractVector{<:Real},
    A_drift::Matrix{<:Real},
    A_drives::Vector{<:Matrix{<:Real}}
)
    return A_drift + sum(a .* A_drives)
end

@inline function operator_anticomm_operator(
    a::AbstractVector{<:Real},
    A_drift_anticomm_B_drift::Matrix{<:Real},
    A_drift_anticomm_B_drives::Vector{<:Matrix{<:Real}},
    B_drift_anticomm_A_drives::Vector{<:Matrix{<:Real}},
    A_drives_anticomm_B_drives::Matrix{<:Matrix{<:Real}},
    n_drives::Int
)
    A_anticomm_B = A_drift_anticomm_B_drift
    for i = 1:n_drives
        aⁱ = a[i]
        A_anticomm_B += aⁱ * A_drift_anticomm_B_drives[i]
        A_anticomm_B += aⁱ * B_drift_anticomm_A_drives[i]
        A_anticomm_B += aⁱ^2 * A_drives_anticomm_B_drives[i, i]
        for j = i+1:n_drives
            aʲ = a[j]
            A_anticomm_B += 2 * aⁱ * aʲ * A_drives_anticomm_B_drives[i, j]
        end
    end
    return A_anticomm_B
end

@inline function operator_anticomm_term(
    a::AbstractVector{<:Real},
    A_drift_anticomm_B_drives::Vector{<:Matrix{<:Real}},
    A_drives_anticomm_B_drives::Matrix{<:Matrix{<:Real}},
    n_drives::Int,
    j::Int
)
    A_anticomm_Bⱼ = A_drift_anticomm_B_drives[j]
    for i = 1:n_drives
        aⁱ = a[i]
        A_anticomm_Bⱼ += aⁱ * A_drives_anticomm_B_drives[i, j]
    end
    return A_anticomm_Bⱼ
end


@inline function B_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector{<:Real},
    Δt::Real
) where R
    HI = operator(a, P.H_drift_imag, P.H_drives_imag)

    HI² = squared_operator(
        a,
        P.H_drift_imag_squared,
        P.H_drift_imag_anticomm_H_drives_imag,
        P.H_drive_imag_anticomms,
        P.n_drives
    )

    HR² = squared_operator(
        a,
        P.H_drift_real_squared,
        P.H_drift_real_anticomm_H_drives_real,
        P.H_drive_real_anticomms,
        P.n_drives
    )

    return I(P.N) - Δt / 2 * HI + Δt^2 / 9 * (HI² - HR²)
end

@inline function B_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector{<:Real},
    Δt::Real
) where R
    HR = operator(a, P.H_drift_real, P.H_drives_real)

    HR_anticomm_HI = operator_anticomm_operator(
        a,
        P.H_drift_real_anticomm_H_drift_imag,
        P.H_drift_real_anticomm_H_drives_imag,
        P.H_drift_imag_anticomm_H_drives_real,
        P.H_drives_real_anticomm_H_drives_imag,
        P.n_drives
    )

    return Δt / 2 * HR - Δt^2 / 9 * HR_anticomm_HI
end

@inline function F_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector{<:Real},
    Δt::Real
) where R
    HI = operator(a, P.H_drift_imag, P.H_drives_imag)

    HI² = squared_operator(
        a,
        P.H_drift_imag_squared,
        P.H_drift_imag_anticomm_H_drives_imag,
        P.H_drive_imag_anticomms,
        P.n_drives
    )

    HR² = squared_operator(
        a,
        P.H_drift_real_squared,
        P.H_drift_real_anticomm_H_drives_real,
        P.H_drive_real_anticomms,
        P.n_drives
    )

    return I(P.N) + Δt / 2 * HI + Δt^2 / 9 * (HI² - HR²)
end

@inline function F_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector{<:Real},
    Δt::Real
) where R
    HR = operator(a, P.H_drift_real, P.H_drives_real)

    HR_anticomm_HI = operator_anticomm_operator(
        a,
        P.H_drift_real_anticomm_H_drift_imag,
        P.H_drift_real_anticomm_H_drives_imag,
        P.H_drift_imag_anticomm_H_drives_real,
        P.H_drives_real_anticomm_H_drives_imag,
        P.n_drives
    )

    return Δt / 2 * HR + Δt^2 / 9 * HR_anticomm_HI
end

# function (P::UnitaryFourthOrderPade)(
#     Ũ⃗ₜ₊₁::AbstractVector,
#     Ũ⃗ₜ::AbstractVector,
#     aₜ::AbstractVector,
#     Δt::Real,
# )
#     BR = B_real(P, aₜ, Δt)
#     BI = B_imag(P, aₜ, Δt)
#     FR = F_real(P, aₜ, Δt)
#     FI = F_imag(P, aₜ, Δt)
#     B̂ = P.I_2N ⊗ BR + P.Ω_2N ⊗ BI
#     F̂ = P.I_2N ⊗ FR - P.Ω_2N ⊗ FI
#     return B̂ * Ũ⃗ₜ₊₁ - F̂ * Ũ⃗ₜ
# end


@views function (P::UnitaryFourthOrderPade{R})(
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
) where R <: Real
    N² = P.N^2
    UₜR = reshape(Ũ⃗ₜ[1:N²], P.N, P.N)
    UₜI = reshape(Ũ⃗ₜ[N²+1:2N²], P.N, P.N)
    Uₜ₊₁R = reshape(Ũ⃗ₜ₊₁[1:N²], P.N, P.N)
    Uₜ₊₁I = reshape(Ũ⃗ₜ₊₁[N²+1:2N²], P.N, P.N)
    BR = B_real(P, aₜ, Δt)
    BI = B_imag(P, aₜ, Δt)
    FR = F_real(P, aₜ, Δt)
    FI = F_imag(P, aₜ, Δt)
    δUR = BR * Uₜ₊₁R - BI * Uₜ₊₁I - FR * UₜR - FI * UₜI
    δUI = BR * Uₜ₊₁I + BI * Uₜ₊₁R - FR * UₜI + FI * UₜR
    δŨ⃗ = vec(hcat(δUR, δUI))
    return δŨ⃗
end

@views function(P::UnitaryFourthOrderPade{R})(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
) where R <: Real
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]
    if P.drive_symb isa Tuple
        aₜ = vcat([zₜ[traj.components[s]] for s in P.drive_symb]...)
    else
        aₜ = zₜ[traj.components[P.drive_symb]]
    end
    Δtₜ = zₜ[traj.components[P.timestep_symb]][1]
    return P(Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
end

function ∂aₜʲB_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂aʲBR = -Δt / 2 * P.H_drives_imag[j]
    ∂aʲBR += Δt^2 / 9 * P.H_drift_imag_anticomm_H_drives_imag[j]
    ∂aʲBR += -Δt^2 / 9 * P.H_drift_real_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂aʲBR += Δt^2 / 9 * aⁱ * P.H_drive_imag_anticomms[i, j]
        ∂aʲBR += -Δt^2 / 9 * aⁱ * P.H_drive_real_anticomms[i, j]
    end
    return ∂aʲBR
end

function ∂aₜʲB_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂aʲBI = Δt / 2 * P.H_drives_real[j]
    ∂aʲBI += -Δt^2 / 9 * P.H_drift_real_anticomm_H_drives_imag[j]
    ∂aʲBI += -Δt^2 / 9 * P.H_drift_imag_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂aʲBI += -Δt^2 / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[i, j]
        ∂aʲBI += -Δt^2 / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[j, i]
    end
    return ∂aʲBI
end

function ∂aₜʲF_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂aʲFR = Δt / 2 * P.H_drives_imag[j]
    ∂aʲFR += Δt^2 / 9 * P.H_drift_imag_anticomm_H_drives_imag[j]
    ∂aʲFR += -Δt^2 / 9 * P.H_drift_real_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂aʲFR += Δt^2 / 9 * aⁱ * P.H_drive_imag_anticomms[i, j]
        ∂aʲFR += -Δt^2 / 9 * aⁱ * P.H_drive_real_anticomms[i, j]
    end
    return ∂aʲFR
end

function ∂aₜʲF_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂aʲFI = Δt / 2 * P.H_drives_real[j]
    ∂aʲFI += Δt^2 / 9 * P.H_drift_real_anticomm_H_drives_imag[j]
    ∂aʲFI += Δt^2 / 9 * P.H_drift_imag_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂aʲFI += Δt^2 / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[i, j]
        ∂aʲFI += Δt^2 / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[j, i]
    end
    return ∂aʲFI
end

function ∂aₜ(
    P::UnitaryFourthOrderPade{R},
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real,
    drive_indices=1:P.n_drives
) where R <: Real
    n_drives = length(aₜ)
    ∂a = spzeros(R, 2P.N^2, n_drives)
    for j = 1:n_drives
        ∂aʲBR = ∂aₜʲB_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂aʲBI = ∂aₜʲB_imag(P, aₜ, Δtₜ, drive_indices[j])
        ∂aʲFR = ∂aₜʲF_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂aʲFI = ∂aₜʲF_imag(P, aₜ, Δtₜ, drive_indices[j])
        # TODO: make this more efficient
        ∂a[:, j] =
            (P.I_2N ⊗ ∂aʲBR + P.Ω_2N ⊗ ∂aʲBI) * Ũ⃗ₜ₊₁ -
            (P.I_2N ⊗ ∂aʲFR - P.Ω_2N ⊗ ∂aʲFI) * Ũ⃗ₜ
    end
    return ∂a
end


function ∂ΔtₜB_real(
    P::UnitaryFourthOrderPade{R},
    aₜ::AbstractVector,
    Δtₜ::Real
) where R
    HI = operator(aₜ, P.H_drift_imag, P.H_drives_imag)
    HI² = squared_operator(
        aₜ,
        P.H_drift_imag_squared,
        P.H_drift_imag_anticomm_H_drives_imag,
        P.H_drive_imag_anticomms,
        P.n_drives
    )
    HR² = squared_operator(
        aₜ,
        P.H_drift_real_squared,
        P.H_drift_real_anticomm_H_drives_real,
        P.H_drive_real_anticomms,
        P.n_drives
    )
    return - 1 / 2 * HI + 2Δtₜ / 9 * (HI² - HR²)
end

function ∂ΔtₜB_imag(
    P::UnitaryFourthOrderPade{R},

    aₜ::AbstractVector,
    Δtₜ::Real
) where R
    HR = operator(aₜ, P.H_drift_real, P.H_drives_real)
    HR_anticomm_HI = operator_anticomm_operator(
        aₜ,
        P.H_drift_real_anticomm_H_drift_imag,
        P.H_drift_real_anticomm_H_drives_imag,
        P.H_drift_imag_anticomm_H_drives_real,
        P.H_drives_real_anticomm_H_drives_imag,
        P.n_drives
    )
    return 1 / 2 * HR - 2Δtₜ / 9 * HR_anticomm_HI
end

function ∂ΔtₜF_real(
    P::UnitaryFourthOrderPade{R},
    aₜ::AbstractVector,
    Δtₜ::Real
) where R
    HI = operator(aₜ, P.H_drift_imag, P.H_drives_imag)
    HI² = squared_operator(
        aₜ,
        P.H_drift_imag_squared,
        P.H_drift_imag_anticomm_H_drives_imag,
        P.H_drive_imag_anticomms,
        P.n_drives
    )
    HR² = squared_operator(
        aₜ,
        P.H_drift_real_squared,
        P.H_drift_real_anticomm_H_drives_real,
        P.H_drive_real_anticomms,
        P.n_drives
    )
    return 1 / 2 * HI + 2Δtₜ / 9 * (HI² - HR²)
end

function ∂ΔtₜF_imag(
    P::UnitaryFourthOrderPade{R},
    aₜ::AbstractVector,
    Δtₜ::Real
) where R
    HR = operator(aₜ, P.H_drift_real, P.H_drives_real)
    HR_anticomm_HI = operator_anticomm_operator(
        aₜ,
        P.H_drift_real_anticomm_H_drift_imag,
        P.H_drift_real_anticomm_H_drives_imag,
        P.H_drift_imag_anticomm_H_drives_real,
        P.H_drives_real_anticomm_H_drives_imag,
        P.n_drives
    )
    return 1 / 2 * HR + 2Δtₜ / 9 * HR_anticomm_HI
end

function ∂Δtₜ(
    P::UnitaryFourthOrderPade{R},
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
) where R <: Real
    ∂ΔtₜBR = ∂ΔtₜB_real(P, aₜ, Δtₜ)
    ∂ΔtₜBI = ∂ΔtₜB_imag(P, aₜ, Δtₜ)
    ∂ΔtₜFR = ∂ΔtₜF_real(P, aₜ, Δtₜ)
    ∂ΔtₜFI = ∂ΔtₜF_imag(P, aₜ, Δtₜ)
    ∂ΔtₜP = (P.I_2N ⊗ ∂ΔtₜBR + P.Ω_2N ⊗ ∂ΔtₜBI) * Ũ⃗ₜ₊₁ -
            (P.I_2N ⊗ ∂ΔtₜFR - P.Ω_2N ⊗ ∂ΔtₜFI) * Ũ⃗ₜ
    return sparse(∂ΔtₜP)
end


function jacobian(
    P::UnitaryFourthOrderPade,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]
    Δtₜ = zₜ[traj.components[P.timestep_symb]][1]

    if P.drive_symb isa Tuple
        aₜs = Tuple(zₜ[traj.components[s]] for s ∈ P.drive_symb)
        ∂aₜPs = []
        let H_drive_mark = 0
            for aₜᵢ ∈ aₜs
                n_aᵢ_drives = length(aₜᵢ)
                drive_indices = (H_drive_mark + 1):(H_drive_mark + n_aᵢ_drives)
                ∂aₜᵢP = ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜᵢ, Δtₜ, drive_indices)
                push!(∂aₜPs, ∂aₜᵢP)
                H_drive_mark += n_aᵢ_drives
            end
        end
        ∂aₜP = tuple(∂aₜPs...)
        ∂ΔtₜP = ∂Δtₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, vcat(aₜs...), Δtₜ)
        BR = B_real(P, vcat(aₜs...), Δtₜ)
        BI = B_imag(P, vcat(aₜs...), Δtₜ)
        FR = F_real(P, vcat(aₜs...), Δtₜ)
        FI = F_imag(P, vcat(aₜs...), Δtₜ)
    else
        aₜ = zₜ[traj.components[P.drive_symb]]
        ∂aₜP = ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
        ∂ΔtₜP = ∂Δtₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
        BR = B_real(P, aₜ, Δtₜ)
        BI = B_imag(P, aₜ, Δtₜ)
        FR = F_real(P, aₜ, Δtₜ)
        FI = F_imag(P, aₜ, Δtₜ)
    end

    F̂ = P.I_2N ⊗ FR - P.Ω_2N ⊗ FI
    B̂ = P.I_2N ⊗ BR + P.Ω_2N ⊗ BI

    ∂Ũ⃗ₜP = -F̂
    ∂Ũ⃗ₜ₊₁P = B̂

    return ∂Ũ⃗ₜP, ∂Ũ⃗ₜ₊₁P, ∂aₜP, ∂ΔtₜP
end

function μ∂aₜ∂Ũ⃗ₜ(
    P::UnitaryFourthOrderPade,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector,
    drive_indices=1:P.n_drives
)
    n_drives = length(aₜ)
    μ∂aₜ∂Ũ⃗ₜP = spzeros(2P.N^2, n_drives)
    for j = 1:n_drives
        ∂aʲFR = ∂aₜʲF_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂aʲFI = ∂aₜʲF_imag(P, aₜ, Δtₜ, drive_indices[j])
        μ∂aₜ∂Ũ⃗ₜP[:, j] = -(P.I_2N ⊗ ∂aʲFR - P.Ω_2N ⊗ ∂aʲFI)' * μₜ
    end
    return sparse(μ∂aₜ∂Ũ⃗ₜP)
end

function μ∂Ũ⃗ₜ₊₁∂aₜ(
    P::UnitaryFourthOrderPade,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector,
    drive_indices=1:P.n_drives
)
    n_drives = length(aₜ)
    μ∂Ũ⃗ₜ₊₁∂aₜP = spzeros(n_drives, 2P.N^2)
    for j = 1:n_drives
        ∂aʲBR = ∂aₜʲB_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂aʲBI = ∂aₜʲB_imag(P, aₜ, Δtₜ, drive_indices[j])
        μ∂Ũ⃗ₜ₊₁∂aₜP[j, :] = μₜ' * (P.I_2N ⊗ ∂aʲBR + P.Ω_2N ⊗ ∂aʲBI)
    end
    return sparse(μ∂Ũ⃗ₜ₊₁∂aₜP)
end

function μ∂²aₜ(
    P::UnitaryFourthOrderPade,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector,
    drive_indices=1:P.n_drives
)
    n_drives = length(drive_indices)
    μ∂²aₜP = spzeros(n_drives, n_drives)
    for j = 1:n_drives
        for i = 1:j
            ∂aⁱ∂aʲBR = Δtₜ^2 / 9 * (
                P.H_drive_imag_anticomms[drive_indices[i], drive_indices[j]] -
                P.H_drive_real_anticomms[drive_indices[i], drive_indices[j]]
            )
            ∂aⁱ∂aʲBI = -Δtₜ^2 / 9 * (
                P.H_drives_real_anticomm_H_drives_imag[drive_indices[i], drive_indices[j]] +
                P.H_drives_real_anticomm_H_drives_imag[drive_indices[j], drive_indices[i]]
            )
            ∂aⁱ∂aʲFR = Δtₜ^2 / 9 * (
                P.H_drive_imag_anticomms[drive_indices[i], drive_indices[j]] -
                P.H_drive_real_anticomms[drive_indices[i], drive_indices[j]]
            )
            ∂aⁱ∂aʲFI = Δtₜ^2 / 9 * (
                P.H_drives_real_anticomm_H_drives_imag[drive_indices[i], drive_indices[j]] +
                P.H_drives_real_anticomm_H_drives_imag[drive_indices[j], drive_indices[i]]
            )
            ∂aⁱ∂aʲB̂ = P.I_2N ⊗ ∂aⁱ∂aʲBR + P.Ω_2N ⊗ ∂aⁱ∂aʲBI
            ∂aⁱ∂aʲF̂ = P.I_2N ⊗ ∂aⁱ∂aʲFR - P.Ω_2N ⊗ ∂aⁱ∂aʲFI
            μ∂²aₜP[i, j] = μₜ' * (∂aⁱ∂aʲB̂ * Ũ⃗ₜ₊₁ - ∂aⁱ∂aʲF̂ * Ũ⃗ₜ)
        end
    end
    return sparse(μ∂²aₜP)
end

function ∂Δtₜ∂aₜʲB_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂Δt∂aʲBR = -1 / 2 * P.H_drives_imag[j]
    ∂Δt∂aʲBR += 2Δt / 9 * P.H_drift_imag_anticomm_H_drives_imag[j]
    ∂Δt∂aʲBR += -2Δt / 9 * P.H_drift_real_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂Δt∂aʲBR += 2Δt / 9 * aⁱ * P.H_drive_imag_anticomms[i, j]
        ∂Δt∂aʲBR += -2Δt / 9 * aⁱ * P.H_drive_real_anticomms[i, j]
    end
    return sparse(∂Δt∂aʲBR)
end

function ∂Δtₜ∂aₜʲB_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂Δt∂aʲBI = 1 / 2 * P.H_drives_real[j]
    ∂Δt∂aʲBI += -2Δt / 9 * P.H_drift_real_anticomm_H_drives_imag[j]
    ∂Δt∂aʲBI += -2Δt / 9 * P.H_drift_imag_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂Δt∂aʲBI += -2Δt / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[i, j]
        ∂Δt∂aʲBI += -2Δt / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[j, i]
    end
    return sparse(∂Δt∂aʲBI)
end

function ∂Δtₜ∂aₜʲF_real(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂Δt∂aʲFR = 1 / 2 * P.H_drives_imag[j]
    ∂Δt∂aʲFR += 2Δt / 9 * P.H_drift_imag_anticomm_H_drives_imag[j]
    ∂Δt∂aʲFR += -2Δt / 9 * P.H_drift_real_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂Δt∂aʲFR += 2Δt / 9 * aⁱ * P.H_drive_imag_anticomms[i, j]
        ∂Δt∂aʲFR += -2Δt / 9 * aⁱ * P.H_drive_real_anticomms[i, j]
    end
    return sparse(∂Δt∂aʲFR)
end

function ∂Δtₜ∂aₜʲF_imag(
    P::UnitaryFourthOrderPade{R},
    a::AbstractVector,
    Δt::Real,
    j::Int
) where R
    ∂Δt∂aʲFI = 1 / 2 * P.H_drives_real[j]
    ∂Δt∂aʲFI += 2Δt / 9 * P.H_drift_real_anticomm_H_drives_imag[j]
    ∂Δt∂aʲFI += 2Δt / 9 * P.H_drift_imag_anticomm_H_drives_real[j]
    for (i, aⁱ) ∈ enumerate(a)
        ∂Δt∂aʲFI += 2Δt / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[i, j]
        ∂Δt∂aʲFI += 2Δt / 9 * aⁱ * P.H_drives_real_anticomm_H_drives_imag[j, i]
    end
    return sparse(∂Δt∂aʲFI)
end

function μ∂Δtₜ∂aₜ(
    P::UnitaryFourthOrderPade,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector,
    drive_indices=1:P.n_drives
)
    n_drives = length(aₜ)
    μ∂Δtₜ∂aₜP = spzeros(n_drives)
    for j = 1:n_drives
        ∂Δtₜ∂aʲBR = ∂Δtₜ∂aₜʲB_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂Δtₜ∂aʲBI = ∂Δtₜ∂aₜʲB_imag(P, aₜ, Δtₜ, drive_indices[j])
        ∂Δtₜ∂aʲFR = ∂Δtₜ∂aₜʲF_real(P, aₜ, Δtₜ, drive_indices[j])
        ∂Δtₜ∂aʲFI = ∂Δtₜ∂aₜʲF_imag(P, aₜ, Δtₜ, drive_indices[j])
        ∂Δtₜ∂aʲB̂ = P.I_2N ⊗ ∂Δtₜ∂aʲBR + P.Ω_2N ⊗ ∂Δtₜ∂aʲBI
        ∂Δtₜ∂aʲF̂ = P.I_2N ⊗ ∂Δtₜ∂aʲFR - P.Ω_2N ⊗ ∂Δtₜ∂aʲFI
        μ∂Δtₜ∂aₜP[j] = μₜ' * (∂Δtₜ∂aʲB̂ * Ũ⃗ₜ₊₁ - ∂Δtₜ∂aʲF̂ * Ũ⃗ₜ)
    end
    return μ∂Δtₜ∂aₜP
end

function μ∂Δtₜ∂Ũ⃗ₜ(
    P::UnitaryFourthOrderPade,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    ∂ΔtF_real = ∂ΔtₜF_real(P, aₜ, Δtₜ)
    ∂ΔtF_imag = ∂ΔtₜF_imag(P, aₜ, Δtₜ)
    return -(P.I_2N ⊗ ∂ΔtF_real - P.Ω_2N ⊗ ∂ΔtF_imag)' * μₜ
end

function μ∂Ũ⃗ₜ₊₁∂Δtₜ(
    P::UnitaryFourthOrderPade,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    ∂ΔtB_real = ∂ΔtₜB_real(P, aₜ, Δtₜ)
    ∂ΔtB_imag = ∂ΔtₜB_imag(P, aₜ, Δtₜ)
    return μₜ' * (P.I_2N ⊗ ∂ΔtB_real + P.Ω_2N ⊗ ∂ΔtB_imag)
end

function μ∂²Δtₜ(
    P::UnitaryFourthOrderPade,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    μₜ::AbstractVector
)
    HI² = squared_operator(
        aₜ,
        P.H_drift_imag_squared,
        P.H_drift_imag_anticomm_H_drives_imag,
        P.H_drive_imag_anticomms,
        P.n_drives
    )
    HR² = squared_operator(
        aₜ,
        P.H_drift_real_squared,
        P.H_drift_real_anticomm_H_drives_real,
        P.H_drive_real_anticomms,
        P.n_drives
    )
    HR_anticomm_HI = operator_anticomm_operator(
        aₜ,
        P.H_drift_real_anticomm_H_drift_imag,
        P.H_drift_real_anticomm_H_drives_imag,
        P.H_drift_imag_anticomm_H_drives_real,
        P.H_drives_real_anticomm_H_drives_imag,
        P.n_drives
    )
    ∂²ΔtₜBR = 2 / 9 * (HI² - HR²)
    ∂²ΔtₜBI = -2 / 9 * HR_anticomm_HI
    ∂²ΔtₜFR = 2 / 9 * (HI² - HR²)
    ∂²ΔtₜFI = 2 / 9 * HR_anticomm_HI
    ∂²ΔtₜB̂ = P.I_2N ⊗ ∂²ΔtₜBR + P.Ω_2N ⊗ ∂²ΔtₜBI
    ∂²ΔtₜF̂ = P.I_2N ⊗ ∂²ΔtₜFR - P.Ω_2N ⊗ ∂²ΔtₜFI
    return μₜ' * (∂²ΔtₜB̂ * Ũ⃗ₜ₊₁ - ∂²ΔtₜF̂ * Ũ⃗ₜ)
end

function hessian_of_the_lagrangian(
    P::UnitaryFourthOrderPade,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    μₜ::AbstractVector,
    traj::NamedTrajectory
)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]

    Δtₜ = zₜ[traj.components[P.timestep_symb]][1]

    if P.drive_symb isa Tuple
        aₜ = Tuple(zₜ[traj.components[s]] for s ∈ P.drive_symb)

        μ∂aₜᵢ∂Ũ⃗ₜPs = []
        μ∂²aₜᵢPs = []
        μ∂Δtₜ∂aₜᵢPs = []
        μ∂Ũ⃗ₜ₊₁∂aₜᵢPs = []

        H_drive_mark = 0

        for aₜᵢ ∈ aₜ
            n_aᵢ_drives = length(aₜᵢ)

            drive_indices = (H_drive_mark + 1):(H_drive_mark + n_aᵢ_drives)

            μ∂aₜᵢ∂Ũ⃗ₜP = μ∂aₜ∂Ũ⃗ₜ(P, aₜᵢ, Δtₜ, μₜ, drive_indices)
            push!(μ∂aₜᵢ∂Ũ⃗ₜPs, μ∂aₜᵢ∂Ũ⃗ₜP)

            μ∂²aₜᵢP = μ∂²aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, Δtₜ, μₜ, drive_indices)
            push!(μ∂²aₜᵢPs, μ∂²aₜᵢP)

            μ∂Δtₜ∂aₜᵢP = μ∂Δtₜ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜᵢ, Δtₜ, μₜ, drive_indices)
            push!(μ∂Δtₜ∂aₜᵢPs, μ∂Δtₜ∂aₜᵢP)

            μ∂Ũ⃗ₜ₊₁∂aₜᵢP = μ∂Ũ⃗ₜ₊₁∂aₜ(P, aₜᵢ, Δtₜ, μₜ, drive_indices)
            push!(μ∂Ũ⃗ₜ₊₁∂aₜᵢPs, μ∂Ũ⃗ₜ₊₁∂aₜᵢP)

            H_drive_mark += n_aᵢ_drives
        end

        μ∂aₜ∂Ũ⃗ₜP = tuple(μ∂aₜᵢ∂Ũ⃗ₜPs...)
        μ∂²aₜP = tuple(μ∂²aₜᵢPs...)
        μ∂Δtₜ∂aₜP = tuple(μ∂Δtₜ∂aₜᵢPs...)
        μ∂Ũ⃗ₜ₊₁∂aₜP = tuple(μ∂Ũ⃗ₜ₊₁∂aₜᵢPs...)

    else
        aₜ = zₜ[traj.components[P.drive_symb]]

        μ∂aₜ∂Ũ⃗ₜP = μ∂aₜ∂Ũ⃗ₜ(P, aₜ, Δtₜ, μₜ)
        μ∂²aₜP = μ∂²aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, Δtₜ, μₜ)
        μ∂Δtₜ∂aₜP = μ∂Δtₜ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ, μₜ)
        μ∂Ũ⃗ₜ₊₁∂aₜP = μ∂Ũ⃗ₜ₊₁∂aₜ(P, aₜ, Δtₜ, μₜ)
    end

    if aₜ isa Tuple
        aₜ = vcat(aₜ...)
    end

    μ∂Δtₜ∂Ũ⃗ₜP = μ∂Δtₜ∂Ũ⃗ₜ(P, aₜ, Δtₜ, μₜ)
    μ∂²ΔtₜP = μ∂²Δtₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, μₜ)
    μ∂Ũ⃗ₜ₊₁∂ΔtₜP = μ∂Ũ⃗ₜ₊₁∂Δtₜ(P, aₜ, Δtₜ, μₜ)

    return (
        μ∂aₜ∂Ũ⃗ₜP,
        μ∂²aₜP,
        μ∂Δtₜ∂Ũ⃗ₜP,
        μ∂Δtₜ∂aₜP,
        μ∂²ΔtₜP,
        μ∂Ũ⃗ₜ₊₁∂aₜP,
        μ∂Ũ⃗ₜ₊₁∂ΔtₜP
    )
end


function (P::FourthOrderPade)(
    xₜ₊₁::AbstractVector{<:Real},
    xₜ::AbstractVector{<:Real},
    Δt::Real;
    kwargs...
)
    return P(xₜ₊₁, xₜ, zeros(eltype(xₜ), length(P.G_drives)), Δt; kwargs...)
end

struct TenthOrderPade <: QuantumStateIntegrator
    G_drift::Matrix
    G_drives::Vector{Matrix}
    nqstates::Int
    isodim::Int

    TenthOrderPade(sys::QuantumSystem) =
        new(sys.G_drift, sys.G_drives, sys.nqstates, sys.isodim)
end


function (P::TenthOrderPade)(
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
        Ũₜ₊₁ = iso_vec_to_iso_operator(xₜ₊₁)
        Ũₜ = iso_vec_to_iso_operator(xₜ)
        δŨ = (Id + Δt^2 / 9 * Gₜ^2 + Δt^4 * Gₜ^4 / 1008) * (Ũₜ₊₁ - Ũₜ) -
            (Δt / 2 * Gₜ + Δt^3 * Gₜ^3 / 72 + Δt^5 * Gₜ^3 / 30240) * (Ũₜ₊₁ + Ũₜ)
        δŨ⃗ = iso_operator_to_iso_vec(δŨ)
        return δŨ⃗
    else
        δx = (Id + Δt^2 / 9 * Gₜ^2 + Δt^4 * Gₜ^4 / 1008) * (xₜ₊₁ - xₜ) -
        (Δt / 2 * Gₜ + Δt^3 * Gₜ^3 / 72 + Δt^5 * Gₜ^3 / 30240) * (xₜ₊₁ + xₜ)
        return δx
    end
end

function (P::TenthOrderPade)(
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


function sixth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    Gₜ² = Gₜ^2
    Gₜ3 = Gₜ^3
    return inv(Id - 1 / 2 * Gₜ + 1 / 9 * Gₜ² - 1/72 * Gₜ3) *
        (Id + 1 / 2 * Gₜ + 1 / 9 * Gₜ² + 1/72 * Gₜ3)
end

function eighth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    Gₜ² = Gₜ^2
    Gₜ² = Gₜ^2
    Gₜ³ = Gₜ^3
    Gₜ⁴ = Gₜ^4
    return inv(Id - 1 / 2 * Gₜ + 1 / 9 * Gₜ² - 1/72 * Gₜ³ + 1/1008*Gₜ4) *
        (Id + 1 / 2 * Gₜ + 1 / 9 * Gₜ² + 1/72 * Gₜ³ + 1/1008 * Gₜ⁴)
end

function tenth_order_pade(Gₜ::Matrix)
    Id = I(size(Gₜ, 1))
    Gₜ² = Gₜ^2
    Gₜ² = Gₜ^2
    Gₜ³ = Gₜ^3
    Gₜ⁴ = Gₜ^4
    Gₜ⁵ = Gₜ^5
    return inv(Id - 1 / 2 * Gₜ + 1 / 9 * Gₜ² - 1/72 * Gₜ³ + 1/1008*Gₜ⁴ - 1/30240 * Gₜ⁵) *
        (Id + 1 / 2 * Gₜ + 1 / 9 * Gₜ² + 1/72 * Gₜ³ + 1/1008 * Gₜ⁴ + 1/30240 * Gₜ⁵)
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
