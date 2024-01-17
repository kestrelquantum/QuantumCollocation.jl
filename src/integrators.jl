module Integrators

export AbstractIntegrator

export QuantumIntegrator

export QuantumPadeIntegrator
export QuantumStatePadeIntegrator
export UnitaryPadeIntegrator

export DerivativeIntegrator

export state
export controls
export timestep
export comps
export dim

export jacobian
export hessian_of_the_lagrangian

export nth_order_pade
export fourth_order_pade
export sixth_order_pade
export eighth_order_pade
export tenth_order_pade

using TrajectoryIndexingUtils
using ..QuantumSystems
using ..QuantumUtils

using NamedTrajectories
using LinearAlgebra
using SparseArrays



function nth_order_pade(Gₜ::Matrix, n::Int)
    @assert n ∈ keys(PADE_COEFFICIENTS)
    coeffs = PADE_COEFFICIENTS[n]
    Id = 1.0I(size(Gₜ, 1))
    p = n ÷ 2
    G_powers = [Gₜ^i for i = 1:p]
    B = Id + sum((-1)^k * coeffs[k] * G_powers[k] for k = 1:p)
    F = Id + sum(coeffs[k] * G_powers[k] for k = 1:p)
    return inv(B) * F
end



fourth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 4)
sixth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 6)
eighth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 8)
tenth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 10)

function compute_powers(G::AbstractMatrix{T}, order::Int) where T <: Number
    powers = Array{typeof(G)}(undef, order)
    powers[1] = G
    for k = 2:order
        powers[k] = powers[k-1] * G
    end
    return powers
end


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

anticomm(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Number = A * B + B * A

function anticomm(
    A::AbstractMatrix{R},
    Bs::AbstractVector{<:AbstractMatrix{R}}
) where R <: Number
    return [anticomm(A, B) for B in Bs]
end

function anticomm(
    As::AbstractVector{<:AbstractMatrix{R}},
    Bs::AbstractVector{<:AbstractMatrix{R}}
) where R <: Number
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

pade(n, k) = (factorial(n + k) // (factorial(n - k) * factorial(k) * 2^n))
pade_coeffs(n) = [pade(n, k) for k = n:-1:0][2:end] // pade(n, n)

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

function build_anticomms(
    G_drift::AbstractMatrix{R},
    G_drives::Vector{<:AbstractMatrix{R}},
    n_drives::Int) where R <: Number

    drive_anticomms = fill(
            zeros(size(G_drift)),
            n_drives,
            n_drives
        )

        for j = 1:n_drives
            for k = 1:j
                if k == j
                    drive_anticomms[k, k] = 2 * G_drives[k]^2
                else
                    drive_anticomms[k, j] =
                        anticomm(G_drives[k], G_drives[j])
                end
            end
        end

        drift_anticomms = [
            anticomm(G_drive, G_drift)
                for G_drive in G_drives
        ]

    return Symmetric(drive_anticomms), drift_anticomms
end

#
# integrator types
#

abstract type AbstractIntegrator end

abstract type QuantumIntegrator <: AbstractIntegrator end

abstract type QuantumPadeIntegrator <: QuantumIntegrator end


function comps(P::AbstractIntegrator, traj::NamedTrajectory)
    state_comps = traj.components[state(P)]
    u = controls(P)
    if u isa Tuple
        control_comps = Tuple(traj.components[uᵢ] for uᵢ ∈ u)
    else
        control_comps = traj.components[u]
    end
    if traj.timestep isa Float64
        return state_comps, control_comps
    else
        timestep_comp = traj.components[traj.timestep]
        return state_comps, control_comps, timestep_comp
    end
end

dim(integrator::AbstractIntegrator) = integrator.dim
dim(integrators::AbstractVector{<:AbstractIntegrator}) = sum(dim, integrators)


###
### Derivative Integrator
###

struct DerivativeIntegrator <: AbstractIntegrator
    variable::Symbol
    derivative::Symbol
    dim::Int
end

function DerivativeIntegrator(
    variable::Symbol,
    derivative::Symbol,
    traj::NamedTrajectory
)
    return DerivativeIntegrator(variable, derivative, traj.dims[variable])
end

state(integrator::DerivativeIntegrator) = integrator.variable
controls(integrator::DerivativeIntegrator) = integrator.derivative

@views function (D::DerivativeIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    xₜ = zₜ[traj.components[D.variable]]
    xₜ₊₁ = zₜ₊₁[traj.components[D.variable]]
    dxₜ = zₜ[traj.components[D.derivative]]
    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end
    return xₜ₊₁ - xₜ - Δtₜ * dxₜ
end

@views function jacobian(
    D::DerivativeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    dxₜ = zₜ[traj.components[D.derivative]]
    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end
    ∂xₜD = sparse(-1.0I(D.dim))
    ∂xₜ₊₁D = sparse(1.0I(D.dim))
    ∂dxₜD = sparse(-Δtₜ * I(D.dim))
    ∂ΔtₜD = -dxₜ
    return ∂xₜD, ∂xₜ₊₁D, ∂dxₜD, ∂ΔtₜD
end

# key is the order of the integrator
# and the value are the Pade coefficients
# for each term
const PADE_COEFFICIENTS = Dict{Int,Vector{Float64}}(
    4 => [1/2, 1/12],
    6 => [1/2, 1/10, 1/120],
    8 => [1/2, 3/28, 1/84, 1/1680],
    10 => [1/2, 1/9, 1/72, 1/1008, 1/30240]
)

"""
"""
struct UnitaryPadeIntegrator <: QuantumPadeIntegrator
    I_2N::SparseMatrixCSC{Float64, Int}
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    G_drive_anticomms::Union{Nothing, Symmetric}
    G_drift_anticomms::Union{Nothing, Vector{Matrix{Float64}}}
    unitary_symb::Union{Symbol, Nothing}
    drive_symb::Union{Symbol, Tuple{Vararg{Symbol}}, Nothing}
    n_drives::Int
    N::Int
    dim::Int
    order::Int
    autodiff::Bool
    G::Union{Function, Nothing}

    """
        UnitaryPadeIntegrator(
            sys::AbstractQuantumSystem,
            unitary_symb::Symbol,
            drive_symb::Union{Symbol,Tuple{Vararg{Symbol}}};
            order::Int=4,
            autodiff::Bool=order != 4
        ) where R <: Real

    Construct a `UnitaryPadeIntegrator` for the quantum system `sys`.

    # Examples

    ## a bare integrator
    ```julia
        P = UnitaryPadeIntegrator(sys)
    ```

    ## for a single drive `a`:
    ```julia
        P = UnitaryPadeIntegrator(sys, :Ũ⃗, :a)
    ```

    ## for two drives `α` and `γ`, order `4`, and autodiffed:
    ```julia
        P = UnitaryPadeIntegrator(sys, :Ũ⃗, (:α, :γ); order=4, autodiff=true)
    ```

    # Arguments
    - `sys::AbstractQuantumSystem`: the quantum system
    - `unitary_symb::Union{Symbol,Nothing}=nothing`: the symbol for the unitary
    - `drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing`: the symbol(s) for the drives
    - `order::Int=4`: the order of the Pade approximation. Must be in `[4, 6, 8, 10]`. If order is not `4` and `autodiff` is `false`, then the integrator will use the hand-coded fourth order derivatives.
    - `autodiff::Bool=order != 4`: whether to use automatic differentiation to compute the jacobian and hessian of the lagrangian

    """
    function UnitaryPadeIntegrator(
        sys::AbstractQuantumSystem,
        unitary_symb::Union{Symbol,Nothing}=nothing,
        drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing;
        order::Int=4,
        autodiff::Bool=false,
        G::Union{Function, Nothing}=nothing,
    )
        @assert order ∈ [4, 6, 8, 10] "order must be in [4, 6, 8, 10]"
        @assert !isnothing(unitary_symb) "must specify unitary symbol"
        @assert !isnothing(drive_symb) "must specify drive symbol"

        n_drives = length(sys.H_drives)
        N = size(sys.H_drift, 1)
        dim = 2N^2

        I_2N = sparse(I(2N))

        G_drift = sys.G_drift
        G_drives = sys.G_drives

        drive_anticomms, drift_anticomms =
            order == 4 ? build_anticomms(G_drift, G_drives, n_drives) : (nothing, nothing)

        return new(
            I_2N,
            G_drift,
            G_drives,
            drive_anticomms,
            drift_anticomms,
            unitary_symb,
            drive_symb,
            n_drives,
            N,
            dim,
            order,
            autodiff,
            G,
        )
    end
end

state(P::UnitaryPadeIntegrator) = P.unitary_symb
controls(P::UnitaryPadeIntegrator) = P.drive_symb

struct QuantumStatePadeIntegrator <: QuantumPadeIntegrator
    I_2N::SparseMatrixCSC{Float64, Int}
    G_drift::Matrix{Float64}
    G_drives::Vector{Matrix{Float64}}
    G_drive_anticomms::Union{Symmetric, Nothing}
    G_drift_anticomms::Union{Vector{Matrix{Float64}}, Nothing}
    state_symb::Union{Symbol,Nothing}
    drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}
    n_drives::Int
    N::Int
    dim::Int
    order::Int
    autodiff::Bool
    G::Function

    """
        QuantumStatePadeIntegrator(
            sys::AbstractQuantumSystem,
            state_symb::Union{Symbol,Nothing}=nothing,
            drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing,
            timestep_symb::Union{Symbol,Nothing}=nothing;
            order::Int=4,
            autodiff::Bool=false
        ) where R <: Real

    Construct a `QuantumstatePadeIntegrator` for the quantum system `sys`.

    # Examples

    ## for a single drive `a`:
    ```julia
        P = QuantumstatePadeIntegrator(sys, :ψ̃, :a)
    ```

    ## for two drives `α` and `γ`, order `8`, and autodiffed:
    ```julia
        P = QuantumstatePadeIntegrator(sys, :ψ̃, (:α, :γ); order=8, autodiff=true)
    ```

    # Arguments
    - `sys::AbstractQuantumSystem`: the quantum system
    - `state_symb::Symbol`: the symbol for the quantum state
    - `drive_symb::Union{Symbol,Tuple{Vararg{Symbol}}}`: the symbol(s) for the drives
    - `order::Int=4`: the order of the Pade approximation. Must be in `[4, 6, 8, 10]`. If order is not `4` and `autodiff` is `false`, then the integrator will use the hand-coded fourth order derivatives.
    - `autodiff::Bool=false`: whether to use automatic differentiation to compute the jacobian and hessian of the lagrangian
    """
    function QuantumStatePadeIntegrator(
        sys::AbstractQuantumSystem,
        state_symb::Union{Symbol,Nothing}=nothing,
        drive_symb::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing;
        order::Int=4,
        autodiff::Bool=order != 4,
        G::Union{Function, Nothing}=nothing,
    )
        @assert order ∈ [4, 6, 8, 10] "order must be in [4, 6, 8, 10]"
        @assert !isnothing(state_symb) "state_symb must be specified"
        @assert !isnothing(drive_symb) "drive_symb must be specified"
        n_drives = length(sys.H_drives_real)
        N = size(sys.H_drift_real, 1)
        dim = 2N
        I_2N = sparse(I(2N))

        G_drift = sys.G_drift
        G_drives = sys.G_drives

        drive_anticomms, drift_anticomms =
            order == 4 ? build_anticomms(G_drift, G_drives, n_drives) : (nothing, nothing)

        return new(
            I_2N,
            G_drift,
            G_drives,
            drive_anticomms,
            drift_anticomms,
            state_symb,
            drive_symb,
            n_drives,
            N,
            dim,
            order,
            autodiff,
            G
        )
    end
end

state(P::QuantumStatePadeIntegrator) = P.state_symb
controls(P::QuantumStatePadeIntegrator) = P.drive_symb

function nth_order_pade(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Ũₜ₊₁ = iso_vec_to_iso_operator(Ũ⃗ₜ₊₁)
    Ũₜ = iso_vec_to_iso_operator(Ũ⃗ₜ)
    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ, P.G_drift, P.G_drives)
    n = P.order ÷ 2
    Gₜ_powers = compute_powers(Gₜ, n)
    B = P.I_2N + sum([
        (-1)^k * PADE_COEFFICIENTS[P.order][k] * Δt^k * Gₜ_powers[k]
            for k = 1:n
    ])
    F = P.I_2N + sum([
        PADE_COEFFICIENTS[P.order][k] * Δt^k * Gₜ_powers[k]
            for k = 1:n
    ])
    δŨ = B * Ũₜ₊₁ - F * Ũₜ
    return iso_operator_to_iso_vec(δŨ)
end

@views function(P::UnitaryPadeIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]
    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end
    if P.drive_symb isa Tuple
        aₜ = vcat([zₜ[traj.components[s]] for s in P.drive_symb]...)
    else
        aₜ = zₜ[traj.components[P.drive_symb]]
    end
    return nth_order_pade(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
end



function nth_order_pade(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ, P.G_drift, P.G_drives)
    n = P.order ÷ 2
    Gₜ_powers = compute_powers(Gₜ, n)
    B = P.I_2N + sum([
        (-1)^k * PADE_COEFFICIENTS[P.order][k] * Δt^k * Gₜ_powers[k]
            for k = 1:n
    ])
    F = P.I_2N + sum([
        PADE_COEFFICIENTS[P.order][k] * Δt^k * Gₜ_powers[k]
            for k = 1:n
    ])
    δψ̃ = B * ψ̃ₜ₊₁ - F * ψ̃ₜ
    return δψ̃
end


@views function(P::QuantumStatePadeIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    ψ̃ₜ₊₁ = zₜ₊₁[traj.components[P.state_symb]]
    ψ̃ₜ = zₜ[traj.components[P.state_symb]]
    if P.drive_symb isa Tuple
        aₜ = vcat([zₜ[traj.components[s]] for s in P.drive_symb]...)
    else
        aₜ = zₜ[traj.components[P.drive_symb]]
    end
    if traj.timestep isa Symbol
        Δtₜ = zₜ[traj.components[traj.timestep]][1]
    else
        Δtₜ = traj.timestep
    end
    return nth_order_pade(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)
end

# aₜ should be a vector with all the controls. concatenate all the named traj controls
function ∂aₜ(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector{T},
    Ũ⃗ₜ::AbstractVector{T},
    aₜ::AbstractVector{T},
    Δtₜ::Real,
) where T <: Real

    if P.autodiff || !isnothing(P.G)

        # then we need to use the nth_order_pade function
        # which handles nonlinear G and higher order Pade integrators

        f(a) = nth_order_pade(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, a, Δtₜ)
        ∂aP = ForwardDiff.jacobian(f, aₜ)

    # otherwise we don't have a nonlinear G or are fine with using
    # the fourth order derivatives

    elseif P.order == 4
        n_drives = length(aₜ)
        ∂aP = Array{T}(undef, P.dim, n_drives)
        isodim = 2*P.N
        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Gʲ_anticomm_Gₜ =
                G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            for i = 0:P.N-1
                ψ̃ⁱₜ₊₁ = @view Ũ⃗ₜ₊₁[i * isodim .+ (1:isodim)]
                ψ̃ⁱₜ = @view Ũ⃗ₜ[i * isodim .+ (1:isodim)]
                ∂aP[i*isodim .+ (1:isodim), j] =
                    -Δtₜ / 2 * Gʲ * (ψ̃ⁱₜ₊₁ + ψ̃ⁱₜ) +
                    Δtₜ^2 / 12 * Gʲ_anticomm_Gₜ * (ψ̃ⁱₜ₊₁ - ψ̃ⁱₜ)
            end
        end
    else
        ## higher order pade code goes here
    end
    return ∂aP
end


function ∂aₜ(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector{T},
    ψ̃ₜ::AbstractVector{T},
    aₜ::AbstractVector{T},
    Δtₜ::Real,
) where T <: Real
    if P.autodiff || !isnothing(P.G)

        # then we need to use the nth_order_pade function
        # which handles nonlinear G and higher order Pade integrators

        f(a) = nth_order_pade(P, ψ̃ₜ₊₁, ψ̃ₜ, a, Δtₜ)
        ∂aP = ForwardDiff.jacobian(f, aₜ)

    # otherwise we don't have a nonlinear G or are fine with using
    # the fourth order derivatives

    elseif P.order == 4
        n_drives = length(aₜ)
        ∂aP = zeros(P.dim, n_drives)
        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Gʲ_anticomm_Gₜ =
                G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            ∂aP[:, j] =
                -Δtₜ / 2 * Gʲ * (ψ̃ₜ₊₁ + ψ̃ₜ) +
                Δtₜ^2 / 12 * Gʲ_anticomm_Gₜ * (ψ̃ₜ₊₁ - ψ̃ₜ)
        end
    else
        ### code for arbitrary Pade goes here
    end
    return ∂aP
end




function ∂Δtₜ(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)

    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ)
    Ũₜ₊₁ = iso_vec_to_iso_operator(Ũ⃗ₜ₊₁)
    Ũₜ = iso_vec_to_iso_operator(Ũ⃗ₜ)
    if P.order == 4
        ∂ΔtₜP_operator = -1/2 * Gₜ * (Ũₜ₊₁ + Ũₜ) + 1/6 * Δtₜ * Gₜ^2 * (Ũₜ₊₁ - Ũₜ)
        ∂ΔtₜP = iso_operator_to_iso_vec(∂ΔtₜP_operator)
    else
        n = P.order ÷ 2
        Gₜ_powers = compute_powers(Gₜ, n)
        B = sum([
            (-1)^k * k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^(k-1) * Gₜ_powers[k]
                for k = 1:n
        ])
        F = sum([
            k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^(k-1) * Gₜ_powers[k]
                for k = 1:n
        ])
        ∂ΔtₜP_operator = B * Ũₜ₊₁ - F * Ũₜ
        ∂ΔtₜP = iso_operator_to_iso_vec(∂ΔtₜP_operator)
    end

    return ∂ΔtₜP
end

function ∂Δtₜ(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)

    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ)
    if P.order==4
        ∂ΔtₜP = -1/2 * Gₜ * (ψ̃ₜ₊₁ + ψ̃ₜ) + 1/6 * Δtₜ * Gₜ^2 * (ψ̃ₜ₊₁ - ψ̃ₜ)
    else
        n = P.order ÷ 2
        Gₜ_powers = [Gₜ^i for i in 1:n]
        B = sum([(-1)^k * k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^(k-1) * Gₜ_powers[k] for k = 1:n])
        F = sum([k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^(k-1) * Gₜ_powers[k] for k = 1:n])
        ∂ΔtₜP = B*ψ̃ₜ₊₁ - F*ψ̃ₜ
    end
    return ∂ΔtₜP
end


@views function jacobian(
    P::UnitaryPadeIntegrator,
    zₜ::AbstractVector{T},
    zₜ₊₁::AbstractVector{T},
    traj::NamedTrajectory
) where T <: Number
    free_time = traj.timestep isa Symbol

    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_symb isa Tuple
        inds = [traj.components[s] for s in P.drive_symb]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_symb]
    end

    for i = 1:length(inds) - 1
        @assert inds[i] + 1 == inds[i + 1] "Controls must be in order"
    end

    aₜ = zₜ[inds]
    ∂aₜP = ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
    if free_time
        ∂ΔtₜP = ∂Δtₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
    end

    ∂Ũ⃗ₜP = spzeros(T, P.dim, P.dim)
    ∂Ũ⃗ₜ₊₁P = spzeros(T, P.dim, P.dim)
    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ, P.G_drift, P.G_drives)
    n = P.order ÷ 2

    # can memoize this chunk of code, prly memoize G powers
    Gₜ_powers = compute_powers(Gₜ, n)
    B = P.I_2N + sum([(-1)^k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^k * Gₜ_powers[k] for k = 1:n])
    F = P.I_2N + sum([PADE_COEFFICIENTS[P.order][k] * Δtₜ^k * Gₜ_powers[k] for k = 1:n])

    ∂Ũ⃗ₜ₊₁P = blockdiag(fill(sparse(B), P.N)...)
    ∂Ũ⃗ₜP = blockdiag(fill(sparse(-F), P.N)...)

    if free_time
        return ∂Ũ⃗ₜP, ∂Ũ⃗ₜ₊₁P, ∂aₜP, ∂ΔtₜP
    else
        return ∂Ũ⃗ₜP, ∂Ũ⃗ₜ₊₁P, ∂aₜP
    end
end

@views function jacobian(
    P::QuantumStatePadeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    traj::NamedTrajectory
)
    free_time = traj.timestep isa Symbol

    ψ̃ₜ₊₁ = zₜ₊₁[traj.components[P.state_symb]]
    ψ̃ₜ = zₜ[traj.components[P.state_symb]]


    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_symb isa Tuple
        inds = [traj.components[s] for s in P.drive_symb]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_symb]
    end

    for i = 1:length(inds) - 1
        @assert inds[i] + 1 == inds[i + 1] "Controls must be in order"
    end

    aₜ = zₜ[inds]

    ∂aₜP = ∂aₜ(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)
    if free_time
        ∂ΔtₜP = ∂Δtₜ(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)
    end

    Gₜ = isnothing(P.G) ? G(aₜ, P.G_drift, P.G_drives) : P.G(aₜ, P.G_drift, P.G_drives)
    n = P.order ÷ 2
    Gₜ_powers = compute_powers(Gₜ, n)
    B = P.I_2N + sum([(-1)^k * PADE_COEFFICIENTS[P.order][k] * Δtₜ^k * Gₜ_powers[k] for k = 1:n])
    F = P.I_2N + sum([PADE_COEFFICIENTS[P.order][k] * Δtₜ^k * Gₜ_powers[k] for k = 1:n])

    ∂ψ̃ₜP = -F
    ∂ψ̃ₜ₊₁P = B

    if free_time
        return ∂ψ̃ₜP, ∂ψ̃ₜ₊₁P, ∂aₜP, ∂ΔtₜP
    else
        return ∂ψ̃ₜP, ∂ψ̃ₜ₊₁P, ∂aₜP
    end
end


# ---------------------------------------
# Hessian of the Lagrangian
# ---------------------------------------

#calculate a deriv first and then indexing game
function μ∂aₜ∂Ũ⃗ₜ(
    P::UnitaryPadeIntegrator,
    aₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives

    if P.autodiff || !isnothing(P.G)

    elseif P.order == 4
        μ∂aₜ∂Ũ⃗ₜP = Array{T}(undef, P.dim, n_drives)

        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            ∂aₜ∂Ũ⃗ₜ_block_i = -(Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ)
            # sparse is necessary since blockdiag doesn't accept dense matrices
            ∂aₜ∂Ũ⃗ₜ = blockdiag(fill(sparse(∂aₜ∂Ũ⃗ₜ_block_i), P.N)...)
            μ∂aₜ∂Ũ⃗ₜP[:, j] = ∂aₜ∂Ũ⃗ₜ' * μₜ
        end
    else
        ## higher order pade goes here
    end
    return μ∂aₜ∂Ũ⃗ₜP
end

function μ∂Ũ⃗ₜ₊₁∂aₜ(
    P::UnitaryPadeIntegrator,
    aₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂Ũ⃗ₜ₊₁∂aₜP = zeros(T, n_drives, P.dim)

    for j = 1:n_drives
        Gʲ = P.G_drives[j]
        Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
        ∂Ũ⃗ₜ₊₁∂aₜ_block_i = -Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ
        # sparse is necessary since blockdiag doesn't accept dense matrices
        ∂Ũ⃗ₜ₊₁∂aₜ = blockdiag(fill(sparse(∂Ũ⃗ₜ₊₁∂aₜ_block_i), P.N)...)
        μ∂Ũ⃗ₜ₊₁∂aₜP[j, :] = μₜ' * ∂Ũ⃗ₜ₊₁∂aₜ
    end

    return μ∂Ũ⃗ₜ₊₁∂aₜP
end

function μ∂aₜ∂ψ̃ₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂aₜ∂ψ̃ₜP = zeros(T, P.dim, n_drives)

    for j = 1:n_drives
        Gʲ = P.G_drives[j]
        Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
        ∂aₜ∂ψ̃ₜP = -(Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ)
        μ∂aₜ∂ψ̃ₜP[:, j] = ∂aₜ∂ψ̃ₜP' * μₜ
    end

    return μ∂aₜ∂ψ̃ₜP
end

function μ∂ψ̃ₜ₊₁∂aₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂ψ̃ₜ₊₁∂aₜP = zeros(T, n_drives, P.dim)

    for j = 1:n_drives
        Gʲ = P.G_drives[j]
        Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
        ∂ψ̃ₜ₊₁∂aₜP = -Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ
        μ∂ψ̃ₜ₊₁∂aₜP[j, :] = μₜ' * ∂ψ̃ₜ₊₁∂aₜP
    end

    #can add if else for higher order derivatives
    return μ∂ψ̃ₜ₊₁∂aₜP
end

function μ∂²aₜ(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector{T},
    Ũ⃗ₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂²aₜP = zeros(T, n_drives, n_drives)

    if P.order==4
        for i = 1:n_drives
            for j = 1:i
                ∂aʲ∂aⁱP_block =
                    Δtₜ^2 / 12 * P.G_drive_anticomms[i, j]
                ∂aʲ∂aⁱP = blockdiag(fill(sparse(∂aʲ∂aⁱP_block), P.N)...)
                μ∂²aₜP[j, i] = dot(μₜ, ∂aʲ∂aⁱP*(Ũ⃗ₜ₊₁ - Ũ⃗ₜ))
            end
        end
    end

    return Symmetric(μ∂²aₜP)
end

function μ∂²aₜ(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector{T},
    ψ̃ₜ::AbstractVector{T},
    Δtₜ::Real,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = length(drive_indices)
    μ∂²aₜP = Array{T}(undef, n_drives, n_drives)

    if P.order==4
        for i = 1:n_drives
            for j = 1:i
                ∂aʲ∂aⁱP = Δtₜ^2 / 12 * P.G_drive_anticomms[i, j] * (ψ̃ₜ₊₁ - ψ̃ₜ)
                μ∂²aₜP[j, i] = dot(μₜ, ∂aʲ∂aⁱP)
            end
        end
    end

    return Symmetric(μ∂²aₜP)
end

function μ∂Δtₜ∂aₜ(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector{T},
    Ũ⃗ₜ::AbstractVector{T},
    aₜ::AbstractVector{T},
    Δtₜ::T,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂Δtₜ∂aₜP = Array{T}(undef, n_drives)

    if P.order == 4
        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            B = blockdiag(fill(sparse(-1/2 * Gʲ + 1/6 * Δtₜ * Ĝʲ), P.N)...)
            F = blockdiag(fill(sparse(1/2 * Gʲ + 1/6 * Δtₜ * Ĝʲ), P.N)...)
            ∂Δtₜ∂aₜ_j =  B*Ũ⃗ₜ₊₁ - F*Ũ⃗ₜ
            μ∂Δtₜ∂aₜP[j] = dot(μₜ, ∂Δtₜ∂aₜ_j)
        end
    end
    return μ∂Δtₜ∂aₜP
end

function μ∂Δtₜ∂aₜ(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector{T},
    ψ̃ₜ::AbstractVector{T},
    aₜ::AbstractVector{T},
    Δtₜ::T,
    μₜ::AbstractVector{T},
) where T <: Real

    n_drives = P.n_drives
    μ∂Δtₜ∂aₜP = Array{T}(undef, n_drives)

    if P.order == 4
        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Ĝʲ = G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            ∂Δt∂aʲP =
                -1 / 2 * Gʲ * (ψ̃ₜ₊₁ + ψ̃ₜ) +
                1 / 6 * Δtₜ * Ĝʲ * (ψ̃ₜ₊₁ - ψ̃ₜ)
            μ∂Δtₜ∂aₜP[j] = dot(μₜ, ∂Δt∂aʲP)
        end
    end
    return μ∂Δtₜ∂aₜP
end

function μ∂Δtₜ∂Ũ⃗ₜ(
    P::UnitaryPadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    minus_F = -(1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2)
    big_minus_F = blockdiag(fill(sparse(minus_F), P.N)...)
    return big_minus_F' * μₜ
end

function μ∂Ũ⃗ₜ₊₁∂Δtₜ(
    P::UnitaryPadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    B = -1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2
    big_B = blockdiag(fill(sparse(B), P.N)...)
    return μₜ' * big_B
end

function μ∂Δtₜ∂ψ̃ₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    # memoize the calc here
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    minus_F = -(1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2)
    return minus_F' * μₜ
end

function μ∂ψ̃ₜ₊₁∂Δtₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    B = -1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2
    return μₜ' * B
end

function μ∂²Δtₜ(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    μₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    ∂²Δtₜ_gen_block = 1/6 * Gₜ^2
    ∂²Δtₜ_gen = blockdiag(fill(sparse(∂²Δtₜ_gen_block), P.N)...)
    ∂²Δtₜ = ∂²Δtₜ_gen * (Ũ⃗ₜ₊₁ -  Ũ⃗ₜ)
    return μₜ' * ∂²Δtₜ
end

function μ∂²Δtₜ(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    μₜ::AbstractVector
)
    Gₜ = G(aₜ, P.G_drift, P.G_drives)
    ∂²Δtₜ = 1/6 * Gₜ^2 * (ψ̃ₜ₊₁ - ψ̃ₜ)
    return μₜ' * ∂²Δtₜ
end

@views function hessian_of_the_lagrangian(
    P::UnitaryPadeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    μₜ::AbstractVector,
    traj::NamedTrajectory
)
    free_time = traj.timestep isa Symbol

    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_symb]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_symb]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_symb isa Tuple
        inds = [traj.components[s] for s in P.drive_symb]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_symb]
    end

    aₜ = zₜ[inds]

    μ∂aₜ∂Ũ⃗ₜP = μ∂aₜ∂Ũ⃗ₜ(P, aₜ, Δtₜ, μₜ)
    μ∂²aₜP = μ∂²aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, Δtₜ, μₜ)
    if free_time
        μ∂Δtₜ∂aₜP = μ∂Δtₜ∂aₜ(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ, μₜ)
    end

    μ∂Ũ⃗ₜ₊₁∂aₜP = μ∂Ũ⃗ₜ₊₁∂aₜ(P, aₜ, Δtₜ, μₜ)

    if free_time
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
    else
        return (
            μ∂aₜ∂Ũ⃗ₜP,
            μ∂²aₜP,
            μ∂Ũ⃗ₜ₊₁∂aₜP
        )
    end
end

@views function hessian_of_the_lagrangian(
    P::QuantumStatePadeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    μₜ::AbstractVector,
    traj::NamedTrajectory
)
    free_time = traj.timestep isa Symbol

    ψ̃ₜ₊₁ = zₜ₊₁[traj.components[P.state_symb]]
    ψ̃ₜ = zₜ[traj.components[P.state_symb]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_symb isa Tuple
        inds = [traj.components[s] for s in P.drive_symb]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_symb]
    end

    aₜ = zₜ[inds]

    μ∂aₜ∂ψ̃ₜP = μ∂aₜ∂ψ̃ₜ(P, aₜ, Δtₜ, μₜ)
    μ∂²aₜP = μ∂²aₜ(P, ψ̃ₜ₊₁, ψ̃ₜ, Δtₜ, μₜ)
    if free_time
        μ∂Δtₜ∂aₜP = μ∂Δtₜ∂aₜ(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ, μₜ)
    end
    μ∂ψ̃ₜ₊₁∂aₜP = μ∂ψ̃ₜ₊₁∂aₜ(P, aₜ, Δtₜ, μₜ)

    if free_time
        μ∂Δtₜ∂ψ̃ₜP = μ∂Δtₜ∂ψ̃ₜ(P, aₜ, Δtₜ, μₜ)
        μ∂²ΔtₜP = μ∂²Δtₜ(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, μₜ)
        μ∂ψ̃ₜ₊₁∂ΔtₜP = μ∂ψ̃ₜ₊₁∂Δtₜ(P, aₜ, Δtₜ, μₜ)

        return (
            μ∂aₜ∂ψ̃ₜP,
            μ∂²aₜP,
            μ∂Δtₜ∂ψ̃ₜP,
            μ∂Δtₜ∂aₜP,
            μ∂²ΔtₜP,
            μ∂ψ̃ₜ₊₁∂aₜP,
            μ∂ψ̃ₜ₊₁∂ΔtₜP
        )
    else
        return (
            μ∂aₜ∂ψ̃ₜP,
            μ∂²aₜP,
            μ∂ψ̃ₜ₊₁∂aₜP
        )
    end
end


end
