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
twelth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 12)
fourteenth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 14)
sixteenth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 16)
eighteenth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 18)
twentieth_order_pade(Gₜ::Matrix) = nth_order_pade(Gₜ, 20)

function compute_powers(G::AbstractMatrix{T}, order::Int) where T <: Number
    powers = Array{typeof(G)}(undef, order)
    powers[1] = G
    for k = 2:order
        powers[k] = powers[k-1] * G
    end
    return powers
end

# key is the order of the integrator
# and the value are the Pade coefficients
# for each term
const PADE_COEFFICIENTS = OrderedDict{Int,Vector{Float64}}(
    4 => [1/2, 1/12],
    6 => [1/2, 1/10, 1/120],
    8 => [1/2, 3/28, 1/84, 1/1680],
    10 => [1/2, 1/9, 1/72, 1/1008, 1/30240],
    12 => [1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280],
    14 => [1/2, 3/26, 5/312, 5/3432, 1/11440, 1/308880, 1/17297280],
    16 => [1/2, 7/60, 1/60, 1/624, 1/9360, 1/205920, 1/7207200, 1/518918400],
    18 => [1/2, 2/17, 7/408, 7/4080, 1/8160, 1/159120, 1/4455360, 1/196035840, 1/17643225600],
    20 => [1/2, 9/76, 1/57, 7/3876, 7/51680, 7/930240, 1/3255840, 1/112869120, 1/6094932480, 1/670442572800]
)

function pade_operator(
    Id::AbstractMatrix,
    G_powers::Vector{<:AbstractMatrix},
    coeffs::Vector{<:Real}
)
    return Id + sum(coeffs .* G_powers)
end

function forward_pade_coefficients(Δt::Real, pade_order::Int; timestep_derivative=false)
    n = pade_order ÷ 2
    if !timestep_derivative
        return PADE_COEFFICIENTS[2n] .* (Δt .^ (1:n))
    else
        return PADE_COEFFICIENTS[2n] .* (Δt .^ (0:n-1)) .* (1:n)
    end
end

function backward_pade_coefficients(Δt::Real, pade_order::Int; timestep_derivative=false)
    n = pade_order ÷ 2
    if !timestep_derivative
        return PADE_COEFFICIENTS[2n] .* ((-Δt) .^ (1:n))
    else
        return PADE_COEFFICIENTS[2n] .* ((-1) .^ (1:n)) .* (Δt .^ (0:n-1)) .* (1:n)
    end
end

function pade_coefficients(Δt::Real, pade_order::Int; timestep_derivative=false)
    F_coeffs = forward_pade_coefficients(Δt, pade_order;
        timestep_derivative=timestep_derivative
    )
    B_coeffs = backward_pade_coefficients(Δt, pade_order;
        timestep_derivative=timestep_derivative
    )
    return F_coeffs, B_coeffs
end

function backward_operator(
    G_powers::Vector{<:AbstractMatrix},
    Id::AbstractMatrix,
    Δt::Real;
    timestep_derivative=false
)
    pade_order = 2 * length(G_powers)
    coeffs = backward_pade_coefficients(Δt, pade_order; timestep_derivative=timestep_derivative)
    return pade_operator(Id, G_powers, coeffs)
end

backward_operator(G::AbstractMatrix, pade_order::Int, args...; kwargs...) =
    backward_operator(compute_powers(G, pade_order ÷ 2), args...; kwargs...)

function forward_operator(
    G_powers::Vector{<:AbstractMatrix},
    Id::AbstractMatrix,
    Δt::Real;
    timestep_derivative=false
)
    pade_order = 2 * length(G_powers)
    coeffs = forward_pade_coefficients(Δt, pade_order; timestep_derivative=timestep_derivative)
    return pade_operator(Id, G_powers, coeffs)
end

forward_operator(G::AbstractMatrix, pade_order::Int, args...; kwargs...) =
    forward_operator(compute_powers(G, pade_order ÷ 2), args...; kwargs...)

function pade_operators(
    G_powers::Vector{<:AbstractMatrix},
    Id::AbstractMatrix,
    Δt::Real;
    kwargs...
)
    F = forward_operator(G_powers, Id, Δt; kwargs...)
    B = backward_operator(G_powers, Id, Δt; kwargs...)
    return F, B
end

function pade_operators(
    G_powers::Vector{<:SparseMatrixCSC},
    Id::SparseMatrixCSC,
    Δt::Real;
    kwargs...
)
    F = forward_operator(G_powers, Id, Δt; kwargs...)
    B = backward_operator(G_powers, Id, Δt; kwargs...)
    droptol!(F, 1e-12)
    droptol!(B, 1e-12)
    return F, B
end

pade_operators(G::AbstractMatrix, pade_order::Int, args...; kwargs...) =
    pade_operators(compute_powers(G, pade_order ÷ 2), args...; kwargs...)

@views function ∂aʲF(
    P::QuantumIntegrator,
    G_powers::Vector{<:AbstractMatrix},
    Δt::Real,
    ∂G_∂aʲ::AbstractMatrix
)
    F_coeffs = forward_pade_coefficients(Δt, P.order)
    ∂F_∂aʲ = zeros(size(G_powers[1]))
    n = length(G_powers)
    for p = 1:n
        if p == 1
            ∂F_∂aʲ += F_coeffs[p] * ∂G_∂aʲ
        else
            for k = 1:p
                if k == 1
                    ∂F_∂aʲ += F_coeffs[p] * ∂G_∂aʲ * G_powers[p-1]
                elseif k == p
                    ∂F_∂aʲ += F_coeffs[p] * G_powers[p-1] * ∂G_∂aʲ
                else
                    ∂F_∂aʲ += F_coeffs[p] * G_powers[k-1] * ∂G_∂aʲ * G_powers[p-k]
                end
            end
        end
    end
    return ∂F_∂aʲ
end

@views function ∂aʲB(
    P::QuantumIntegrator,
    G_powers::Vector{<:AbstractMatrix},
    Δt::Real,
    ∂G_∂aʲ::AbstractMatrix
)
    B_coeffs = backward_pade_coefficients(Δt, P.order)
    ∂B_∂aʲ = zeros(size(G_powers[1]))
    for p = 1:(P.order ÷ 2)
        if p == 1
            ∂B_∂aʲ += B_coeffs[p] * ∂G_∂aʲ
        else
            for k = 1:p
                if k == 1
                    ∂B_∂aʲ += B_coeffs[p] * ∂G_∂aʲ * G_powers[p-1]
                elseif k == p
                    ∂B_∂aʲ += B_coeffs[p] * G_powers[p-1] * ∂G_∂aʲ
                else
                    ∂B_∂aʲ += B_coeffs[p] * G_powers[k-1] * ∂G_∂aʲ * G_powers[p-k]
                end
            end
        end
    end
    return ∂B_∂aʲ
end



# ----------------------------------------------------------------
#                       Quantum Pade Integrator
# ----------------------------------------------------------------



abstract type QuantumPadeIntegrator <: QuantumIntegrator end



# ----------------------------------------------------------------
#                       Unitary Pade Integrator
# ----------------------------------------------------------------


"""
"""
struct UnitaryPadeIntegrator <: QuantumPadeIntegrator
    unitary_components::Vector{Int}
    drive_components::Vector{Int}
    timestep::Union{Real, Int} # either the timestep or the index of the timestep
    freetime::Bool
    n_drives::Int
    ketdim::Int
    dim::Int
    zdim::Int
    order::Int
    autodiff::Bool
    G::Function
    ∂G::Function

    @doc raw"""
        UnitaryPadeIntegrator(
            unitary_name::Symbol,
            drive_name::Union{Symbol,Tuple{Vararg{Symbol}}},
            G::Function,
            ∂G::Function,
            traj::NamedTrajectory;
            order::Int=4,
            calculate_pade_operators_structure::Bool=true,
            autodiff::Bool=false
        )

    Construct a `UnitaryPadeIntegrator` which computes

    ```math
    \text{isovec}\qty(B^{(n)}(a_t) U_{t+1} - F^{(n)}(a_t) U_t)
    ```

    where `U_t` is the unitary at time `t`, `a_t` is the control at time `t`, and `B^{(n)}(a_t)` and `F^{(n)}(a_t)` are the `n`th order Pade operators of the exponential of the drift operator `G(a_t)`.

    # Arguments
    - `unitary_name::Symbol`: the name of the unitary in the trajectory
    - `drive_name::Union{Symbol,Tuple{Vararg{Symbol}}}`: the name of the drive(s) in the trajectory
    - `G::Function`: a function which takes the control vector `a_t` and returns the drive `G(a_t)`, $G(a_t) = \text{iso}(-i H(a_t))$
    - `∂G::Function`: a function which takes the control vector `a_t` and returns a vector of matrices $\qty(\ldots, \pdv{G}{a^j_t}, \ldots)$
    - `traj::NamedTrajectory`: the trajectory

    # Keyword Arguments
    - `order::Int=4`: the order of the Pade approximation. Must be in `[4, 6, 8, 10, 12, 14, 16, 18, 20]`.

    """
    function UnitaryPadeIntegrator(
        unitary_name::Symbol,
        drive_name::Union{Symbol,Tuple{Vararg{Symbol}}},
        G::Function,
        ∂G::Function,
        traj::NamedTrajectory;
        order::Int=4,
        autodiff::Bool=false
    )
        @assert order ∈ keys(PADE_COEFFICIENTS) "order ∉ $(keys(PADE_COEFFICIENTS))"

        dim = traj.dims[unitary_name]
        ketdim = Int(sqrt(dim ÷ 2))

        unitary_components = traj.components[unitary_name]

        if drive_name isa Tuple
            drive_components = vcat((traj.components[s] for s ∈ drive_name)...)
        else
            drive_components = traj.components[drive_name]
        end

        n_drives = length(drive_components)

        @assert all(diff(drive_components) .== 1) "controls must be in order"

        freetime = traj.timestep isa Symbol

        if freetime
            timestep = traj.components[traj.timestep][1]
        else
            timestep = traj.timestep
        end

        return new(
            unitary_components,
            drive_components,
            timestep,
            freetime,
            n_drives,
            ketdim,
            dim,
            traj.dim,
            order,
            autodiff,
            G,
            ∂G
        )
    end
end

function get_comps(P::UnitaryPadeIntegrator, traj::NamedTrajectory)
    if P.freetime
        return P.unitary_components, P.drive_components, traj.components[traj.timestep]
    else
        return P.unitary_components, P.drive_components
    end
end

function (integrator::UnitaryPadeIntegrator)(
    traj::NamedTrajectory;
    unitary_name::Union{Symbol, Nothing}=nothing,
    drive_name::Union{Symbol, Tuple{Vararg{Symbol}}, Nothing}=nothing,
    order::Int=integrator.order,
    G::Function=integrator.G,
    ∂G::Function=integrator.∂G,
    autodiff::Bool=integrator.autodiff
)
    @assert !isnothing(unitary_name) "unitary_name must be provided"
    @assert !isnothing(drive_name) "drive_name must be provided"
    return UnitaryPadeIntegrator(
        unitary_name,
        drive_name,
        G,
        ∂G,
        traj;
        order=order,
        autodiff=autodiff
    )
end

# ------------------- Integrator -------------------



function nth_order_pade(
    P::UnitaryPadeIntegrator,
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = P.G(aₜ)

    F, B = pade_operators(Gₜ, P.order, I(2P.ketdim), Δt)

    I_N = sparse(I, P.ketdim, P.ketdim)

    return (I_N ⊗ B) * Ũ⃗ₜ₊₁ - (I_N ⊗ F) * Ũ⃗ₜ
end

@views function(P::UnitaryPadeIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
)
    Ũ⃗ₜ₊₁ = zₜ₊₁[P.unitary_components]
    Ũ⃗ₜ = zₜ[P.unitary_components]
    aₜ = zₜ[P.drive_components]

    if P.freetime
        Δtₜ = zₜ[P.timestep]
    else
        Δtₜ = P.timestep
    end


    return nth_order_pade(P, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)
end

# ------------------- Jacobians -------------------

# aₜ should be a vector with all the controls. concatenate all the named traj controls
function ∂aₜ(
    P::UnitaryPadeIntegrator,
    G_powers::Vector{<:AbstractMatrix},
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    ∂aP = zeros(eltype(Ũ⃗ₜ), P.dim, P.n_drives)

    ∂G_∂aₜ = P.∂G(aₜ)

    I_N = sparse(I, P.ketdim, P.ketdim)

    for j = 1:P.n_drives

        # TODO: maybe rework for arbitrary drive indices eventually

        ∂aₜʲF = ∂aʲF(P, G_powers, Δtₜ, ∂G_∂aₜ[j])
        ∂aₜʲB = ∂aʲB(P, G_powers, Δtₜ, ∂G_∂aₜ[j])

        ∂aP[:, j] = (I_N ⊗ ∂aₜʲB) * Ũ⃗ₜ₊₁ - (I_N ⊗ ∂aₜʲF) * Ũ⃗ₜ
    end

    return ∂aP
end


function ∂Δtₜ(
    P::UnitaryPadeIntegrator,
    Gₜ_powers::Vector{<:AbstractMatrix},
    Ũ⃗ₜ₊₁::AbstractVector,
    Ũ⃗ₜ::AbstractVector,
    Δtₜ::Real
)
    ∂ΔtₜF_coeffs, ∂ΔtₜB_coeffs = pade_coefficients(Δtₜ, P.order;
        timestep_derivative=true
    )

    ∂ΔtₜF = sum(∂ΔtₜF_coeffs .* Gₜ_powers)
    ∂ΔtₜB = sum(∂ΔtₜB_coeffs .* Gₜ_powers)

    I_N = sparse(I, P.ketdim, P.ketdim)

    return (I_N ⊗ ∂ΔtₜB) * Ũ⃗ₜ₊₁ - (I_N ⊗ ∂ΔtₜF) * Ũ⃗ₜ
end

@views function jacobian(
    P::UnitaryPadeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
)
    # obtain state and control vectors
    Ũ⃗ₜ₊₁ = zₜ₊₁[P.unitary_components]
    Ũ⃗ₜ = zₜ[P.unitary_components]
    aₜ = zₜ[P.drive_components]

    Gₜ = P.G(aₜ)

    # obtain timestep
    if P.freetime
        Δtₜ = zₜ[P.timestep]
    else
        Δtₜ = P.timestep
    end

    Gₜ_powers = compute_powers(Gₜ, P.order ÷ 2)

    ∂aₜP = ∂aₜ(P, Gₜ_powers, Ũ⃗ₜ₊₁, Ũ⃗ₜ, aₜ, Δtₜ)

    Id = sparse(I, P.ketdim, P.ketdim)

    Fₜ, Bₜ = pade_operators(Gₜ_powers, I(2P.ketdim), Δtₜ)

    ∂Ũ⃗ₜP = -Id ⊗ Fₜ
    ∂Ũ⃗ₜ₊₁P = Id ⊗ Bₜ

    if P.freetime
        ∂ΔtₜP = ∂Δtₜ(P, Gₜ_powers, Ũ⃗ₜ₊₁, Ũ⃗ₜ, Δtₜ)
        return ∂Ũ⃗ₜP, ∂Ũ⃗ₜ₊₁P, ∂aₜP, ∂ΔtₜP
    else
        return ∂Ũ⃗ₜP, ∂Ũ⃗ₜ₊₁P, ∂aₜP
    end
end
# ----------------------------------------------------------------
#                  Quantum State Pade Integrator
# ----------------------------------------------------------------

struct QuantumStatePadeIntegrator <: QuantumPadeIntegrator
    state_components::Vector{Int}
    drive_components::Vector{Int}
    timestep::Union{Real, Int} # either the timestep or the index of the timestep
    freetime::Bool
    n_drives::Int
    ketdim::Int
    dim::Int
    zdim::Int
    order::Int
    autodiff::Bool
    G::Function
    ∂G::Function

    """
        QuantumStatePadeIntegrator(
            sys::AbstractQuantumSystem,
            state_name::Union{Symbol,Nothing}=nothing,
            drive_name::Union{Symbol,Tuple{Vararg{Symbol}},Nothing}=nothing,
            timestep_name::Union{Symbol,Nothing}=nothing;
            order::Int=4,
            autodiff::Bool=false
        ) where R <: Real

    Construct a `QuantumStatePadeIntegrator` for the quantum system `sys`.

    # Examples

    ## for a single drive `a`:
    ```julia
        P = QuantumStatePadeIntegrator(sys, :ψ̃, :a)
    ```

    ## for two drives `α` and `γ`, order `8`, and autodiffed:
    ```julia
        P = QuantumStatePadeIntegrator(sys, :ψ̃, (:α, :γ); order=8, autodiff=true)
    ```

    # Arguments
    - `sys::AbstractQuantumSystem`: the quantum system
    - `state_name::Symbol`: the nameol for the quantum state
    - `drive_name::Union{Symbol,Tuple{Vararg{Symbol}}}`: the nameol(s) for the drives
    - `order::Int=4`: the order of the Pade approximation. Must be in `[4, 6, 8, 10]`. If order is not `4` and `autodiff` is `false`, then the integrator will use the hand-coded fourth order derivatives.
    - `autodiff::Bool=false`: whether to use automatic differentiation to compute the jacobian and hessian of the lagrangian
    """
    function QuantumStatePadeIntegrator(
        state_name::Symbol,
        drive_name::Union{Symbol,Tuple{Vararg{Symbol}}},
        G::Function,
        ∂G::Function,
        traj::NamedTrajectory;
        order::Int=4,
        autodiff::Bool=false,
    )
        @assert order ∈ keys(PADE_COEFFICIENTS) "order ∉ $(keys(PADE_COEFFICIENTS))"

        dim = traj.dims[state_name]
        ketdim = dim ÷ 2

        state_components = traj.components[state_name]

        if drive_name isa Tuple
            drive_components = vcat((traj.components[s] for s ∈ drive_name)...)
        else
            drive_components = traj.components[drive_name]
        end

        n_drives = length(drive_components)

        @assert all(diff(drive_components) .== 1) "controls must be in order"

        freetime = traj.timestep isa Symbol

        if freetime
            timestep = traj.components[traj.timestep][1]
        else
            timestep = traj.timestep
        end


        return new(
            state_components,
            drive_components,
            timestep,
            freetime,
            n_drives,
            ketdim,
            dim,
            traj.dim,
            order,
            autodiff,
            G,
            ∂G
        )
    end
end

function get_comps(P::QuantumStatePadeIntegrator, traj::NamedTrajectory)
    if P.freetime
        return P.state_components, P.drive_components, traj.components[traj.timestep]
    else
        return P.state_components, P.drive_components
    end
end

function (integrator::QuantumStatePadeIntegrator)(
    traj::NamedTrajectory;
    state_name::Union{Symbol, Nothing}=nothing,
    drive_name::Union{Symbol, Tuple{Vararg{Symbol}}, Nothing}=nothing,
    order::Int=integrator.order,
    G::Function=integrator.G,
    ∂G::Function=integrator.∂G,
    autodiff::Bool=integrator.autodiff
)
    @assert !isnothing(state_name) "state_name must be provided"
    @assert !isnothing(drive_name) "drive_name must be provided"
    return QuantumStatePadeIntegrator(
        state_name,
        drive_name,
        G,
        ∂G,
        traj;
        order=order,
        autodiff=autodiff
    )
end

# ------------------- Integrator -------------------

function nth_order_pade(
    P::QuantumStatePadeIntegrator,
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δt::Real
)
    Gₜ = P.G(aₜ)

    F, B = pade_operators(Gₜ, P.order, I(2P.ketdim), Δt)

    return B * ψ̃ₜ₊₁ - F * ψ̃ₜ
end


@views function(P::QuantumStatePadeIntegrator)(
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
)
    ψ̃ₜ₊₁ = zₜ₊₁[P.state_components]
    ψ̃ₜ = zₜ[P.state_components]
    aₜ = zₜ[P.drive_components]

    if P.freetime
        Δtₜ = zₜ[P.timestep]
    else
        Δtₜ = P.timestep
    end
    return nth_order_pade(P, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)
end


function ∂aₜ(
    P::QuantumStatePadeIntegrator,
    G_powers::Vector{<:AbstractMatrix},
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    aₜ::AbstractVector,
    Δtₜ::Real
)
    ∂aP = zeros(eltype(ψ̃ₜ), P.dim, P.n_drives)

    ∂G_∂aₜ = P.∂G(aₜ)

    for j = 1:P.n_drives

        ∂aₜʲF = ∂aʲF(P, G_powers, Δtₜ, ∂G_∂aₜ[j])
        ∂aₜʲB = ∂aʲB(P, G_powers, Δtₜ, ∂G_∂aₜ[j])

        ∂aP[:, j] = ∂aₜʲB * ψ̃ₜ₊₁ - ∂aₜʲF * ψ̃ₜ
    end

    return ∂aP
end



function ∂Δtₜ(
    P::QuantumStatePadeIntegrator,
    Gₜ_powers::Vector{<:AbstractMatrix},
    ψ̃ₜ₊₁::AbstractVector,
    ψ̃ₜ::AbstractVector,
    Δtₜ::Real
)
    ∂ΔtₜF_coeffs, ∂ΔtₜB_coeffs = pade_coefficients(Δtₜ, P.order;
        timestep_derivative=true
    )

    ∂ΔtₜF = sum(∂ΔtₜF_coeffs .* Gₜ_powers)
    ∂ΔtₜB = sum(∂ΔtₜB_coeffs .* Gₜ_powers)

    return ∂ΔtₜB * ψ̃ₜ₊₁ - ∂ΔtₜF * ψ̃ₜ
end

@views function jacobian(
    P::QuantumStatePadeIntegrator,
    zₜ::AbstractVector,
    zₜ₊₁::AbstractVector,
    t::Int
)
    # obtain state and control vectors
    ψ̃ₜ₊₁ = zₜ₊₁[P.state_components]
    ψ̃ₜ = zₜ[P.state_components]
    aₜ = zₜ[P.drive_components]

    Gₜ = P.G(aₜ)

    # obtain timestep
    if P.freetime
        Δtₜ = zₜ[P.timestep]
    else
        Δtₜ = P.timestep
    end

    Gₜ_powers = compute_powers(Gₜ, P.order ÷ 2)

    ∂aₜP = ∂aₜ(P, Gₜ_powers, ψ̃ₜ₊₁, ψ̃ₜ, aₜ, Δtₜ)

    # jacobian wrt state
    Fₜ, Bₜ = pade_operators(Gₜ_powers, I(2P.ketdim), Δtₜ)

    ∂ψ̃ₜP = -Fₜ
    ∂ψ̃ₜ₊₁P = Bₜ

    if P.freetime
        ∂ΔtₜP = ∂Δtₜ(P, Gₜ_powers, ψ̃ₜ₊₁, ψ̃ₜ, Δtₜ)
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

    if P.autodiff

    elseif P.order == 4
        μ∂aₜ∂Ũ⃗ₜP = Array{T}(undef, P.dim, n_drives)

        for j = 1:n_drives
            Gʲ = P.G_drives[j]
            Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            ∂aₜ∂Ũ⃗ₜ_block_i = -(Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ)
            # sparse is necessary since blockdiag doesn't accept dense matrices
            ∂aₜ∂Ũ⃗ₜ = blockdiag(fill(sparse(∂aₜ∂Ũ⃗ₜ_block_i), P.ketdim)...)
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
        Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
        ∂Ũ⃗ₜ₊₁∂aₜ_block_i = -Δtₜ / 2 * Gʲ + Δtₜ^2 / 12 * Ĝʲ
        # sparse is necessary since blockdiag doesn't accept dense matrices
        ∂Ũ⃗ₜ₊₁∂aₜ = blockdiag(fill(sparse(∂Ũ⃗ₜ₊₁∂aₜ_block_i), P.ketdim)...)
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
        Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
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
        Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
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
                ∂aʲ∂aⁱP = blockdiag(fill(sparse(∂aʲ∂aⁱP_block), P.ketdim)...)
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

    n_drives = P.n_drives
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
            Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
            B = blockdiag(fill(sparse(-1/2 * Gʲ + 1/6 * Δtₜ * Ĝʲ), P.ketdim)...)
            F = blockdiag(fill(sparse(1/2 * Gʲ + 1/6 * Δtₜ * Ĝʲ), P.ketdim)...)
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
            Ĝʲ = P.G(aₜ, P.G_drift_anticomms[j], P.G_drive_anticomms[:, j])
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
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
    minus_F = -(1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2)
    big_minus_F = blockdiag(fill(sparse(minus_F), P.ketdim)...)
    return big_minus_F' * μₜ
end

function μ∂Ũ⃗ₜ₊₁∂Δtₜ(
    P::UnitaryPadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
    B = -1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2
    big_B = blockdiag(fill(sparse(B), P.ketdim)...)
    return μₜ' * big_B
end

function μ∂Δtₜ∂ψ̃ₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    # memoize the calc here
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
    minus_F = -(1/2 * Gₜ + 1/6 * Δtₜ * Gₜ^2)
    return minus_F' * μₜ
end

function μ∂ψ̃ₜ₊₁∂Δtₜ(
    P::QuantumStatePadeIntegrator,
    aₜ::AbstractVector,
    Δtₜ::Real,
    μₜ::AbstractVector
)
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
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
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
    ∂²Δtₜ_gen_block = 1/6 * Gₜ^2
    ∂²Δtₜ_gen = blockdiag(fill(sparse(∂²Δtₜ_gen_block), P.ketdim)...)
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
    Gₜ = P.G(aₜ, P.G_drift, P.G_drives)
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

    Ũ⃗ₜ₊₁ = zₜ₊₁[traj.components[P.unitary_name]]
    Ũ⃗ₜ = zₜ[traj.components[P.unitary_name]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_name isa Tuple
        inds = [traj.components[s] for s in P.drive_name]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_name]
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

    ψ̃ₜ₊₁ = zₜ₊₁[traj.components[P.state_name]]
    ψ̃ₜ = zₜ[traj.components[P.state_name]]

    Δtₜ = free_time ? zₜ[traj.components[traj.timestep]][1] : traj.timestep

    if P.drive_name isa Tuple
        inds = [traj.components[s] for s in P.drive_name]
        inds = vcat(collect.(inds)...)
    else
        inds = traj.components[P.drive_name]
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
