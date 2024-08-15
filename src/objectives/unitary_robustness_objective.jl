export UnitaryRobustnessObjective
export PairwiseUnitaryRobustnessObjective

### 
### UnitaryRobustnessObjective
###

@doc raw"""
UnitaryRobustnessObjective(;
    H::::Union{OperatorType, Nothing}=nothing,
    eval_hessian::Bool=false,
    symb::Symbol=:Ũ⃗
)

Create a control objective which penalizes the sensitivity of the infidelity to the provided
operator defined in the subspace of the control dynamics, thereby realizing robust control.

The control dynamics are
```math
U_C(a)= \prod_t \exp{-i H_C(a_t)}
```

In the control frame, the H operator is (proportional to)
```math
R_{Robust}(a) = \frac{1}{T \norm{H_e}_2} \sum_t U_C(a_t)^\dag H_e U_C(a_t) \Delta t
```
where we have adjusted to a unitless expression of the operator.

The robustness objective is
```math
R_{Robust}(a) = \frac{1}{N} \norm{R}^2_F
```
where N is the dimension of the Hilbert space.
"""
function UnitaryRobustnessObjective(;
    H_error::Union{OperatorType, Nothing}=nothing,
    eval_hessian::Bool=false,
    symb::Symbol=:Ũ⃗
)
    @assert !isnothing(H_error) "H_error must be specified"

    # Indices of all non-zero subspace components for iso_vec_operators
    if H_error isa EmbeddedOperator
        H = unembed(H_error)
        subspace = get_iso_vec_subspace_indices(H_error)
    else
        H = H_error
        subspace = 1:length(operator_to_iso_vec(H))
    end

    @views function get_timesteps(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        return map(1:Z.T) do t
            if Z.timestep isa Symbol
                Z⃗[slice(t, Z.components[Z.timestep], Z.dim)][1]
            else
                Z.timestep
            end
        end
    end

    # Control frame
    @views function rotate(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        Δts = get_timesteps(Z⃗, Z)
        T = sum(Δts)
        Z_comps = Z.components[symb][subspace]
        R = sum(
            map(1:Z.T) do t
                Uₜ = iso_vec_to_operator(Z⃗[slice(t, Z_comps, Z.dim)])
                Uₜ'H*Uₜ .* Δts[t]
            end
        ) / norm(H) / T
        return R
    end

    function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        R = rotate(Z⃗, Z)
        return real(tr(R'R)) / size(R, 1)
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(eltype(Z⃗), Z.dim * Z.T + Z.global_dim)
        R = rotate(Z⃗, Z)
        Δts = get_timesteps(Z⃗, Z)
        Z_comps = Z.components[symb][subspace]
        T = sum(Δts)
        units = 1 / norm(H) / T
        Threads.@threads for t ∈ 1:Z.T
            # State
            Uₜ_slice = slice(t, Z_comps, Z.dim)
            Uₜ = iso_vec_to_operator(Z⃗[Uₜ_slice])

            # State gradient
            ∇[Uₜ_slice] .= operator_to_iso_vec(2 * H * Uₜ * R * Δts[t]) * units

            # Time gradient
            if Z.timestep isa Symbol
                ∂R = Uₜ'H*Uₜ
                ∇[slice(t, Z.components[Z.timestep], Z.dim)] .= real(tr(∂R*R + R*∂R)) * units
            end
        end
        return ∇ / size(R, 1)
    end

    # Hessian is dense (Control frame R contains sum over all unitaries).
    if eval_hessian
        # TODO
		∂²L = (Z⃗, Z) -> []
		∂²L_structure = Z -> []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

    params = Dict(
        :type => :UnitaryRobustnessObjective,
        :H_error => H_error,
        :eval_hessian => eval_hessian,
        :symb => symb
    )

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end

### 
### PairwiseUnitaryRobustnessObjective
###

"""
    PairwiseUnitaryRobustnessObjective(;
        H1::Union{OperatorType, Nothing}=nothing,
        H2_error::Union{OperatorType, Nothing}=nothing,
        symb1::Symbol=:Ũ⃗1,
        symb2::Symbol=:Ũ⃗2,
        eval_hessian::Bool=false,
    )

Create a control objective which penalizes the sensitivity of the infidelity to the provided operators
defined in the subspaces of the control dynamics, thereby realizing robust control.
"""
function PairwiseUnitaryRobustnessObjective(;
    H1_error::Union{OperatorType, Nothing}=nothing,
    H2_error::Union{OperatorType, Nothing}=nothing,
    symb1::Symbol=:Ũ⃗1,
    symb2::Symbol=:Ũ⃗2,
    eval_hessian::Bool=false,
)
    @assert !isnothing(H1_error) "H1_error must be specified"
    @assert !isnothing(H2_error) "H2_error must be specified"

    if H1_error isa EmbeddedOperator
        H1 = unembed(H1_error)
        subspace1 = get_iso_vec_subspace_indices(H1_error)
    else
        H1 = H1_error
        subspace1 = 1:length(operator_to_iso_vec(H1))
    end

    if H2_error isa EmbeddedOperator
        H2 = unembed(H2_error)
        subspace2 = get_iso_vec_subspace_indices(H2_error)
    else
        H2 = H2_error
        subspace2 = 1:length(operator_to_iso_vec(H2))
    end

    @views function get_timesteps(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        return map(1:Z.T) do t
            if Z.timestep isa Symbol
                Z⃗[slice(t, Z.components[Z.timestep], Z.dim)][1]
            else
                Z.timestep
            end
        end
    end

    function L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        Δts = get_timesteps(Z⃗, Z)
        Z1_comps = Z.components[symb1][subspace1]
        Z2_comps = Z.components[symb2][subspace2]
        T = sum(Δts)
        R = 0.0
        for (i₁, Δt₁) ∈ enumerate(Δts)
            for (i₂, Δt₂) ∈ enumerate(Δts)
                # States
                U1ₜ₁ = iso_vec_to_operator(Z⃗[slice(i₁, Z1_comps, Z.dim)])
                U1ₜ₂ = iso_vec_to_operator(Z⃗[slice(i₂, Z1_comps, Z.dim)])
                U2ₜ₁ = iso_vec_to_operator(Z⃗[slice(i₁, Z2_comps, Z.dim)])
                U2ₜ₂ = iso_vec_to_operator(Z⃗[slice(i₂, Z2_comps, Z.dim)])

                # Rotating frame
                rH1ₜ₁ = U1ₜ₁'H1*U1ₜ₁
                rH1ₜ₂ = U1ₜ₂'H1*U1ₜ₂
                rH2ₜ₁ = U2ₜ₁'H2*U2ₜ₁
                rH2ₜ₂ = U2ₜ₂'H2*U2ₜ₂

                # Robustness
                units = 1 / T^2 / norm(H1)^2 / norm(H2)^2
                R += real(tr(rH1ₜ₁'rH1ₜ₂) * tr(rH2ₜ₁'rH2ₜ₂) * Δt₁ * Δt₂ * units)
            end
        end
        return R / size(H1, 1) / size(H2, 1)
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real}, Z::NamedTrajectory)
        ∇ = zeros(Z.dim * Z.T + Z.global_dim)
        Δts = get_timesteps(Z⃗, Z)
        Z1_comps = Z.components[symb1][subspace1]
        Z2_comps = Z.components[symb2][subspace2]
        T = sum(Δts)
        Threads.@threads for (i₁, i₂) ∈ vec(collect(Iterators.product(1:Z.T, 1:Z.T)))
            # Times
            Δt₁ = Δts[i₁]
            Δt₂ = Δts[i₂]

            # States
            U1ₜ₁_slice = slice(i₁, Z1_comps, Z.dim)
            U1ₜ₂_slice = slice(i₂, Z1_comps, Z.dim)
            U2ₜ₁_slice = slice(i₁, Z2_comps, Z.dim)
            U2ₜ₂_slice = slice(i₂, Z2_comps, Z.dim)
            U1ₜ₁ = iso_vec_to_operator(Z⃗[U1ₜ₁_slice])
            U1ₜ₂ = iso_vec_to_operator(Z⃗[U1ₜ₂_slice])
            U2ₜ₁ = iso_vec_to_operator(Z⃗[U2ₜ₁_slice])
            U2ₜ₂ = iso_vec_to_operator(Z⃗[U2ₜ₂_slice])

            # Rotating frame
            rH1ₜ₁ = U1ₜ₁'H1*U1ₜ₁
            rH1ₜ₂ = U1ₜ₂'H1*U1ₜ₂
            rH2ₜ₁ = U2ₜ₁'H2*U2ₜ₁
            rH2ₜ₂ = U2ₜ₂'H2*U2ₜ₂

            # ∇Uiₜⱼ (assume H's are Hermitian)
            units = 1 / T^2 / norm(H1)^2 / norm(H2)^2
            R1 = tr(rH1ₜ₁'rH1ₜ₂) * Δt₁ * Δt₂ * units
            R2 = tr(rH2ₜ₁'rH2ₜ₂) * Δt₁ * Δt₂ * units
            ∇[U1ₜ₁_slice] += operator_to_iso_vec(2 * H1 * U1ₜ₁ * rH1ₜ₂) * R2
            ∇[U1ₜ₂_slice] += operator_to_iso_vec(2 * H1 * U1ₜ₂ * rH1ₜ₁) * R2
            ∇[U2ₜ₁_slice] += operator_to_iso_vec(2 * H2 * U2ₜ₁ * rH2ₜ₂) * R1
            ∇[U2ₜ₂_slice] += operator_to_iso_vec(2 * H2 * U2ₜ₂ * rH2ₜ₁) * R1

            # Time gradients
            if Z.timestep isa Symbol
                R = real(tr(rH1ₜ₁'rH1ₜ₂) * tr(rH2ₜ₁'rH2ₜ₂)) * units
                ∇[slice(i₁, Z.components[Z.timestep], Z.dim)] .= R * Δt₂
                ∇[slice(i₂, Z.components[Z.timestep], Z.dim)] .= R * Δt₁
            end
        end
        return ∇ / size(H1, 1) / size(H2, 1)
    end

    # Hessian is dense (Control frame R contains sum over all unitaries).
    if eval_hessian
        # TODO
		∂²L = (Z⃗, Z) -> []
		∂²L_structure = Z -> []
	else
		∂²L = nothing
		∂²L_structure = nothing
	end

    params = Dict(
        :type => :PairwiseUnitaryRobustnessObjective,
        :H1_error => H1_error,
        :H2_error => H2_error,
        :symb1 => symb1,
        :symb2 => symb2,
        :eval_hessian => eval_hessian
    )

    return Objective(L, ∇L, ∂²L, ∂²L_structure, Dict[params])
end