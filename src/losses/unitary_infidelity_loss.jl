export unitary_fidelity
export iso_vec_unitary_fidelity
export unitary_free_phase_fidelity
export iso_vec_unitary_free_phase_fidelity

export UnitaryInfidelityLoss
export UnitaryFreePhaseInfidelityLoss

###
### UnitaryInfidelityLoss
###

@doc raw"""
    unitary_fidelity(U::Matrix, U_goal::Matrix; kwargs...)
    unitary_fidelity(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector; kwargs...)

Calculate the fidelity between two unitary operators `U` and `U_goal`.

```math
\mathcal{F}(U, U_{\text{goal}}) = \frac{1}{n} \abs{\tr \qty(U_{\text{goal}}^\dagger U)}
```

where $n$ is the dimension of the unitary operators.

# Keyword Arguments
- `subspace::AbstractVector{Int}`: The subspace to calculate the fidelity over.
"""
@views function unitary_fidelity(
    U::Matrix,
    U_goal::Matrix;
    subspace::AbstractVector{Int}=axes(U_goal, 1)
)
    U_goal = U_goal[subspace, subspace]
    U = U[subspace, subspace]
    return 1 / size(U_goal, 1) * abs(tr(U_goal'U))
end


@doc raw"""
    iso_vec_unitary_fidelity(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)

Returns the fidelity between the isomorphic unitary vector $\vec{\widetilde{U}} \sim U \in SU(n)$
and the isomorphic goal unitary vector $\vec{\widetilde{U}}_{\text{goal}}$.

```math
\begin{aligned}
\mathcal{F}(\vec{\widetilde{U}}, \vec{\widetilde{U}}_{\text{goal}}) &= \frac{1}{n} \abs{\tr \qty(U_{\text{goal}}^\dagger U)} \\
&= \frac{1}{n} \sqrt{T_R^{2} + T_I^{2}}
\end{aligned}
```

where $T_R = \langle \vec{\widetilde{U}}_{\text{goal}, R}, \vec{\widetilde{U}}_R \rangle + \langle \vec{\widetilde{U}}_{\text{goal}, I}, \vec{\widetilde{U}}_I \rangle$ and $T_I = \langle \vec{\widetilde{U}}_{\text{goal}, R}, \vec{\widetilde{U}}_I \rangle - \langle \vec{\widetilde{U}}_{\text{goal}, I}, \vec{\widetilde{U}}_R \rangle$.

"""
@inline @views function iso_vec_unitary_fidelity(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    U = iso_vec_to_operator(Ũ⃗)[subspace, subspace]
    n = size(U, 1)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)[subspace, subspace]
    U⃗ᵣ, U⃗ᵢ = vec(real(U)), vec(imag(U))
    Ū⃗ᵣ, Ū⃗ᵢ = vec(real(U_goal)), vec(imag(U_goal))
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    return 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
end

@views function iso_vec_unitary_infidelity(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    ℱ = iso_vec_unitary_fidelity(Ũ⃗, Ũ⃗_goal, subspace=subspace)
    return abs(1 - ℱ)
end

@views function iso_vec_unitary_infidelity_gradient(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    U = iso_vec_to_operator(Ũ⃗)[subspace, subspace]
    n = size(U, 1)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)[subspace, subspace]
    U⃗ᵣ, U⃗ᵢ = vec(real(U)), vec(imag(U))
    Ū⃗ᵣ, Ū⃗ᵢ = vec(real(U_goal)), vec(imag(U_goal))
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    ℱ = 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
    ∇ᵣℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵣ - Tᵢ * Ū⃗ᵢ)
    ∇ᵢℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵢ + Tᵢ * Ū⃗ᵣ)
    ∇ℱ = [∇ᵣℱ; ∇ᵢℱ]
    permutation = vcat(vcat([[slice(j, n), slice(j, n) .+ n^2] for j = 1:n]...)...)
    ∇ℱ = ∇ℱ[permutation]
    return -sign(1 - ℱ) * ∇ℱ
end

@views function iso_vec_unitary_infidelity_hessian(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector;
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    U = iso_vec_to_operator(Ũ⃗)[subspace, subspace]
    n = size(U, 1)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)[subspace, subspace]
    U⃗ᵣ, U⃗ᵢ = vec(real(U)), vec(imag(U))
    Ū⃗ᵣ, Ū⃗ᵢ = vec(real(U_goal)), vec(imag(U_goal))
    Tᵣ = Ū⃗ᵣ' * U⃗ᵣ + Ū⃗ᵢ' * U⃗ᵢ
    Tᵢ = Ū⃗ᵣ' * U⃗ᵢ - Ū⃗ᵢ' * U⃗ᵣ
    Wᵣᵣ = Ū⃗ᵣ * Ū⃗ᵣ'
    Wᵢᵢ = Ū⃗ᵢ * Ū⃗ᵢ'
    Wᵣᵢ = Ū⃗ᵣ * Ū⃗ᵢ'
    Wᵢᵣ = Wᵣᵢ'
    ℱ = 1 / n * sqrt(Tᵣ^2 + Tᵢ^2)
    ∇ᵣℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵣ - Tᵢ * Ū⃗ᵢ)
    ∇ᵢℱ = 1 / (n^2 * ℱ) * (Tᵣ * Ū⃗ᵢ + Tᵢ * Ū⃗ᵣ)
    ∂ᵣ²ℱ = 1 / ℱ * (-∇ᵣℱ * ∇ᵣℱ' + 1 / n^2 * (Wᵣᵣ + Wᵢᵢ))
    ∂ᵢ²ℱ = 1 / ℱ * (-∇ᵢℱ * ∇ᵢℱ' + 1 / n^2 * (Wᵣᵣ + Wᵢᵢ))
    ∂ᵣ∂ᵢℱ = 1 / ℱ * (-∇ᵢℱ * ∇ᵣℱ' + 1 / n^2 * (Wᵢᵣ - Wᵣᵢ))
    ∂²ℱ = [∂ᵣ²ℱ ∂ᵣ∂ᵢℱ; ∂ᵣ∂ᵢℱ' ∂ᵢ²ℱ]
    # TODO: This should be moved to Isomorphisms.jl
    permutation = vcat(vcat([[slice(j, n), slice(j, n) .+ n^2] for j = 1:n]...)...)
    ∂²ℱ = ∂²ℱ[permutation, permutation]
    return -sign(1 - ℱ) * ∂²ℱ
end

struct UnitaryInfidelityLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function UnitaryInfidelityLoss(
        name::Symbol,
        Ũ⃗_goal::AbstractVector;
        subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
    )
        l = Ũ⃗ -> iso_vec_unitary_infidelity(Ũ⃗, Ũ⃗_goal, subspace=subspace)

        @views ∇l = Ũ⃗ -> begin
            subspace_rows, subspace_cols = (subspace, subspace)
            n_subspace = length(subspace_rows)
            n_full = Int(sqrt(length(Ũ⃗) ÷ 2))
            ∇l_subspace = iso_vec_unitary_infidelity_gradient(Ũ⃗, Ũ⃗_goal, subspace=subspace)
            ∇l_full = zeros(2 * n_full^2)
            for j ∈ eachindex(subspace_cols)
                ∇l_full[slice(2subspace_cols[j] - 1, subspace_rows, n_full)] =
                    ∇l_subspace[slice(2j - 1, n_subspace)]
                ∇l_full[slice(2subspace_cols[j], subspace_rows, n_full)] =
                    ∇l_subspace[slice(2j, n_subspace)]
            end
            return ∇l_full
        end

        @views ∇²l = Ũ⃗ -> begin
            subspace_rows, subspace_cols = (subspace, subspace)
            n_subspace = length(subspace_rows)
            n_full = Int(sqrt(length(Ũ⃗) ÷ 2))
            ∇²l_subspace = iso_vec_unitary_infidelity_hessian(Ũ⃗, Ũ⃗_goal, subspace=subspace)
            ∇²l_full = zeros(2 * n_full^2, 2 * n_full^2)
            # NOTE: Assumes subspace_rows = subspace_cols
            for k ∈ eachindex(subspace_cols)
                for j ∈ eachindex(subspace_cols)
                    ∇²l_full[
                        slice(2subspace_cols[k] - 1, subspace_rows, n_full),
                        slice(2subspace_cols[j] - 1, subspace_rows, n_full)
                    ] = ∇²l_subspace[
                        slice(2k - 1, n_subspace),
                        slice(2j - 1, n_subspace)
                    ]

                    ∇²l_full[
                        slice(2subspace_cols[k] - 1, subspace_rows, n_full),
                        slice(2subspace_cols[j], subspace_rows, n_full)
                    ] = ∇²l_subspace[
                        slice(2k - 1, n_subspace),
                        slice(2j, n_subspace)
                    ]

                    ∇²l_full[
                        slice(2subspace_cols[k], subspace_rows, n_full),
                        slice(2subspace_cols[j] - 1, subspace_rows, n_full)
                    ] = ∇²l_subspace[
                        slice(2k, n_subspace),
                        slice(2j - 1, n_subspace)
                    ]

                    ∇²l_full[
                        slice(2subspace_cols[k], subspace_rows, n_full),
                        slice(2subspace_cols[j], subspace_rows, n_full)
                    ] = ∇²l_subspace[
                        slice(2k, n_subspace),
                        slice(2j, n_subspace)
                    ]
                end
            end
            return ∇²l_full
        end

        Ũ⃗_dim = length(Ũ⃗_goal)
        ∇²l_structure = []
        for (i, j) ∈ Iterators.product(1:Ũ⃗_dim, 1:Ũ⃗_dim)
            if i ≤ j
                push!(∇²l_structure, (i, j))
            end
        end
        return new(l, ∇l, ∇²l, ∇²l_structure, name)
    end
end

function (loss::UnitaryInfidelityLoss)(
    Ũ⃗_end::AbstractVector{<:Real};
    gradient=false,
    hessian=false
)
    @assert !(gradient && hessian)
    if !(gradient || hessian)
        return loss.l(Ũ⃗_end)
    elseif gradient
        return loss.∇l(Ũ⃗_end)
    elseif hessian
        return loss.∇²l(Ũ⃗_end)
    end
end

###
### UnitaryFreePhaseInfidelityLoss
###

function free_phase(
    ϕs::AbstractVector,
    Hs::AbstractVector{<:AbstractMatrix}
)
    # NOTE: switch to expv for ForwardDiff
    # return reduce(kron, [exp(im * ϕ * H) for (ϕ, H) ∈ zip(ϕs, Hs)])
    Id = Matrix{eltype(Hs[1])}(I, size(Hs[1]))
    return reduce(kron, [expv(im * ϕ, H, Id) for (ϕ, H) ∈ zip(ϕs, Hs)])
end

# TODO: in-place
function free_phase_gradient(
    ϕs::AbstractVector,
    Hs::AbstractVector{<:AbstractMatrix}
)
    R = free_phase(ϕs, Hs)
    result = [zeros(eltype(R), size(R)) for _ in eachindex(ϕs)]

    # store identity matrices
    identities = [Matrix{eltype(H)}(I, size(H)) for H in Hs]

    for (i, H) in enumerate(Hs)
        # insert H into identities
        identities[i] .= H
        result[i] .= im * reduce(kron, identities) * R
        # reset identities
        identities[i] .= Matrix{eltype(H)}(I, size(H))
    end
    return result
end

@views function unitary_free_phase_fidelity(
    U::Matrix,
    U_goal::Matrix,
    phases::AbstractVector,
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(U_goal, 1)
)
    # extract phase rotation (assume phase operators span goal subspace)
    R = zeros(eltype(U), size(U))
    R[subspace, subspace] = free_phase(phases, phase_operators)

    # calculate fidelity
    return unitary_fidelity(R * U, U_goal, subspace=subspace)
end

@views function iso_vec_unitary_free_phase_fidelity(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector,
    phases::AbstractVector,
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    U = iso_vec_to_operator(Ũ⃗)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)
    return unitary_free_phase_fidelity(U, U_goal, phases, phase_operators, subspace=subspace)
end

@views function iso_vec_unitary_free_phase_infidelity(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector,
    phases::AbstractVector,
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    ℱ = iso_vec_unitary_free_phase_fidelity(
        Ũ⃗, Ũ⃗_goal, phases, phase_operators, 
        subspace=subspace
    )
    return abs(1 - ℱ)
end

@views function iso_vec_unitary_free_phase_infidelity_gradient(
    Ũ⃗::AbstractVector,
    Ũ⃗_goal::AbstractVector,
    phases::AbstractVector,
    phase_operators::AbstractVector{<:AbstractMatrix};
    subspace::AbstractVector{Int}=axes(iso_vec_to_operator(Ũ⃗_goal), 1)
)
    n_phases = length(phases)
    n_subspace = 2 * length(subspace) * length(subspace)
    ∂ = spzeros(n_subspace + n_phases)

    # extract full state
    U = iso_vec_to_operator(Ũ⃗)

    # extract full phase rotation
    R = zeros(eltype(U), size(U))
    R[subspace, subspace] = free_phase(phases, phase_operators)

    # loss gradient in subspace
    ∂ℱ_∂Ũ⃗_subspace = iso_vec_unitary_infidelity_gradient(
        operator_to_iso_vec(R * U), Ũ⃗_goal, subspace=subspace
    )

    # state slice in subspace
    ∂[1:n_subspace] = operator_to_iso_vec(R[subspace, subspace]'iso_vec_to_operator(∂ℱ_∂Ũ⃗_subspace))

    # phase slice
    ∂[n_subspace .+ (1:n_phases)] = [
        ∂ℱ_∂Ũ⃗_subspace'operator_to_iso_vec(∂R_∂ϕⱼ_subspace * U[subspace, subspace])
        for ∂R_∂ϕⱼ_subspace in free_phase_gradient(phases, phase_operators)
    ]

    return ∂
end

struct UnitaryFreePhaseInfidelityLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function UnitaryFreePhaseInfidelityLoss(
        Ũ⃗_goal::AbstractVector,
        phase_operators::AbstractVector{<:AbstractMatrix};
        subspace::Union{AbstractVector{Int}, Nothing}=nothing,
    )
        if isnothing(subspace)
            subspace = axes(iso_vec_to_operator(Ũ⃗_goal), 1)
        end

        @assert reduce(*, size.(phase_operators, 1)) == length(subspace) "phase operators must span the subspace"

        @views function l(Ũ⃗::AbstractVector, ϕ⃗::AbstractVector)
            return iso_vec_unitary_free_phase_infidelity(Ũ⃗, Ũ⃗_goal, ϕ⃗, phase_operators, subspace=subspace)
        end

        @views function ∇l(Ũ⃗::AbstractVector, ϕ⃗::AbstractVector)
            subspace_rows = subspace_cols = subspace
            n_phase = length(ϕ⃗)
            n_rows = length(subspace_rows)
            n_cols = length(subspace_cols)
            n_full = Int(sqrt(length(Ũ⃗) ÷ 2))

            # gradient in subspace
            ∇l_subspace = iso_vec_unitary_free_phase_infidelity_gradient(
                Ũ⃗, Ũ⃗_goal, ϕ⃗, phase_operators, subspace=subspace
            )
            ∇l_full = zeros(2 * n_full^2 + n_phase)

            # state slice
            for j ∈ 1:n_cols
                ∇l_full[slice(2subspace_cols[j] - 1, subspace_rows, n_full)] =
                    ∇l_subspace[slice(2j - 1, n_rows)]
                ∇l_full[slice(2subspace_cols[j], subspace_rows, n_full)] =
                    ∇l_subspace[slice(2j, n_rows)]
            end

            # phase slice
            ∇l_full[2 * n_full^2 .+ (1:n_phase)] = ∇l_subspace[2 * n_rows * n_cols .+ (1:n_phase)]
            return ∇l_full
        end

        # TODO: implement analytic hessian
        ∇²l(Ũ⃗, ϕ⃗) = []
        ∇²l_structure = []

        return new(l, ∇l, ∇²l, ∇²l_structure, :NA)
    end
end

function (loss::UnitaryFreePhaseInfidelityLoss)(
    Ũ⃗_end::AbstractVector{<:Real},
    ϕ⃗::AbstractVector{<:Real};
    gradient=false,
    hessian=false
)
    @assert !(gradient && hessian)
    if !(gradient || hessian)
        return loss.l(Ũ⃗_end, ϕ⃗)
    elseif gradient
        return loss.∇l(Ũ⃗_end, ϕ⃗)
    elseif hessian
        return loss.∇²l(Ũ⃗_end, ϕ⃗)
    end
end

# ============================================================================= #

@testitem "Unitary fidelity" begin
    X = [0 1; 1 0]
    X = [0 1; 1 0]
    Y = [0 -im; im 0]
    @test unitary_fidelity(X, X) ≈ 1
    @test unitary_fidelity(X, Y) ≈ 0
end

@testitem "Isovec Unitary Fidelity" begin
    using LinearAlgebra

    U_X = [0 1; 1 0]
    U_Y = [0 -im; im 0]

    for U in [U_X, U_Y]
        @test U'U ≈ I
    end

    Ũ⃗_X = operator_to_iso_vec(U_X)
    Ũ⃗_Y = operator_to_iso_vec(U_Y)

    # Test gate fidelity
    @test iso_vec_unitary_fidelity(Ũ⃗_X, Ũ⃗_X) ≈ 1
    @test iso_vec_unitary_fidelity(Ũ⃗_X, Ũ⃗_Y) ≈ 0


    # Test asymmetric fidelity
    U_fn(λ, φ) = [1 -exp(im * λ); exp(im * φ) exp(im * (φ + λ))] / sqrt(2)
    U_1 = U_fn(π/4, π/3)
    U_2 = U_fn(1.5, .33)

    for U in [U_1, U_2]
        @test U'U ≈ I
    end

    Ũ⃗_1 = operator_to_iso_vec(U_1)
    Ũ⃗_2 = operator_to_iso_vec(U_2)

    @test iso_vec_unitary_fidelity(Ũ⃗_1, Ũ⃗_1) ≈ 1
    @test iso_vec_unitary_fidelity(Ũ⃗_2, Ũ⃗_2) ≈ 1
    @test iso_vec_unitary_fidelity(Ũ⃗_1, Ũ⃗_2) ≈ iso_vec_unitary_fidelity(Ũ⃗_2, Ũ⃗_1)
    @test iso_vec_unitary_fidelity(Ũ⃗_1, Ũ⃗_2) ≈ abs(tr(U_1'U_2)) / 2


    # Test random fidelity
    U_H1 = haar_random(2)
    U_H2 = haar_random(2)

    for U in [U_H1, U_H2]
        @test U'U ≈ I
    end

    Ũ⃗_H1 = operator_to_iso_vec(U_H1)
    Ũ⃗_H2 = operator_to_iso_vec(U_H2)

    @test iso_vec_unitary_fidelity(Ũ⃗_H1, Ũ⃗_H1) ≈ 1
    @test iso_vec_unitary_fidelity(Ũ⃗_H2, Ũ⃗_H2) ≈ 1
    @test iso_vec_unitary_fidelity(Ũ⃗_H1, Ũ⃗_X) ≈ abs(tr(U_H1'U_X)) / 2
    @test iso_vec_unitary_fidelity(Ũ⃗_H1, Ũ⃗_H2) ≈ abs(tr(U_H1'U_H2)) / 2
end


@testitem "Isovec Unitary Fidelity Subspace" begin
    using LinearAlgebra

    function test_iso_vec_unitary_fidelity(
        U₁::AbstractMatrix,
        U₂::AbstractMatrix,
        subspace
    )
        Ũ⃗₁ = operator_to_iso_vec(U₁)
        Ũ⃗₂ = operator_to_iso_vec(U₂)
        return iso_vec_unitary_fidelity(Ũ⃗₁, Ũ⃗₂, subspace=subspace)
    end

    # Test random fidelity
    test_subspaces = [
        get_subspace_indices([1:2, 1:1], [2, 2]),
        get_subspace_indices([1:2, 2:2], [2, 2]),
    ]

    for ii in test_subspaces
        U_H1 = kron(haar_random(2), haar_random(2))
        U_H1_sub = U_H1[ii, ii]
        U_H2 = kron(haar_random(2), haar_random(2))
        U_H2_sub = U_H2[ii, ii]

        # NOTE: subspace may not be unitary (?)
        for U in [U_H1, U_H2]
            @test U'U ≈ I
        end

        fid = test_iso_vec_unitary_fidelity(U_H1, U_H2, ii)
        fid_sub = test_iso_vec_unitary_fidelity(U_H1_sub, U_H2_sub, axes(U_H1_sub, 1))
        @test fid ≈ fid_sub
    end
end


@testitem "Isovec Unitary Fidelity Gradient" begin

end
