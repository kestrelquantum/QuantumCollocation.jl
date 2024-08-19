export QuantumLoss
export QuantumLossGradient
export QuantumLossHessian


struct QuantumLoss
    cs::Vector{Function}
    isodim::Int

    function QuantumLoss(
        sys::AbstractQuantumSystem,
        loss::Symbol = :infidelity_loss
    )
        if loss == :energy_loss
            cs = [ψ̃ⁱ -> eval(loss)(ψ̃ⁱ, sys.H_target) for i = 1:sys.nqstates]
        elseif loss == :neg_entropy_loss
            cs = [ψ̃ⁱ -> eval(loss)(ψ̃ⁱ) for i = 1:sys.nqstates]
        else
            cs = [
                ψ̃ⁱ -> eval(loss)(
                    ψ̃ⁱ,
                    sys.ψ̃goal[slice(i, sys.isodim)]
                ) for i = 1:sys.nqstates
            ]
        end
        return new(cs, sys.isodim)
    end
end

function (qloss::QuantumLoss)(ψ̃::AbstractVector)
    loss = 0.0
    for (i, cⁱ) in enumerate(qloss.cs)
        loss += cⁱ(ψ̃[slice(i, qloss.isodim)])
    end
    return loss
end

struct QuantumLossGradient
    ∇cs::Vector{Function}
    isodim::Int

    function QuantumLossGradient(
        loss::QuantumLoss;
        simplify=true
    )
        Symbolics.@variables ψ̃[1:loss.isodim]

        ψ̃ = collect(ψ̃)

        ∇cs_symbs = [
            Symbolics.gradient(c(ψ̃), ψ̃; simplify=simplify)
                for c in loss.cs
        ]

        ∇cs_exprs = [
            Symbolics.build_function(∇c, ψ̃)
                for ∇c in ∇cs_symbs
        ]

        ∇cs = [
            eval(∇c_expr[1])
                for ∇c_expr in ∇cs_exprs
        ]

        return new(∇cs, loss.isodim)
    end
end

@views function (∇c::QuantumLossGradient)(
    ψ̃::AbstractVector
)
    ∇ = similar(ψ̃)

    for (i, ∇cⁱ) in enumerate(∇c.∇cs)

        ψ̃ⁱ_slice = slice(i, ∇c.isodim)

        ∇[ψ̃ⁱ_slice] = ∇cⁱ(ψ̃[ψ̃ⁱ_slice])
    end

    return ∇
end

struct QuantumLossHessian
    ∇²cs::Vector{Function}
    ∇²c_structures::Vector{Vector{Tuple{Int, Int}}}
    isodim::Int

    function QuantumLossHessian(
        loss::QuantumLoss;
        simplify=true
    )

        Symbolics.@variables ψ̃[1:loss.isodim]
        ψ̃ = collect(ψ̃)

        ∇²c_symbs = [
            Symbolics.sparsehessian(
                c(ψ̃),
                ψ̃;
                simplify=simplify
            ) for c in loss.cs
        ]

        ∇²c_structures = []

        for ∇²c_symb in ∇²c_symbs
            K, J, _ = findnz(∇²c_symb)

            KJ = collect(zip(K, J))

            filter!(((k, j),) -> k ≤ j, KJ)

            push!(∇²c_structures, KJ)
        end

        ∇²c_exprs = [
            Symbolics.build_function(∇²c_symb, ψ̃)
                for ∇²c_symb in ∇²c_symbs
        ]

        ∇²cs = [
            eval(∇²c_expr[1])
                for ∇²c_expr in ∇²c_exprs
        ]

        return new(∇²cs, ∇²c_structures, loss.isodim)
    end
end

function StructureUtils.structure(
    H::QuantumLossHessian,
    T::Int,
    vardim::Int
)
    H_structure = []

    T_offset = index(T, 0, vardim)

    for (i, KJⁱ) in enumerate(H.∇²c_structures)

        i_offset = index(i, 0, H.isodim)

        for kj in KJⁱ
            push!(H_structure, (T_offset + i_offset) .+ kj)
        end
    end

    return H_structure
end

@views function (H::QuantumLossHessian)(ψ̃::AbstractVector)

    Hs = []

    for (i, ∇²cⁱ) in enumerate(H.∇²cs)

        ψ̃ⁱ = ψ̃[slice(i, H.isodim)]

        for (k, j) in H.∇²c_structures[i]

            Hⁱᵏʲ = ∇²cⁱ(ψ̃ⁱ)[k, j]

            append!(Hs, Hⁱᵏʲ)
        end
    end

    return Hs
end