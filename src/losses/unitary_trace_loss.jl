export UnitaryTraceLoss


function unitary_trace_loss(Ũ⃗::AbstractVector, Ũ⃗_goal::AbstractVector)
    U = iso_vec_to_operator(Ũ⃗)
    Ugoal = iso_vec_to_operator(Ũ⃗_goal)
    return 1 / 2 * tr(sqrt((U - Ugoal)' * (U - Ugoal)))
end

struct UnitaryTraceLoss <: AbstractLoss
    l::Function
    ∇l::Function
    ∇²l::Function
    ∇²l_structure::Vector{Tuple{Int,Int}}
    name::Symbol

    function UnitaryTraceLoss(
        name::Symbol,
        Ũ⃗_goal::AbstractVector
    )
        l = Ũ⃗ -> unitary_trace_loss(Ũ⃗, Ũ⃗_goal)
        ∇l = Ũ⃗ -> ForwardDiff.gradient(l, Ũ⃗)
        ∇²l = Ũ⃗ -> ForwardDiff.hessian(l, Ũ⃗)
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

function (loss::UnitaryTraceLoss)(
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