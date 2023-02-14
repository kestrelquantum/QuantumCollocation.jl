module IndexingUtils

export index
export slice

"""
this module contains helper functions for indexing and taking slices of the full problem variable vector

definitions:

the problem vector: Z = [z₁, z₂, ..., zT]

    knot point:
        zₜ = [xₜ, uₜ]

    augmented state vector:
        xₜ = [ψ̃ₜ, ψ̃²ₜ, ..., ψ̃ⁿₜ, ∫aₜ, aₜ, daₜ, ..., dᶜ⁻¹aₜ]

    where:
        c = control_order


also, below, we use dim(zₜ) = dim

examples:

Z[index(t, pos, dim)] = zₜ[pos]
Z[index(t, dim)]      = zₜ[dim]

Z[slice(t, pos1, pos2, dim)]      = zₜ[pos1:pos2]
Z[slice(t, pos, dim)]             = zₜ[1:pos]
Z[slice(t, dim)]                  = zₜ[1:dim] := zₜ
Z[slice(t, dim; stretch=stretch)] = zₜ[1:(dim + stretch)]
Z[slice(t, indices, dim)]         = zₜ[indices]
Z[slice(t1:t2, dim)]              = [zₜ₁;...;zₜ₂]

the functions are also used to access the zₜ vectors, e.g.

zₜ[slice(i, isodim)]                             = ψ̃ⁱₜ
zₜ[n_wfn_states .+ slice(1, ncontrols)]          = ∫aₜ
zₜ[n_wfn_states .+ slice(2, ncontrols)]          = aₜ
zₜ[n_wfn_states .+ slice(augdim + 1, ncontrols)] = uₜ = ddaₜ
"""

index(t::Int, pos::Int, dim::Int) = dim * (t - 1) + pos

index(t, dim) = index(t, dim, dim)


slice(t, pos1, pos2, dim) =
    index(t, pos1, dim):index(t, pos2, dim)

slice(t, pos, dim) = slice(t, 1, pos, dim)

slice(t, dim; stretch=0) = slice(t, 1, dim + stretch, dim)

slice(t::Int, indices::AbstractVector{Int}, dim::Int) =
    dim * (t - 1) .+ indices

slice(ts::UnitRange{Int}, dim::Int) = slice(ts[1], length(ts) * dim, dim)

end
