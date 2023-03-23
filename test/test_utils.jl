"""
    utility functions for debugging tests
"""

"""
    dense(vals, structure, shape)

Convert sparse data to dense matrix.

# Arguments
- `vals`: vector of values
- `structure`: vector of tuples of indices
- `shape`: tuple of matrix dimensions
"""
function dense(vals, structure, shape)

    M = zeros(shape)

    for (v, (k, j)) in zip(vals, structure)
        M[k, j] += v
    end

    if shape[1] == shape[2]
        return Symmetric(M)
    else
        return M
    end
end

"""
    show_diffs(A::Matrix, B::Matrix)

Show differences between matrices.
"""
function show_diffs(A::AbstractMatrix, B::AbstractMatrix)
    for (i, (a, b)) in enumerate(zip(A, B))
        inds = Tuple(CartesianIndices(A)[i])
        if !isapprox(a, b, atol=ATOL) && inds[1] ≤ inds[2]
            println((a, b), " @ ", inds)
        end
    end
end
