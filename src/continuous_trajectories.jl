module ContinuousTrajectories

export ContinuousTrajectory
export plot

using CairoMakie
using HDF5

import NamedTrajectories: plot, save

struct ContinuousTrajectory
    A::AbstractMatrix{Float64}
    dA::AbstractMatrix{Float64}
    times::AbstractVector{Float64}
end

function cubic_spline_coeffs(aᵢ, ȧᵢ, aᵢ₊₁, ȧᵢ₊₁, tᵢ, tᵢ₊₁)
    M = [
        1  tᵢ    tᵢ^2    tᵢ^3   ;
        0   1   2tᵢ     3tᵢ^2   ;
        1  tᵢ₊₁  tᵢ₊₁^2  tᵢ₊₁^3 ;
        0   1   2tᵢ₊₁   3tᵢ₊₁^2
    ]
    A = vcat(transpose.([aᵢ, ȧᵢ, aᵢ₊₁, ȧᵢ₊₁])...)
    B = inv(M) * A
    return B
end

function (ct::ContinuousTrajectory)(t::Real)
    @assert t >= ct.times[1] && t <= ct.times[end] "t = $(t) must be in the range of the trajectory, i.e. t ∈ [$(ct.times[1]), $(ct.times[end])]."
    if t == ct.times[1]
        return ct.A[:, 1]
    elseif t == ct.times[end]
        return ct.A[:, end]
    end
    tₜ = findfirst(t .< ct.times)
    tₜ₋₁ = tₜ - 1
    tᵢ = ct.times[tₜ₋₁]
    tᵢ₊₁ = ct.times[tₜ]
    aᵢ = ct.A[:, tₜ₋₁]
    aᵢ₊₁ = ct.A[:, tₜ]
    ȧᵢ = ct.dA[:, tₜ₋₁]
    ȧᵢ₊₁ = ct.dA[:, tₜ]
    B = cubic_spline_coeffs(aᵢ, ȧᵢ, aᵢ₊₁, ȧᵢ₊₁, tᵢ, tᵢ₊₁)
    T = [1, t, t^2, t^3]
    return B' * T
end

function (ct::ContinuousTrajectory)(ts::AbstractVector{<:Real})
    @assert issorted(ts)
    return hcat([ct(tᵢ) for tᵢ in ts]...)
end

function (ct::ContinuousTrajectory)(nsamples::Int; return_times=false)
    ts = LinRange(ct.times[1], ct.times[end], nsamples)
    if !return_times
        return ct(ts)
    else
        return ct(ts), A
    end
end

function plot(
    ct::ContinuousTrajectory,
    ts::AbstractVector{<:Real};
    res=(800, 400),
)
    fig = Figure(resolution=res)
    ax = Axis(fig[1, 1])
    A = ct(ts)
    series!(ax, ts, A)
    return fig
end

function plot(
    ct::ContinuousTrajectory,
    nsamples::Int;
    res=(800, 400),
)
    fig = Figure(resolution=res)
    ax = Axis(fig[1, 1])
    A, ts = ct(nsamples; return_times=true)
    series!(ax, ts, A)
    return fig
end

function save(path::String, ct::ContinuousTrajectory, ts::AbstractVector{<:Real})
    A = ct(ts)
    mkdir(dirname(path))
    h5open(path, "w") do file
        file["controls"] = A
        file["times"] = ts
        file["T"] = length(ts)
    end
end

function save(path::String, ct::ContinuousTrajectory, nsamples::Int)
    A, ts = ct(nsamples; return_times=true)
    mkdir(dirname(path))
    h5open(path, "w") do file
        file["controls"] = A
        file["times"] = ts
        file["T"] = length(ts)
    end
end

end
