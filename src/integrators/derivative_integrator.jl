
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
