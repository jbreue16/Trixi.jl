
# This file contains general numerical fluxes that are not specific to certain equations

"""
    flux_central(u_ll, u_rr, orientation_or_normal, equations::AbstractEquations)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation_or_normal, equations::AbstractEquations)
  # Calculate regular 1D fluxes
  f_ll = flux(u_ll, orientation_or_normal, equations)
  f_rr = flux(u_rr, orientation_or_normal, equations)

  # Average regular fluxes
  return 0.5 * (f_ll + f_rr)
end

# used to solve Auxiliary Equation for viscous flux
@inline function flux_central_auxiliary(u_ll, u_rr, orientation_or_normal, equations::AbstractEquations)
  # Calculate regular 1D fluxes
  f_ll = flux(u_ll, orientation_or_normal, equations)
  f_rr = flux(u_rr, orientation_or_normal, equations)

  return 0.5 * (f_rr + f_ll)
end

@inline function surface_flux_plus_viscous_central(u_ll, u_rr, q1_ll, q1_rr, q2_ll, q2_rr, orientation_or_normal, equations::AbstractEquations, surface_flux)
  numerical_flux = surface_flux(u_ll, u_rr, orientation_or_normal, equations)
  numerical_flux -= 0.5* (viscous_flux(u_ll, q1_ll, q2_ll, orientation_or_normal, equations) + viscous_flux(u_rr, q1_rr, q2_rr, orientation_or_normal, equations))
  return numerical_flux
end

"""
    FluxPlusDissipation(numerical_flux, dissipation)

Combine a `numerical_flux` with a `dissipation` operator to create a new numerical flux.
"""
struct FluxPlusDissipation{NumericalFlux, Dissipation}
  numerical_flux::NumericalFlux
  dissipation::Dissipation
end

@inline function (numflux::FluxPlusDissipation)(u_ll, u_rr, orientation_or_normal, equations)
  @unpack numerical_flux, dissipation = numflux

  return numerical_flux(u_ll, u_rr, orientation_or_normal, equations) + dissipation(u_ll, u_rr, orientation_or_normal, equations)
end

Base.show(io::IO, f::FluxPlusDissipation) = print(io, "FluxPlusDissipation(",  f.numerical_flux, ", ", f.dissipation, ")")


"""
    FluxRotated(numerical_flux)

Compute a `numerical_flux` flux in direction of a normal vector by rotating the solution,
computing the numerical flux in x-direction, and rotating the calculated flux back.

Requires a rotationally invariant equation with equation-specific functions
[`rotate_to_x`](@ref) and [`rotate_from_x`](@ref).

!!! warning "Experimental code"
    This flux is experimental and is likely to change in a future release. Do not use it in production code.
"""
struct FluxRotated{NumericalFlux}
  numerical_flux::NumericalFlux
end


# Rotated surface flux computation (2D version)
@inline function (flux_rotated::FluxRotated)(u_ll, u_rr, normal::AbstractVector, equations::AbstractEquations{2})
  @unpack numerical_flux = flux_rotated

  norm_ = norm(normal)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal_normalized = normal / norm_

  u_ll_rotated = rotate_to_x(u_ll, normal_normalized, equations)
  u_rr_rotated = rotate_to_x(u_rr, normal_normalized, equations)

  f = numerical_flux(u_ll_rotated, u_rr_rotated, 1, equations)

  return rotate_from_x(f, normal_normalized, equations) * norm_
end


# Rotated surface flux computation (3D version)
@inline function (flux_rotated::FluxRotated)(u_ll, u_rr, normal::AbstractVector, equations::AbstractEquations{3})
  @unpack numerical_flux = flux_rotated

  # Storing these vectors could increase the performance by 20 percent
  norm_ = norm(normal)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal_normalized = normal / norm_

  # Some vector that can't be identical to normal (unless normal == 0)
  tangent1 = SVector(normal[2], normal[3], -normal[1])
  # Orthogonal projection
  tangent1 -= dot(normal_normalized, tangent1) * normal_normalized
  tangent1 = normalize(tangent1)

  # Third orthogonal vector
  tangent2 = normalize(cross(normal, tangent1))

  u_ll_rotated = rotate_to_x(u_ll, normal_normalized, tangent1, tangent2, equations)
  u_rr_rotated = rotate_to_x(u_rr, normal_normalized, tangent1, tangent2, equations)

  f = numerical_flux(u_ll_rotated, u_rr_rotated, 1, equations)

  return rotate_from_x(f, normal_normalized, tangent1, tangent2, equations) * norm_
end

Base.show(io::IO, f::FluxRotated) = print(io, "FluxRotated(",  f.numerical_flux, ")")


"""
    DissipationGlobalLaxFriedrichs(λ)

Create a global Lax-Friedrichs dissipation operator with dissipation coefficient `λ`.
"""
struct DissipationGlobalLaxFriedrichs{RealT}
  λ::RealT
end

@inline function (dissipation::DissipationGlobalLaxFriedrichs)(u_ll, u_rr, orientation::Integer, equations)
  @unpack λ = dissipation
  return -λ/2 * (u_rr - u_ll)
end

@inline function (dissipation::DissipationGlobalLaxFriedrichs)(u_ll, u_rr, normal::AbstractVector, equations)
  @unpack λ = dissipation
  return -λ/2 * norm(normal) * (u_rr - u_ll)
end

Base.show(io::IO, d::DissipationGlobalLaxFriedrichs) = print(io, "DissipationGlobalLaxFriedrichs(", d.λ, ")")


"""
    DissipationLocalLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal, equations)`,
defaulting to [`max_abs_speed_naive`](@ref).
"""
struct DissipationLocalLaxFriedrichs{MaxAbsSpeed}
  max_abs_speed::MaxAbsSpeed
end

DissipationLocalLaxFriedrichs() = DissipationLocalLaxFriedrichs(max_abs_speed_naive)

@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, orientation_or_normal, equations)
  λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal, equations)
  return -0.5 * λ * (u_rr - u_ll)
end

Base.show(io::IO, d::DissipationLocalLaxFriedrichs) = print(io, "DissipationLocalLaxFriedrichs(", d.max_abs_speed, ")")


"""
    max_abs_speed_naive(u_ll, u_rr, orientation::Integer,   equations)
    max_abs_speed_naive(u_ll, u_rr, normal::AbstractVector, equations)

Simple and fast estimate of the maximal wave speed of the Riemann problem with left and right states
`u_ll, u_rr`, based only on the local wave speeds associated to `u_ll` and `u_rr`.
"""
function max_abs_speed_naive end


const FluxLaxFriedrichs{MaxAbsSpeed} = FluxPlusDissipation{typeof(flux_central), DissipationLocalLaxFriedrichs{MaxAbsSpeed}}
"""
    FluxLaxFriedrichs(max_abs_speed=max_abs_speed_naive)

Local Lax-Friedrichs (Rusanov) flux with maximum wave speed estimate provided by
`max_abs_speed`, cf. [`DissipationLocalLaxFriedrichs`](@ref) and
[`max_abs_speed_naive`](@ref).
"""
function FluxLaxFriedrichs(max_abs_speed=max_abs_speed_naive)
  FluxPlusDissipation(flux_central, DissipationLocalLaxFriedrichs(max_abs_speed))
end

Base.show(io::IO, f::FluxLaxFriedrichs) = print(io, "FluxLaxFriedrichs(", f.dissipation.max_abs_speed, ")")

# TODO: Shall we deprecate `flux_lax_friedrichs`?
"""
    flux_lax_friedrichs

See [`FluxLaxFriedrichs`](@ref).
"""
const flux_lax_friedrichs = FluxLaxFriedrichs()


"""
    FluxHLL(min_max_speed=min_max_speed_naive)

Create an HLL (Harten, Lax, van Leer) numerical flux where the minimum and maximum
wave speeds are estimated as
`λ_min, λ_max = min_max_speed(u_ll, u_rr, orientation_or_normal, equations)`,
defaulting to [`min_max_speed_naive`](@ref).
"""
struct FluxHLL{MinMaxSpeed}
  min_max_speed::MinMaxSpeed
end

FluxHLL() = FluxHLL(min_max_speed_naive)

"""
    min_max_speed_naive(u_ll, u_rr, orientation::Integer,   equations)
    min_max_speed_naive(u_ll, u_rr, normal::AbstractVector, equations)

Simple and fast estimate of the minimal and maximal wave speed of the Riemann problem with
left and right states `u_ll, u_rr`, usually based only on the local wave speeds associated to
`u_ll` and `u_rr`.
- Amiram Harten, Peter D. Lax, Bram van Leer (1983)
  On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws
  [DOI: 10.1137/1025002](https://doi.org/10.1137/1025002)
"""
function min_max_speed_naive end

@inline function (numflux::FluxHLL)(u_ll, u_rr, orientation_or_normal, equations)
  λ_min, λ_max = numflux.min_max_speed(u_ll, u_rr, orientation_or_normal, equations)

  if λ_min >= 0 && λ_max >= 0
    return flux(u_ll, orientation_or_normal, equations)
  elseif λ_max <= 0 && λ_min <= 0
    return flux(u_rr, orientation_or_normal, equations)
  else
    f_ll = flux(u_ll, orientation_or_normal, equations)
    f_rr = flux(u_rr, orientation_or_normal, equations)
    inv_λ_max_minus_λ_min = inv(λ_max - λ_min)
    factor_ll = λ_max * inv_λ_max_minus_λ_min
    factor_rr = λ_min * inv_λ_max_minus_λ_min
    factor_diss = λ_min * λ_max * inv_λ_max_minus_λ_min
    return factor_ll * f_ll - factor_rr * f_rr + factor_diss * (u_rr - u_ll)
  end
end

Base.show(io::IO, numflux::FluxHLL) = print(io, "FluxHLL(", numflux.min_max_speed, ")")

# TODO: Shall we deprecate `flux_hll`?
"""
    flux_hll

See [`FluxHLL`](@ref).
"""
const flux_hll = FluxHLL()
