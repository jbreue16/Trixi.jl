using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# Test mit Polynomen funktioniert -> Achte aber auf stetige RB
# test mit Trigonometrischen Funktionen Funktioniert
# Krumme Gitter sehen jetzt auch gut aus.
# ABER: welche für boundary conditions für die Hilfsgleichung? unstetige, zb periodisch führen zu
#  "komischen"/großen Ableitungen wenn periodisch keinen "Sinn" macht, also unstetig ist
# Fehler für Konstante Ableitung = 0 wird immer größer mit steigendem c, N... alles andere wird besser


######## notwendige funktionen die geladen werden müssen ##############################################
# mapping O-mesh 
function mapping(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = 5 * (2 + ξ ) * cos(π * (η + 1))
    y = 5 * (2 + ξ ) * sin(π * (η + 1))
    return SVector(x, y)
end
function mapping1zu1(xi_, eta_)
  x = xi_ 
  y = eta_
  return SVector(x, y)
end
function mappingLin(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = cos(pi* ξ) #5 * (2 + ξ ) + η #* cos(π * (η + 1))
    y = sin(pi* η) #5 * (2 + η ) + ξ #* sin(π * (η + 1))
    return SVector(x, y)
  end
function WR2_initial_condition_constant(x, t, equations::Trixi.AbstractEquations)
    rho = 1.0
    rho_v1 = 2
    rho_v2 = 0
    rho_e = 1
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function WR2_initial_condition_polynomial(x, t, equations::Trixi.AbstractEquations)
    return SVector(2, x[1]+x[2],x[1]^2, x[2]^2+x[1]^3)
end
function WR2_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    return SVector(sin(x[1]*π)/π, cos(x[1]*π)/π, sin(x[1]*π)/π+cos(x[2]*π)/π, sin(x[1]*π)/π+cos(x[1]*π)/π)
end
function WR2_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)

    rho = sin(pi * x[1])
    rho_v1 = sin(pi * x[1])
    rho_v2 = sin(pi * x[1])
    rho_e = sin(pi * x[1])
  
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function boundary_condition_constant( u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::Trixi.AbstractEquations)
    # Far Field Conditions
    u_boundary = initial_condition(x , t, equations)

    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
  end
function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh}, equations, dg::DG, cache)
    @boundscheck begin
        @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    unsafe_wrap(Array{eltype(u_ode),ndims(mesh) + 2}, pointer(u_ode),
                (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end
###########################################################################################

# initial_condition = WR2_initial_condition_polynomial
initial_condition = WR2_initial_condition_trigonometric
# initial_condition =   WR2_initial_condition_convergence_test

N = 10
c = 128
cells_per_dimension = (c, c)
coordinates_min = (0.0, 0.0)
coordinates_max = (2.0,  2.0)

boundary_conditions = boundary_condition_constant
# boundary_conditions = boundary_condition_periodic
# boundary_conditions = (x_neg=boundary_condition_x,
#                        x_pos=boundary_condition_x,
#                        y_neg=boundary_condition_periodic,
#                        y_pos=boundary_condition_periodic)

eq = Trixi.AuxiliaryEquation()
equations = CompressibleEulerEquations2D(1.4, viscous = true)

# mesh = CurvedMesh(cells_per_dimension, mapping1zu1, periodicity = false)
mesh = CurvedMesh(cells_per_dimension, mapping, periodicity = false)
# mesh = CurvedMesh(cells_per_dimension, mappingLin, periodicity=true) # for different boundary conditions

volume_integral = VolumeIntegralWeakForm()
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(N)
dg = DGSEM(basis, surface_flux, volume_integral)

solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, (0.0, 0.0))

u = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

@unpack q1 = semi.cache
@unpack q2 = semi.cache
t = 0
Trixi.calc_nabla_u!(
    q1, q2, u, t, mesh, eq,
    boundary_conditions,
    dg, semi.cache)


    # ABleitung der initial condition convergence test

function ABL_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)

    rho = pi * cos(pi * x[1])
    rho_v1 = pi * cos(pi * x[1])
    rho_v2 = pi * cos(pi * x[1])
    rho_e = pi * cos(pi * x[1])
    
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function ABL_initial_condition_polynomial(x, t, equations::Trixi.AbstractEquations)
  return SVector(0, 1, x[1]*2, 3*x[1]^2)
end
function ABL_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    # braucht hohe Auflösung !
    return SVector(cos(x[1]*π), -sin(x[1]*π), cos(x[1]*π), cos(x[1]*π)-sin(x[1]*π))
end
solver = DGSEM(basis, surface_flux, volume_integral)
initial_condition = ABL_initial_condition_trigonometric # ABL_initial_condition_convergence_test # ABL_initial_condition_polynomial # 
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, (0.0, 0.0))

u_ABL = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

for vgl = 1:4
    println("conservative variable $vgl max error: ", maximum(abs.(u_ABL[vgl, :, :, :]-q1[vgl, :, :, :])))
end
