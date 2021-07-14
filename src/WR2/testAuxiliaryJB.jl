using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# Test mit Polynomen funktioniert -> Achte aber auf stetige RB
# test mit Trigonometrischen Funktionen Funktioniert
# Krumme Gitter sehen jetzt auch gut aus.
# ABER: welche boundary conditions für die Hilfsgleichung? unstetige, zb periodisch führen zu
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
function mappingTri(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = cos(pi* ξ)
    y = η
    return SVector(x, y)
  end
  function mappingLin(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = 1.2* ξ
    y = η
    return SVector(x, y)
  end
function WR2_initial_condition_constant(x, t, equations::Trixi.AbstractEquations)
    rho = 1.0
    rho_v1 = 0.1 
    rho_v2 = -0.2 
    rho_e = 10.0
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function WR2_initial_condition_polynomial(x, t, equations::Trixi.AbstractEquations)
    return SVector(2, x[1] + x[2], x[1]^2, x[2]^2+x[1]^3)
end
function ABLx_initial_condition_polynomial(x, t, equations::Trixi.AbstractEquations)
    return SVector(0, 1, x[1]*2, 3*x[1]^2) end
function ABLy_initial_condition_polynomial(x, t, equations::Trixi.AbstractEquations)
    return SVector(0, 1, 0, 2*x[2]) end 


function WR2_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    return SVector(sin(x[1]*π)/π, cos(x[1]*π)/π, sin(x[2]*π)/π + sin(x[1]*π)/π, sin(x[2]*π)/π + cos(x[2]*π)/π + sin(x[1]*π)/π+cos(x[1]*π)/π)
end
    # braucht hohe Auflösung !
function ABLx_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    return SVector(cos(x[1]*π), -sin(x[1]*π), cos(x[1]*π), cos(x[1]*π)-sin(x[1]*π)) end
function ABLy_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    return SVector(0, 0, cos(x[2]*π), cos(x[2]*π)-sin(x[2]*π)) end


function WR2_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    rho = sin(pi * x[1])
    rho_v1 = sin(pi * x[1])
    rho_v2 = sin(pi * x[1])
    rho_e = sin(pi * x[1])
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function ABLx_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    rho = pi * cos(pi * x[1])
    rho_v1 = pi * cos(pi * x[1])
    rho_v2 = pi * cos(pi * x[1])
    rho_e = pi * cos(pi * x[1])
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function ABLy_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    rho = 0
    rho_v1 = 0
    rho_v2 = 0
    rho_e = 0
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

# initial_condition = WR2_initial_condition_constant
initial_condition = WR2_initial_condition_polynomial
# initial_condition = WR2_initial_condition_trigonometric
# initial_condition =   WR2_initial_condition_convergence_test
# Zum Vergleich mit exakter Ableitung !
initial_condition2 = ABLx_initial_condition_polynomial # ABLx_initial_condition_trigonometric #  ABLx_initial_condition_convergence_test #  
initial_condition3 = ABLy_initial_condition_polynomial # ABLy_initial_condition_trigonometric #   ABLy_initial_condition_convergence_test # 
N = 4
c = 16
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
# mesh = CurvedMesh(cells_per_dimension, mappingLin, periodicity = true) #mappingTri

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

# Zum vergleich mit exakter Ableitung !
solver = DGSEM(basis, surface_flux, volume_integral) 
semi_x = SemidiscretizationHyperbolic(mesh, equations, initial_condition2, solver, boundary_conditions=boundary_conditions)
ode_x = semidiscretize(semi_x, (0.0, 0.0))
semi_y = SemidiscretizationHyperbolic(mesh, equations, initial_condition3, solver, boundary_conditions=boundary_conditions)
ode_y = semidiscretize(semi_y, (0.0, 0.0))

u_ABLx = wrap_array(ode_x.u0, mesh, equations, solver, semi.cache)
u_ABLy = wrap_array(ode_y.u0, mesh, equations, solver, semi.cache)

for vgl = 1:4
    println("q1 cons.var. $vgl max error: ", maximum(abs.(u_ABLx[vgl, :, :, :]-q1[vgl, :, :, :])))
end
for vgl = 1:4
    println("q2 cons.var. $vgl max error: ", maximum(abs.(u_ABLy[vgl, :, :, :]-q2[vgl, :, :, :])))
end
