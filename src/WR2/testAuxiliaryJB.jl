using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# Test mit Polynomen funktioniert -> Achte aber auf stetige RB
# test mit Trigonometrischen Funktionen Funktioniert
# Randwerte vom surface integral sind viel zu hoch, berechnen aber mit Mittelung den korrekten Wert
# In der "normalen" semidiskretisierung/rhs! entstehen auch an den Rändern so hohe Werte
# Krumme Gitter muss noch getestet und was für gemacht werden !

######## notwendige funktionen die geladen werden müssen ##############################################
# mapping O-mesh 
function mapping(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = 5 * (2 + ξ ) * cos(π * (η + 1))
    y = 5 * (2 + ξ ) * sin(π * (η + 1))
    return SVector(x, y)
end
function mappingLin(xi_, eta_)
  x = xi_ 
  y = eta_
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

initial_condition = WR2_initial_condition_polynomial
# initial_condition = WR2_initial_condition_trigonometric
# initial_condition =   WR2_initial_condition_convergence_test

N = 12
c = 2
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
equations = CompressibleEulerEquations2D(1.4)

# mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity = false)
# mesh = CurvedMesh(cells_per_dimension, mapping, periodicity=true)
mesh = CurvedMesh(cells_per_dimension, mappingLin, periodicity=false) # for different boundary conditions

volume_integral = VolumeIntegralWeakForm()
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(N)
dg = DGSEM(basis, surface_flux, volume_integral)

solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver) #, boundary_conditions=boundary_conditions
ode = semidiscretize(semi, (0.0, 0.0))

u = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

q = zeros(2, size(u, 1), size(u, 2), size(u, 3), size(u, 4))
t = 0
Trixi.calc_nabla_u!(
    q, u, t, mesh, eq,
    boundary_conditions,
    dg, semi.cache)

dg = DGSEM(basis, flux_central, volume_integral)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, eq, initial_condition, solver) #, boundary_conditions=boundary_conditions
ode = semidiscretize(semi, (0.0, 0.0))
du = q[1,:,:,:,:]
u_hm = Trixi.rhs!(du, u, t,
    mesh, eq,
    initial_condition, boundary_conditions,
    # source_terms,
    dg, semi.cache)


    # ABleitung der initial condition convergence test

function ABL_initial_condition_convergence_test(x, t, equations::Trixi.AuxiliaryEquation)

    rho = pi * cos(pi * x[1])
    rho_v1 = pi * cos(pi * x[1])
    rho_v2 = pi * cos(pi * x[1])
    rho_e = pi * cos(pi * x[1])
    
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function ABL_initial_condition_convergence_test_basis(x, t)
  c = 2
  A = 0.1
  L = 2
  f = 1 / L
  ω = 2 * pi * f # = pi
  ini = A * ω * cos(ω * (x[1] + x[2] - t))
  
  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  # rho_e = 2 * ini * A * ω^2 * (-) * sin(ω * (x[1] + x[2] - t))
  rho_e = 2 * ini * (A * sin(ω * (x[1] + x[2] - t) + c))

  rho = pi * cos(pi * x[1])
  rho_v1 = pi * cos(pi * x[1])
  rho_v2 = pi * cos(pi * x[1])
  rho_e = pi * cos(pi * x[1])
  
  return SVector(rho, rho_v1, rho_v2, rho_e)
end
# solver2 = DGSEM(basis, surface_flux, volume_integral)
# initial_condition2 = ABL_initial_condition_convergence_test # 
# semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition2, solver2, boundary_conditions=boundary_conditions)
# ode2 = semidiscretize(semi, (0.0, 0.0))

# u_ABL = wrap_array(ode2.u0, mesh, equations, solver2, semi2.cache)

# maximum(abs.(q[1,1,2:N,:,:]))
