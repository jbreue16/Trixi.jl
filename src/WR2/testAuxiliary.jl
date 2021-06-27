using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# Test mit Polynomen funktioniert
# test mit trigonometrischen Funktionen funktioniert
# Problem: Unstetigkeiten am Rand führt zu komischen Ableitungen. ZB Periodische RB und Polynome ? 
# Problem: Vorzeichen Fehler für ALLE Werte nach Anwendung der Jacobi !


######## notwendige funktionen die geladen werden müssen ##############################################
# mapping O-mesh 
function mapping(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = 5 * (2 + ξ ) * cos(π * (η + 1))
    y = 5 * (2 + ξ ) * sin(π * (η + 1))
    return SVector(x, y)
end
function WR2_initial_condition_constant(x, t, equations::Trixi.AbstractEquations)
    rho = 1.0
    rho_v1 = 2
    rho_v2 = 0
    rho_e = 1
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
function WR2_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    c = 2
    A = 0.1
    ω = π
    ini = c + A * sin(ω * (x[1] + x[2]))
  
    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2
  
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
  function initial_condition_polynomial(x , t, equations::Trixi.AbstractEquations)
    return SVector(x[1]+x[2], x[2], x[2]^2, x[2]^3)
  end
    function initial_condition_trigonometric(x , t, equations::Trixi.AbstractEquations)
      return SVector(sin(x[1]), cos(x[1]), sin( (x[2]+x[1])), x[1])
    end
      function ABL_trigonometric(x , t, equations::Trixi.AbstractEquations)
        return SVector(cos(x[1]), -sin(x[1]), cos( (x[2]+x[1])), 1)
      end
function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh}, equations, dg::DG, cache)
    @boundscheck begin
      @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
                (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  end
  function boundary_condition_constant_initial_farfield( u_inner, orientation, direction, x, t,
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
###########################################################################################


boundary_conditions = boundary_condition_periodic # boundary_condition_constant_initial_farfield # 
eq = Trixi.AuxiliaryEquation()
equations = CompressibleEulerEquations2D(1.4)

N = 12
c = 1
# initial_condition = initial_condition_polynomial
initial_condition = initial_condition_trigonometric
# initial_condition =  WR2_initial_condition_convergence_test
# initial_condition = WR2_initial_condition_constant 
cells_per_dimension = (c, c)
coordinates_min = (0.0, 0.0)
coordinates_max = (2,  2) # WIRD AUF 2*PI gesetzt wenn initial condition periodic gewählt ist !
if initial_condition == initial_condition_trigonometric
  coordinates_max = (2*π,  2*π)
end   
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
# mesh = CurvedMesh(cells_per_dimension, mapping, periodicity=true)

volume_integral = VolumeIntegralWeakForm()
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(N)
dg = DGSEM(basis, surface_flux, volume_integral)

# u = 
# Array{Float64,4}
# (#variablen, N, N, #Zellen)

solver = DGSEM(basis, surface_flux, volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, (0.0,0.0))

u = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

q = zeros(2, size(u,1), size(u,2), size(u,3), size(u,4))
q1 = zeros(size(u,1), size(u,2), size(u,3), size(u,4))
q2 = zeros(size(u,1), size(u,2), size(u,3), size(u,4))
t = 0
Trixi.calc_nabla_u!(
    q1, q2, u, t, mesh, eq,
    boundary_conditions,
    dg, semi.cache)
# q[1, :, :, :, :] = q1
# q[2, :, :, :, :] = q2


    # ABleitung der initial condition convergence test

    function ABL_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
      c = 2
      A = 0.1

      ini = c + A * sin(π * (x[1] + x[2]))
      ini_abl = A * π * cos(π * (x[1] + x[2]))

      rho = ini_abl
      rho_v1 = ini_abl
      rho_v2 = ini_abl
      rho_e = 2* ini * ini_abl
    
      return SVector(rho, rho_v1, rho_v2, rho_e)
    end



# Macht iwie nur scheiße, statt u0 zu bestimmen ?
# solver2 = DGSEM(basis, surface_flux, volume_integral)
# initial_condition2 = ABL_trigonometric # ABL_initial_condition_convergence_test #
# semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition2, solver2, boundary_conditions=boundary_conditions)
# ode2 = semidiscretize(semi, (0.0, 0.0))
# u_ABL = wrap_array(ode.u0, mesh, equations, solver, semi.cache)
######## u_ABL  = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(semi2)...) # u_ABL = wrap_array(ode2.u0, mesh, equations, solver2, semi2.cache)
######## compute_coefficients!(u_ABL, ABL_initial_condition_convergence_test, 0.0, semi2)


# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#     dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#     save_everystep=false);


# Test for constant inital condition
# println(maximum(q[2, :, :, 2:N-1, :]))
# println(maximum(q[1, :, 2:N-1, :, :]))