using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# Test mit Konstanten funktioniert -> q = 0
# test mit periodischer sinusfunktion muss noch geprüft werden, sieht nicht ganz falsch aus erstmal, aber vermutlich noch nicht wirklich korrekt, da
# q1 = q2 in diesem Beipiel gelten müsste, was aber nicht ganz hinhaut.. max, min ist aber gleich
# .. ahcte auch auf das mapping und so.. 

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
function WR2_initial_condition_linear(x, t, equations::Trixi.AbstractEquations)
    rho = x[1] # + 0.1
    rho_v1 = x[1]
    rho_v2 = x[1]
    rho_e = x[1] # + 0.5
    # return SVector(rho, rho_v1, rho_v2, rho_e)
    return SVector(x[1], x[1]^2, x[1]^3, x[1]^2)
end
function WR2_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    # c = 2
    # A = 0.1
    # L = 2
    # f = 1 / L
    # ω = 2 * pi * f
    # ini = c + A * sin(ω * (x[1] + x[2] - t))
  
    # rho = ini
    # rho_v1 = ini
    # rho_v2 = ini
    # rho_e = ini^2

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


# boundary_conditions = boundary_condition_periodic
boundary_conditions = boundary_condition_constant
eq = Trixi.AuxiliaryEquation()
equations = CompressibleEulerEquations2D(1.4)

N = 9
c = 2
cells_per_dimension = (c, c)
coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0,  1.0)
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max) # for periodic boundary conditions
# mesh = CurvedMesh(cells_per_dimension, mapping, periodicity=true)
# mesh = CurvedMesh(cells_per_dimension, mappingLin, periodicity=(false,true)) # for different boundary conditions

volume_integral = VolumeIntegralWeakForm()
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(N)
dg = DGSEM(basis, surface_flux, volume_integral)

# u = 
# Array{Float64,4}
# (#variablen, N, N, #Zellen)

# Implement boundary conditions
# x neg and x_pos are not periodic 
# boundary_conditions = boundary_condition_periodic
# boundary_conditions = (x_neg=boundary_condition_x,
#                        x_pos=boundary_condition_x,
#                        y_neg=boundary_condition_periodic,
#                        y_pos=boundary_condition_periodic)

solver = DGSEM(basis, surface_flux, volume_integral)
initial_condition =   WR2_initial_condition_linear # WR2_initial_condition_convergence_test # WR2_initial_condition_constant # WR2_initial_condition_linear
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, (0.0, 0.0))

u = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

q = zeros(2, size(u, 1), size(u, 2), size(u, 3), size(u, 4))
t = 0
Trixi.calc_nabla_u!(
    q, u, t, mesh, eq,
    boundary_conditions,
    dg, semi.cache)



    # ABleitung der initial condition convergence test

function ABL_initial_condition_convergence_test(x, t, equations::Trixi.AuxiliaryEquation)
    # c = 2
    # A = 0.1
    # L = 2
    # f = 1 / L
    # ω = 2 * pi * f # = pi
    # ini = A * ω * cos(ω * (x[1] + x[2] - t))
    
    # rho = ini
    # rho_v1 = ini
    # rho_v2 = ini
    # # rho_e = 2 * ini * A * ω^2 * (-) * sin(ω * (x[1] + x[2] - t))
    # rho_e = 2 * ini * (A * sin(ω * (x[1] + x[2] - t) + c))

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
solver2 = DGSEM(basis, surface_flux, volume_integral)
initial_condition2 = ABL_initial_condition_convergence_test # 
semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition2, solver2, boundary_conditions=boundary_conditions)
ode2 = semidiscretize(semi, (0.0, 0.0))

u_ABL = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#     dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#     save_everystep=false);


# Test for constant inital condition
# q1,q1 - #var - #nodes_x - #nodes_y - #cells
println(maximum(q[2, :, :, 2:N - 1, :]))
println(maximum(q[1, :, 2:N - 1, :, :]))