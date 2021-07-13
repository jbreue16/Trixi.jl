using Base: init_active_project
using Trixi
# Testumgebung, um die Korrektheit von q=∇u zu testen
# 

######## notwendige funktionen die geladen werden müssen ##############################################
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
  function mappingCos(xi_, eta_)
    xi = xi_ 
    eta = eta_ 
    x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
    y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)
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
    return SVector(2, x[1], x[1]^2, x[1]^3)
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

# Ableitungen
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
function ABL_initial_condition_constant(x, t, equations::Trixi.AbstractEquations)
    return SVector(0, 0, 0, 0)
  end
function ABL_initial_condition_trigonometric(x, t, equations::Trixi.AbstractEquations)
    # (sin(x[1]*π)/π, cos(x[1]*π)/π, sin(x[1]*π)/π+cos(x[2]*π)/π, sin(x[1]*π)/π+cos(x[1]*π)/π)
    return SVector(cos(x[1]*π), -sin(x[1]*π), cos(x[1]*π), cos(x[1]*π)-sin(x[1]*π))
end


function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh}, equations, dg::DG, cache)
    @boundscheck begin
        @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    unsafe_wrap(Array{eltype(u_ode),ndims(mesh) + 2}, pointer(u_ode),
                (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end
###########################################################################################

N_vec = [3, 5, 7, 9, 11, 13, 17]
NQ_vec = [1, 2, 4, 8, 16, 32]

# boundary_conditions = boundary_condition_periodic
boundary_conditions = boundary_condition_constant
eq = Trixi.AuxiliaryEquation()
equations = CompressibleEulerEquations2D(1.4)

# initial_condition = WR2_initial_condition_constant
initial_condition = WR2_initial_condition_polynomial
# initial_condition = WR2_initial_condition_trigonometric
# initial_condition =   WR2_initial_condition_convergence_test

# initial_condition2 = ABL_initial_condition_convergence_test # 
# initial_condition2 = ABL_initial_condition_constant 
initial_condition2 = ABL_initial_condition_polynomial  

for k in NQ_vec
# for N in N_vec

    N = 5
    # k = 1

    cells_per_dimension = (k, k)
    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0,  1.0)

    mesh = CurvedMesh(cells_per_dimension, mapping1zu1, periodicity = false)
    # mesh = CurvedMesh(cells_per_dimension, mappingCos, periodicity = true)

    volume_integral = VolumeIntegralWeakForm()
    surface_flux = flux_lax_friedrichs
    basis = LobattoLegendreBasis(N)
    dg = DGSEM(basis, surface_flux, volume_integral)

    solver = DGSEM(basis, surface_flux, volume_integral)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions) # , boundary_conditions=boundary_conditions
    ode = semidiscretize(semi, (0.0, 0.0))

    u = wrap_array(ode.u0, mesh, equations, solver, semi.cache)

    q1 = zeros(size(u, 1), size(u, 2), size(u, 3), size(u, 4))
    q2 = zeros(size(u, 1), size(u, 2), size(u, 3), size(u, 4))
    t = 0
    Trixi.calc_nabla_u!(
    q1, q2, u, t, mesh, eq,
    boundary_conditions,
    dg, semi.cache)

    solver2 = DGSEM(basis, surface_flux, volume_integral)
    semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition2, solver2, boundary_conditions=boundary_conditions)
    ode2 = semidiscretize(semi2, (0.0, 0.0))

    u_ABL = wrap_array(ode2.u0, mesh, equations, solver2, semi2.cache)


    # Test for constant inital condition
    # q1,q1 - #var - #nodes_x - #nodes_y - #cells
    # println(maximum(abs.(q1[1,1:N + 1,:,:] - u_ABL[1,1:N + 1,:,:])))
    # println(maximum(abs.(q1[2,1:N + 1,:,:] - u_ABL[2,1:N + 1,:,:])))
    # println(maximum(abs.(q1[3,1:N + 1,:,:] - u_ABL[3,1:N + 1,:,:])))
    # println(maximum(abs.(q1[4,1:N + 1,:,:] - u_ABL[4,1:N + 1,:,:])))
    for vgl = 1:4
        println("N = $N, k = $k: conservative variable $vgl max error: ", maximum(abs.(u_ABL[vgl, 1, N+1, :]-q1[vgl, 1, N+1, :])))
    end

end