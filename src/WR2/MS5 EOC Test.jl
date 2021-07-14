using Base: Float64
using OrdinaryDiffEq
using Trixi
using Plots


# Vgl. elixir_euler_free_stream_curved
###############################################################################
CFL = 0.5          # 2
tspan = (0.0, 2.0)
N = 3
c = 8
mu = 0.0

# Source terms and initial conditions for EOC tests
function initial_condition_convergence_test_viscous(x, t, equations::Trixi.AbstractEquations)
    # μ = equations.mu
    # Pr = equations.Pr
    # κ = equations.gamma
    # k = 2 # k = c_p * μ / Pr
    # k = 114 * μ / Pr
    k = 0
    c = 4 
    ω = 0.5
    ini = sin(k * (x[1] + x[2]) - ω * t) + c

    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2

    return SVector(rho, rho_v1, rho_v2, rho_e)
end

@inline function source_terms_convergence_test_viscous(u, x, t, equations::Trixi.AbstractEquations)
    # Same settings as in `initial_condition`
    μ = equations.mu
    Pr = equations.Pr
    κ = equations.gamma
    # k = 2 # k = c_p * μ / Pr
    k = 114 * μ / Pr
    # k = 0
    c = 4 
    ω = 0.5

    x, y = x
    co = cos(k * (x + y) - ω * t)
    si = sin(2 * k * (x + y) - 2 * ω * t)
    tmp1 = 2 * k - ω
    tmp2 = - ω + 2 * k * c * κ - k * κ + 3 * ω - 2 * k * c
    tmp3 = k * κ - k
    tmp4 = 4 * k * c * κ + 2 * k - 2 * k * κ - 2 * ω * c
    tmp5 = 2 * k * κ - ω

    du1 = co * tmp1
    du2 = co * tmp2 + si * tmp3
    du3 = co * tmp2 + si * tmp3
    du4 = co * tmp4 + si * tmp5 + sin(k * (x + y) - ω * t) * (2 * k^2 * μ * κ / Pr)

    return SVector(du1, du2, du3, du4)
end

function boundary_condition_convergence_test_viscous(u_inner, orientation, direction, x, t,
                                          surface_flux_function, equations::Trixi.AbstractEquations)
    u_boundary = initial_condition_convergence_test_viscous(x, t, equations)

    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

function solution(x, t)
    # μ = equations.mu
    # Pr = equations.Pr
    # κ = equations.gamma
    # k = 2 # k = c_p * μ / Pr
    # k = 114 * μ / Pr
    k = 0
    c = 4 
    ω = 0.5
    ini = sin(k * (x[1] + x[2]) - ω * t) + c

    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2

    return SVector(rho, rho_v1, rho_v2, rho_e)
end

function wrap_array(u_ode::AbstractVector, mesh::Union{TreeMesh{2},CurvedMesh{2},UnstructuredQuadMesh}, equations, dg::DG, cache)
    @boundscheck begin
        @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    unsafe_wrap(Array{eltype(u_ode),ndims(mesh) + 2}, pointer(u_ode),
                (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end

function WR2_initial_condition_convergence_test(x, t, equations::Trixi.AbstractEquations)
    c = 2
    A = 0.1
    L = 2
    f = 1 / L
    ω = 2 * pi * f
    ini = c + A * sin(ω * (x[1] + x[2] - t))
    
    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2
    
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
    

@inline function WR2_source_terms_convergence_test(u, x, t, equations::Trixi.AbstractEquations)
    # Same settings as in `initial_condition`
    c = 2
    A = 0.1
    L = 2
    f = 1 / L
    ω = 2 * pi * f
    γ = equations.gamma
    
    x1, x2 = x
    si, co = sincos((x1 + x2 - t) * ω)
    tmp1 = co * A * ω
    tmp2 = si * A
    tmp3 = γ - 1
    tmp4 = (2 * c - 1) * tmp3
    tmp5 = (2 * tmp2 * γ - 2 * tmp2 + tmp4 + 1) * tmp1
    tmp6 = tmp2 + c
    
    du1 = tmp1
    du2 = tmp5
    du3 = tmp5
    du4 = 2 * ((tmp6 - 1) * tmp3 + tmp6 * γ) * tmp1
    
    return SVector(du1, du2, du3, du4)
end

function WR2_boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::Trixi.AbstractEquations)
u_boundary = WR2_initial_condition_convergence_test(x, t, equations)

# Calculate boundary flux
if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
else # u_boundary is "left" of boundary, u_inner is "right" of boundary
flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
end

return flux
end

# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

# initial_condition = initial_condition_convergence_test_viscous
# source_terms = source_terms_convergence_test_viscous
# boundary_conditions = boundary_condition_convergence_test_viscous

initial_condition = WR2_initial_condition_convergence_test
source_terms = WR2_source_terms_convergence_test
boundary_conditions = WR2_boundary_condition_convergence_test

surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Mapping
function mapping1zu1(xi_, eta_)

    x = xi_ 
    y = eta_

    return SVector(x, y)
end

cells_per_dimension = (c, c)

# mesh = CurvedMesh(cells_per_dimension, mappingCos, periodicity = true)
# mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))
# mesh = CurvedMesh(cells_per_dimension, (-pi/2, -pi/2), (pi/2, pi/2))
mesh = CurvedMesh(cells_per_dimension, (0.0, 0.0), (2.0, 2.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                      # extra_analysis_errors=(:conservation_error,),
                                      # extra_analysis_integrals=(entropy, energy_total,
                                      # energy_kinetic, energy_internal)
                                      )

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        #alive_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


###############################################################################
# 

# solver = DGSEM(basis, surface_flux, volume_integral)
# initial_condition_sol = initial_condition_convergence_test_viscous
# semi_sol = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sol, solver, boundary_conditions=boundary_conditions)
# ode_sol = semidiscretize(semi_sol, (0.1, 0.1))

# u_sol = wrap_array(ode_sol.u0, mesh, equations, solver, semi_sol.cache)
# u_dg = wrap_array(sol[2], mesh, equations, solver, semi.cache)
# u_diff = abs.(u_sol-u_dg)

# u_diff_max1 = maximum(abs.(u_sol[1,:,:,:]-u_dg[1,:,:,:]))
# u_diff_max2 = maximum(abs.(u_sol[2,:,:,:]-u_dg[2,:,:,:]))
# u_diff_max3 = maximum(abs.(u_sol[3,:,:,:]-u_dg[3,:,:,:]))
# u_diff_max4 = maximum(abs.(u_sol[4,:,:,:]-u_dg[4,:,:,:]))

# maximum(abs.(ode_sol.u0-sol.u[2]))
maximum(abs.(sol.u[1] - sol.u[2]))