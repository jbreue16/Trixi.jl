using OrdinaryDiffEq
using Trixi
using Plots


######################Functions that need to be loaded ###########
# mapping O-mesh
function mapping(xi_, eta_)
    ξ = xi_ 
    η = eta_
    x = 2 * (2 + ξ ) * cos(π * (η + 1))
    y = 2 * (2 + ξ ) * sin(π * (η + 1))
    return SVector(x, y)
end
function mapping1zu1(xi_, eta_)
    x = xi_ 
    y = eta_
    return SVector(x, y)
end
function WR2_initial_condition_constant(x, t, equations::Trixi.AbstractEquations)
    rho = 1.0
    rho_v1 = 0.1 
    rho_v2 = -0.2 
    rho_e = 10.0
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

########################    Setting    ######################################
CFL = 0.01
tspan = (0.0, 0.5)
N = 3
c = 16
mu = 0.001

# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

initial_condition = WR2_initial_condition_constant
boundary_conditions = boundary_condition_constant # boundary_condition_periodic
cells_per_dimension = (c, c)
# mesh = CurvedMesh(cells_per_dimension, mapping, periodicity = true)
mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))

surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)#,boundary_conditions=boundary_conditions)



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


plot(sol)