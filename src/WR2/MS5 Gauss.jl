using OrdinaryDiffEq
using Trixi
using Plots


# Vgl. Gauss on an cartesian Grid with coordinates (-1,-1),(1,1) with periodic BC
###############################################################################
CFL = 0.005       # 2
tspan = (0.0, 5)
N = 3
c = 16
mu = 0.001


function initial_condition_gauss(x, t, equations::CompressibleEulerEquations2D)
    rho = 1 + exp(-(x[1]^2 + x[2]^2)) / 2
    v1 = 0
    v2 = 0
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 1
    rho_e = 1 + exp(-(x[1]^2 + x[2]^2)) / 2
    return SVector(rho, rho_v1, rho_v2, rho_e)
end


equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

initial_condition = initial_condition_density_pulse

# boundary_conditions = (x_neg=boundary_condition_periodic,
#                        x_pos=boundary_condition_periodic,
#                        y_neg=boundary_condition_periodic,
#                        y_pos=boundary_condition_periodic)


cells_per_dimension = (c, c)
mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))

surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver) #,boundary_conditions=boundary_conditions)


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
#visualization = VisualizationCallback(interval=28, plot_creator=Trixi.save_plot)
visualization = VisualizationCallback(interval=60,variable_names=["rho"], plot_creator=Trixi.save_plot)    

save_solution = SaveSolutionCallback(interval=50,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        visualization,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
