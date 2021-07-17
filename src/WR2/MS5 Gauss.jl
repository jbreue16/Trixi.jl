using OrdinaryDiffEq
using Trixi
using Plots


# Vgl. Gauss on an cartesian Grid with coordinates (-1,-1),(1,1) with periodic BC
###############################################################################
CFL = 0.01
tspan = (0.0, 6)
N = 9
c = 2
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

function initial_condition_density_pulse_new(x, t, equations::CompressibleEulerEquations2D)
    rho = 1 + exp(-(x[1]^2 + x[2]^2)) / 2
    v1 = 0
    v2 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 1
    rho_e = p / (equations.gamma - 1) + 1 / 2 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e)
    end


equations = CompressibleEulerEquations2D(1.4, viscous = true, mu = mu)

initial_condition = initial_condition_density_pulse_new

# boundary_conditions = (x_neg=boundary_condition_periodic,
#                        x_pos=boundary_condition_periodic,
#                        y_neg=boundary_condition_periodic,
#                        y_pos=boundary_condition_periodic)


cells_per_dimension = (c, c)
mesh = CurvedMesh(cells_per_dimension, (-1.0, -1.0), (1.0, 1.0))

surface_flux = flux_lax_friedrichs #FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive))
basis = LobattoLegendreBasis(N)
volume_flux  = flux_chandrashekar 
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver) #,boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                      # extra_analysis_errors=(:conservation_error,),
                                      # extra_analysis_integrals=(entropy, energy_total,
                                      # energy_kinetic, energy_internal)
                                      )

alive_callback = AliveCallback(analysis_interval=analysis_interval)
#visualization = VisualizationCallback(interval=28, plot_creator=Trixi.save_plot)
visualization = VisualizationCallback(interval=500,variable_names=["rho"], plot_creator=Trixi.save_plot, clims=(1.0, 1.5))    

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
