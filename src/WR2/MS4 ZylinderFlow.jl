
using OrdinaryDiffEq
using Trixi
using Plots

v1 = 0.1 # MACH: v1 = 0.38 -> 10.16%, v1 = 0.1 -> 2.67? %
tspan = (0.0, 5) # ≈ 20 für stabilen Zustand
CFL = 0.8
cells_per_dimension = (16, 16)

visualization = VisualizationCallback(interval=200,variable_names=["v1","v2"], plot_creator=Trixi.save_plot)# clims=(-0.5,0.5), 

function WR2_initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = v1
    rho_v2 = 0
    rho_e = 25.0
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
initial_condition = WR2_initial_condition_constant


surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) # flux_lax_friedrichs #  
volume_flux  = flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(4)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# mapping O-mesh
function mapping(xi_, eta_)

    ξ = xi_ 
    η = eta_

    size = 2    # gebietsgröße
    d = 2       # ≈ durchmesser
    x = size * (d + ξ ) * cos(π * (η + 1))
    y = size * (d + ξ ) * sin(π * (η + 1))

    return SVector(x, y)
end


function boundary_condition_stream(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    # Far Field Conditions
    # if (x[1] < 1)
    #     # if t < π/2
    #     #     u_boundary = SVector(1, 0.38 , 0, 10) #+ 1* sin(π*t/10)*10
    #     # else 
    #     #     u_boundary = SVector(1, 0.38, 0, 10) 
    #     # end
    # else
        u_boundary = SVector(1, v1, 0, 25)
    # end
    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
end

function boundary_condition_constant_farfield( u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    # Far Field Conditions
    u_boundary = SVector(1, v1, 0, 10)
  
    # Calculate boundary flux
    if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
end
  
function boundary_condition_freeslip(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    # Freeslip wall Conditions
    rho, v1, v2, p = cons2prim(u_inner, equations)
    u_boundary = prim2cons(SVector(rho, -v1, -v2, p), equations)
  
    # Calculate boundary flux
    if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # direction == 4 # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
  
    return flux
end
        

mesh = CurvedMesh(cells_per_dimension, mapping,periodicity=(false,true))

#mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity=(false, true))

# Implement boundary conditions
# x neg and x_pos are not periodic 
# boundary_conditions = boundary_condition_periodic
boundary_conditions = (x_neg=boundary_condition_freeslip,
                       x_pos=boundary_condition_stream,
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.



summary_callback = SummaryCallback()

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                                 extra_analysis_errors=(:conservation_error,),
                                                 extra_analysis_integrals=(entropy, energy_total,
                                                                           energy_kinetic, energy_internal)
                                                
                                                )

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback) #, visualization)


###############################################################################
# run the simulation



ode = semidiscretize(semi, tspan)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

# pd=PlotData2D(sol)
# b=plot(pd["rho"])
# plot!(getmesh(pd))

# plot(sol)
# # savefig(plotd, "test.png")