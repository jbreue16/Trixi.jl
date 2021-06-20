
using OrdinaryDiffEq
using Trixi
using Plots



tspan = (0.0, 0.1)
CFL = 0.8
cells_per_dimension = (16, 16) # 40 und 5 bis knapp 8; 32 und 5 bis gut 7
N = 5
# plot_creator=Trixi.save_plot #, clims=(0,1) speicherung, skala
visualization = VisualizationCallback(interval=1000)
save_analysis = false # false is default
equations = CompressibleEulerEquations2D(1.4)

function WR2_initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = 0.38
    rho_v2 = 0
    rho_e = 25
    return SVector(rho, rho_v1, rho_v2, rho_e)
  end
initial_condition = WR2_initial_condition_constant


surface_flux = FluxPlusDissipation(flux_chandrashekar, DissipationLocalLaxFriedrichs(max_abs_speed_naive)) # flux_lax_friedrichs #  
volume_flux  =  flux_chandrashekar
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
basis = LobattoLegendreBasis(N)
# volume_integral = VolumeIntegralWeakForm()
# surface_flux = flux_lax_friedrichs
solver = DGSEM(basis, surface_flux, volume_integral)



###############################################################################
# semidiscretization of the compressible Euler equations



# mapping O-mesh
function mapping(xi_, eta_)

    ξ = xi_ 
    η = eta_

    x = 5 * (2 + ξ ) * cos(π * (η + 1))
    y = 5 * (2 + ξ ) * sin(π * (η + 1))

    return SVector(x, y)
end


function boundary_condition_stream(u_inner, orientation, direction, x, t,
    surface_flux_function,
    equations::CompressibleEulerEquations2D)
    rho = 1.0
    rho_v1 = 0.38
    rho_v2 = 0
    rho_e = 25
    # if -6 < x[2] < 6 && (x[1] < 0)# -6 < x[2] < 6 && 
    #     # if t < π/2
    #         u_boundary = SVector(rho, rho_v1, rho_v2, rho_e)#*sin(t*(π/2)) #cos(π/2*x[1])* v
    #     # else 
    #     #     u_boundary = SVector(rho, rho_v1, rho_v2, rho_e) 
    #     # end
    # else
        u_boundary = WR2_initial_condition_constant(x, t, equations)
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
    rho = 1.0
    rho_v1 = 0.38
    rho_v2 = 0
    rho_e = 25
    u_boundary = SVector(rho, rho_v1, rho_v2, rho_e)
  
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

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=CFL)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=save_analysis,
                                                 extra_analysis_errors=(:conservation_error,),
                                                 extra_analysis_integrals=(entropy, energy_total,
                                                                           energy_kinetic, energy_internal)
                                                
                                                )

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback
                        ,visualization 
                        # ,save_solution            
                        )


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

# compute the Mach number
# i = 1
# mach = zeros(16^2*4^2) # cells^2*(N+1)^2
# while i < 16384 #cells^2*(N+1)^2*4 every four steps
#     # every four steps is the same conservative variable
#     # rho = sol.u[i]
#     # v1_rho = sol.u[i+1]
#     # v2_rho = sol.u[i+2]
#     # e_rho = sol.u[i+3]
#     prims = cons2prims(sol.u[1][i:i+3])
#     v_norm = sqrt(prims[2]^2+prims[3]^2)
    
#     c = sqrt(1.4*(prims[1]/prims[4]))#  dichte durch druck
#     mach[i] = v_norm/c 
#     i = i+4
# end