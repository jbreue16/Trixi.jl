using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = 2.0     # Velocity
N = 7                       # Polynomial degree (3, 5, 7)
NQ = 16                      # Cells (4, 8, 16)
time_start = 0.0            # Start time
time_end = 2                # End time

# Equation
equations = LinearScalarAdvectionEquation1D(advectionvelocity)
surface_flux = flux_lax_friedrichs

# Boundary Condition
boundary_condition = boundary_condition_periodic

# Create DG solver with polynomial degree = N and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=N, surface_flux=surface_flux)

# Coordinates 
coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (NQ,)

# Initial Condition
function initial_condition_shock(x, t, equation::LinearScalarAdvectionEquation1D)
    if x[1] <= 0.75 && x[1] >= 0.25
      u = 1
    else
      u = 0 
    end
    return SVector(u)
end


# Exakte Loesung mit T=2 und a=2
function exact_solution(x, t, a)
  if x[1] <= 0.75 && x[1] >= 0.25
    u = 1
  else
    u = 0 
  end
  return u
end


# Create curved mesh with 16 cells
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_shock, solver, boundary_conditions=boundary_condition)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (time_start, time_end));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.5)

analysis_callback = AnalysisCallback(semi, interval=100, extra_analysis_errors=(:conservation_error,))

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# Exakte Loesung
pd = PlotData1D(sol)
sol_e = exact_solution.(pd.x, time_end, advectionvelocity)

# Plot
plot(pd.x,[pd.data sol_e], title = "Zeitpunkt T = 2", label = ["DGSEM" "Exakte Loesung"] )

# Mesh
#plot!(getmesh(pd))