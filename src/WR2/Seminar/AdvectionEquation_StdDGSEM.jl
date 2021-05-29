using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = 2.0
N = 5
NQ = 16
time_start = 0.0
time_end = 2
equations = LinearScalarAdvectionEquation1D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=N, surface_flux=flux_lax_friedrichs)

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

# Exakte Loesung
function exakte_loesung(x, t, a)
  if (x[1]-a*t) <= 0.75 && (x[1]-a*t) >= 0.25
    u = 1
  else
    u = 0 
  end
  return u
end

# Exakte Loesung mit T=2 und a=2
function exakte_loesung_tsafe(x, t, a)
  if x[1] <= 0.75 && x[1] >= 0.25
    u = 1
  else
    u = 0 
  end
  return u
end


# Create curved mesh with 16 cells
#mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
coordinates_min = 0 # minimum coordinate
coordinates_max =  1 # maximum coordinate

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=1,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_shock, solver)


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
callbacks = CallbackSet(stepsize_callback,analysis_callback)


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
sol_e = exakte_loesung_tsafe.(pd.x, time_end, advectionvelocity)

# Plot
# plot(sol)
plot(pd.x,[pd.data sol_e])