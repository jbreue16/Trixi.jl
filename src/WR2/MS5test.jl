using OrdinaryDiffEq
using Trixi


#Laufen lassen mit sabotiertem VolumeIntegralWeakForm, um die semidiskretisierung auszugeben.
###############################################################################
CFL = 0.8           # 2
tspan = (0.0, 0.0)


equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant


surface_flux = flux_lax_friedrichs
volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(polydeg=5, surface_flux=surface_flux, volume_integral = volume_integral )

# mapping as described in the worksheet
# function mapping(xi_, eta_)

#     xi = xi_ 
#     eta = eta_
  
#     x = xi + 0.15 * cos(0.5 * pi * xi) * cos((3/2) * pi * eta)
#     y = eta + 0.15 * cos(2 * pi * xi) * cos(0.5 * pi * eta)

#     return SVector(x, y)
#   end

cells_per_dimension = (16, 16)
coordinates_min = (0.0, 0.0)
coordinates_max = (2.0,  2.0)
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
# mesh = CurvedMesh(cells_per_dimension, mapping, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
ode = semidiscretize(semi, tspan)


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false);

