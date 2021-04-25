module Numerik4

using LinearAlgebra
using StaticArrays
using SparseArrays
using PrettyTables
using Plots
using Printf
using TimerOutputs
using Polynomials

timeit_debug_enabled() = true

include("interpolation/interpolation.jl")
include("quadrature/quadrature.jl")

include("ode_solver/rk.jl")
include("ode_solver/rk_linear.jl")

include("equations/equations_1d.jl")
include("equations/equations_2d.jl")

include("solvers/dg.jl")
include("solvers/dg_1d.jl")
include("solvers/dg_2d.jl")
include("solvers/dg_linear.jl")

include("exercises/project1.jl")
include("exercises/project2.jl")
include("exercises/project3.jl")
include("exercises/project4.jl")

end # module
