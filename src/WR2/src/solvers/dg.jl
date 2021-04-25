abstract type DG{S, N} end


DG(N, Nq, S, equation, riemann_solver; kwargs...) = DG_1D(N, Nq, S, equation, riemann_solver; kwargs...)


@inline polydeg(::DG{S, N}) where {S, N} = N
@inline nvariables(::DG{S, N}) where {S, N} = S

@inline get_node_vars(u, dg, indices...) = SVector(ntuple(s -> u[s, indices...], nvariables(dg)))


# Nonlinear semidiscretize for lobatto only
function semidiscretize(dg, initial_condition; CFL=1.4, tspan=(0, 1))
    dt(u) = @timeit "compute timestep" timestep(u, CFL, dg)

    u0 = discretize_initial_condition(dg, initial_condition)
    return ODEProblem(rhs!, u0, tspan, dg; dt=dt)
end


function rhs!(du, u, dg, t)
    du .= zero(eltype(du))

    @timeit "calculate volume integral" calc_volume_integral!(du, u, dg)

    @timeit "calculate interface flux" calc_interface_flux!(u, dg)

    @timeit "calculate boundary flux" calc_boundary_flux!(u, dg)

    @timeit "calculate surface integral" calc_surface_integral!(du, u, dg)

    @timeit "inverse jacobian" calc_inverse_jacobian!(du, dg)

    @timeit "source term" calc_source_term!(du, u, t, dg)
end