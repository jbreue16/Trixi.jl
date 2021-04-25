# Define problems similar to OrdinaryDiffEq.jl to be able to switch later
mutable struct ODEProblem{F, U, P, K}
    f::F
    u0::U
    tspan::Tuple{Float64, Float64}
    p::P
    kwargs::K

    function ODEProblem(f, u0, tspan, p=0; kwargs...)
        # If f has 3 arguments (don't know why nargs is 4)
        if first(methods(f)).nargs == 4
        #if !applicable(f, 4)
            # Create inplace version of f
            function f!(du, u, p, t)
                du .= f(u, p, t)

                return nothing
            end
        else
            # f is already inplace
            f! = f
        end

        new{typeof(f!), typeof(u0), typeof(p), typeof(kwargs)}(f!, u0, tspan, p, kwargs)
    end
end


ODEProblem(f, u0::Vector{A} where A <: Integer, tspan, p; kwargs...) = ODEProblem(f, convert(Vector{Float64}, u0), tspan, p; kwargs...)


function solve(ode)
    @assert haskey(ode.kwargs, :dt) "Ein Zeitschritt muss übergeben werden!"
    solve(ode, ode.kwargs[:dt])
end


@inline A_carpenter_kennedy() = @SVector [0.0, -567301805773/1357537059087, -2404267990393/2016746695238,
                        -3550918686646/2091501179385, -1275806237668/842570457699]

@inline B_carpenter_kennedy() = @SVector [1432997174477/9575080441755, 5161836677717/13612068292357,
                        1720146321549/2090206949498, 3134564353537/4481467310338,
                        2277821191437/14882151754819]

@inline c_carpenter_kennedy() = @SVector [0.0, 1432997174477/9575080441755, 2526269341429/6820363962896,
                        2006345519317/3224310063776, 2802321613138/2924317926251]


function solve(ode, timestep::T) where T
    if timestep isa Real
        timestep_func(u) = timestep
    else
        # timestep is already a function
        timestep_func = timestep
    end

    A = A_carpenter_kennedy()
    B = B_carpenter_kennedy()
    c = c_carpenter_kennedy()

    u = ode.u0
    g = similar(u)
    u_tmp = similar(u)
    f! = ode.f
    p = ode.p
    t = convert(Float64, ode.tspan[1])

    while t < ode.tspan[2]
        dt = min(timestep_func(u), ode.tspan[2] - t)
        f!(g, u, p, t)
        for stage in 1:5
            f!(u_tmp, u, p, t + c[stage] * dt)
            for i in eachindex(u)
                g[i] = A[stage] * g[i] + u_tmp[i]
                u[i] += B[stage] * dt * g[i]
            end
        end

        t += dt
    end

    ode.tspan = (ode.tspan[2], ode.tspan[2])

    return u
end
