struct ODEProblemLinear{AT, U, T, K}
    A::AT
    u0::U
    tspan::T
    kwargs::K

    ODEProblemLinear(A, u0, tspan; kwargs...) = new{typeof(A), typeof(u0), typeof(tspan), typeof(kwargs)}(A, u0, tspan, kwargs)
    # Convert Ints to Floats
    ODEProblemLinear(A, u0::Integer, tspan; kwargs...) = new{typeof(A), Float64, typeof(tspan), typeof(kwargs)}(
        A, convert(Float64, u0), tspan, kwargs)
    ODEProblemLinear(A, u0::Vector{A} where A <: Integer, tspan; kwargs...) =
        new{typeof(f), Vector{Float64}, typeof(tspan), typeof(kwargs)}(A, convert(Vector{Float64}, u0), tspan, kwargs)
end


function solve(ode::ODEProblemLinear)
    @assert haskey(ode.kwargs, :dt) "Ein Zeitschritt muss übergeben werden!"
    solve(ode, ode.kwargs[:dt])
end


# Optimized version if A is a matrix
function solve(ode::ODEProblemLinear, dt)
    A = A_carpenter_kennedy()
    B = B_carpenter_kennedy()
    c = c_carpenter_kennedy()

    u = ode.u0
    g = similar(u)
    R = ode.A
    t = convert(Float64, ode.tspan[1])

    while t < ode.tspan[2]
        dt = min(dt, ode.tspan[2] - t)

        # Overwrite g to prevent allocations
        mul!(g, R, u)
        for step in 1:5
            mul!(g, R, u, 1, A[step])
            @. u += B[step] * dt * g
        end

        t += dt
    end

    return u
end
