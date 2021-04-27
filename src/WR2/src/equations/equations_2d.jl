struct EquationLinearAdvection2D{A}
    a::A

    EquationLinearAdvection2D(a) = new{typeof(a)}(a)
end


calcflux(u, orientation, equation::EquationLinearAdvection2D) = equation.a * u
max_abs_eigenvalue(u, equation::EquationLinearAdvection2D) = abs(equation.a)


struct EquationLinearEuler2D{R,L}
    ρ::R
    λ::L

    EquationLinearEuler2D(ρ, λ) = new{typeof(ρ),typeof(λ)}(ρ, λ)
end


function calcflux(u, orientation, equation::EquationLinearEuler2D)
    if orientation == 1
        return @SVector [1 / equation.ρ * u[3], 0, equation.λ * u[1]]
    else
        return @SVector [0, 1 / equation.ρ * u[3], equation.λ * u[2]]
    end
end

max_abs_eigenvalue(u, equation::EquationLinearEuler2D) = sqrt(equation.λ / equation.ρ)

struct EquationComprEuler2D{L}
    γ::L

    EquationComprEuler2D{L} = new{typeof(γ)}(γ)
end


function calcflux(u, orientation, equation::EquationComprEuler2D)
    p = (equation.γ - 1)(u[4] + u[2]^2 / (2u[1]) + u[3]^2 / (2u[1]))
    if orientation == 1
        return @SVector [u[2], u[2]^2 / u[1] + p, u[2]u[3] / u[1] , u[2] / u[1](u[4] + p)]
    else
        return @SVector [u[3], u[2]u[3] / u[1], u[3]^2 / u[1] + p , u[3] / u[1](u[4] + p)]
    end
end

function max_abs_eigenvalue(u, equation::EquationComprEuler2D) 
    v1=u[2]/u[1]
    v2=u[3]/u[1]
    p = (equation.γ - 1)(u[4] + u[2]^2 / (2u[1]) + u[3]^2 / (2u[1]))
    c = sqrt(equation.γ p / u[1])

    # Da c>0 muss nur v1/2 + c betrachtet werden 
    return max(v1+c,v2+c)
end