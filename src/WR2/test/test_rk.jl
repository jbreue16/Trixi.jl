@testset "rk" begin
    @testset "du/dt = 1" begin
        f(u, p, t) = [1]
        ode = Numerik4.ODEProblem(f, [0], (0, 1))
        @test isapprox(Numerik4.solve(ode, 0.0001), [1], rtol = 1e-12)
    end

    @testset "du/dt = 2 parameter" begin
        f(u, p, t) = [p]
        # Pass parameter 2
        ode = Numerik4.ODEProblem(f, [0], (0, 1), 2)
        @test isapprox(Numerik4.solve(ode, 0.0001), [2], rtol = 1e-12)
    end

    @testset "du/dt = u" begin
        f(u, p, t) = u
        ode = Numerik4.ODEProblem(f, [1], (0, 1))
        @test isapprox(Numerik4.solve(ode, 0.0001), [ℯ], rtol = 1e-13)
    end

    @testset "du/dt = u and timestep test" begin
        f(u, p, t) = u
        ode = Numerik4.ODEProblem(f, [1], (0, 1.05))
        @test isapprox(Numerik4.solve(ode, 0.5), [exp(1.05)], rtol = 0.001)
    end

    @testset "du/dt = u system" begin
        f(u, p, t) = u
        ode = Numerik4.ODEProblem(f, [1, 1, 1], (0, 1))
        @test isapprox(Numerik4.solve(ode, 0.0001), [ℯ, ℯ, ℯ], rtol = 1e-13)
    end

    @testset "du/dt = u matrix" begin
        A = I(3)
        function f!(du, u, t, p)
            du = mul!(du, A, u)
            return nothing
        end
        ode = Numerik4.ODEProblem(f!, [1, 1, 1], (0, 1))
        @test isapprox(Numerik4.solve(ode, 0.0001), [ℯ, ℯ, ℯ], rtol = 1e-13)
    end

    @testset "du/dt = t - u/t" begin
        f(u, p, t) = t .- u./t
        ode = Numerik4.ODEProblem(f, [4/3], (1, 2))
        @test isapprox(Numerik4.solve(ode, 0.0001), [11/6], rtol = 1e-13)
    end
end
