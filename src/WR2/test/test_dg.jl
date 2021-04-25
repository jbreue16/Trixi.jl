@testset "dg" begin
    @testset "Well-Balanced with Burgers" begin
        # setting
        equation = Numerik4.EquationBurgers()
        # source = nothing is default
        # method DG is default
        # initializations
        u0(x) = [3, 0]
        u_solution(x) = [3, 0]
        N = 5
        Nq = 40
        xspan = (0,1)
        riemann_solver = Numerik4.rusanov
        dg = Numerik4.DG(N, Nq, length(u0(0.0)), equation, riemann_solver; xspan = xspan)

        #computing
        ode = Numerik4.semidiscretize(dg, u0; tspan = (0,1), CFL = 1.0)
        sol = Numerik4.solve(ode)
        # Compute exact solution vector
        exact = similar(sol)
        for element in 1:Nq
            for node in 1:N+1
                node_vars = u_solution(dg.node_pos[node, element])
                for s in 1:Numerik4.nvariables(dg)
                    exact[s, node, element] = node_vars[s]
                end
            end
        end
        #test
        @test sol == exact

    end


    @testset "Well-Balanced with simplified ShallowWater (no ground Topography)" begin
        # setting
        equation = Numerik4.EquationShallowWater(9.812, x -> 0) #(g, b)
        # source = nothing is default
        # test every method
        riemann_solver = Numerik4.rusanov
        for i in 1:3
            if i == 1
                method =:DG
                riemann_solver = Numerik4.rusanov
            elseif i == 2
                method =:ecDG
                riemann_solver = Numerik4.entropy_flux
            else
                method =:esDG
                riemann_solver = Numerik4.entropy_shock_flux
            end

            # initializations
            u0(x) = [3, 0]
            u_solution(x) = [3, 0]
            N = 5
            Nq = 80
            xspan = (0, 1)

            dg = Numerik4.DG(N, Nq, length(u0(0.0)), equation, riemann_solver; xspan=xspan, method = method)

            #computing
            ode = Numerik4.semidiscretize(dg, u0; tspan = (0,1), CFL = 1.0)
            sol = Numerik4.solve(ode)

            # Compute exact solution vector
            exact = similar(sol)
            for element in 1:Nq
                for node in 1:N+1
                    node_vars = u_solution(dg.node_pos[node, element])
                    for s in 1:Numerik4.nvariables(dg)
                        exact[s, node, element] = node_vars[s]
                    end
                end
            end
            # @info "method $i" maximum(abs.(sol-exact))
            @test isapprox(sol, exact, rtol = 1e-10)
        end
    end

    @testset "Well-Balanced with ShallowWater for every DG method" begin
        # setting
        b(x) = sin(pi / 10 * x) + 1
        equation = Numerik4.EquationShallowWater(9.812, b)
        # test all methods
        riemann_solver = Numerik4.rusanov
        for i in 1:3
            if i == 1
                method =:DG
                riemann_solver = Numerik4.rusanov
            elseif i == 2
                method =:ecDG
                riemann_solver = Numerik4.entropy_flux
            else
                method =:esDG
                riemann_solver = Numerik4.entropy_shock_flux
            end

            # initializations
            u0(x) = [3 - b(x), 0]
            u_solution(x) = u0(x) #static solution
            N = 5
            Nq = 80
            xspan = (0, 20)

            dg = Numerik4.DG(N, Nq, length(u0(0.0)), equation, riemann_solver;
                            xspan=xspan, method = method)

            #computing
            ode = Numerik4.semidiscretize(dg, u0; CFL = 1.0)
            sol = Numerik4.solve(ode)

            # Compute exact solution and topography vector
            # start = similar(sol)
            ground = similar(sol)
            exact = similar(sol)

            for element in 1:Nq
                for node in 1:N+1
                    node_vars = u_solution(dg.node_pos[node, element])
                    gr = b(dg.node_pos[node, element])
                    for s in 1:Numerik4.nvariables(dg)
                        exact[s, node, element] = node_vars[s]
                        if s == 1
                            ground[s, node, element] = gr
                        else
                            ground[s, node, element] = 0
                        end
                    end
                end
            end
            # @info "method $i" maximum(abs.(sol-exact))
            @test isapprox(sol, exact, rtol = 1e-10)
        end
    end


end
