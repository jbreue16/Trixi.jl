@testset "legendre_gauss" begin
    @testset "N=0" begin
        @test Numerik4.legendre_gauss_nodes_weights(0) == ([0], [2])
    end

    @testset "N=1" begin
        @test Numerik4.legendre_gauss_nodes_weights(1) == ([-sqrt(1/3), sqrt(1/3)], [1, 1])
    end

    @testset "N=2" begin
        nodes, weights = @test_nowarn Numerik4.legendre_gauss_nodes_weights(2)
        @test nodes ≈ [-sqrt(3/5), 0, sqrt(3/5)]
        @test weights ≈ [5/9, 8/9, 5/9]
    end
end

@testset "legendre_gauss_lobatto" begin

    @testset "N=1" begin
        @test Numerik4.legendre_gauss_lobatto_nodes_weights(1) == ([-1, 1], [1, 1])
    end

    @testset "N=2" begin
        @test Numerik4.legendre_gauss_lobatto_nodes_weights(2) == ([-1, 0, 1], [1/3, 4/3, 1/3])
    end

    @testset "N=3" begin
        nodes, weights = @test_nowarn Numerik4.legendre_gauss_lobatto_nodes_weights(3)
        @test isapprox(nodes, [-1, -sqrt(1/5), sqrt(1/5), 1]; rtol = 0.000001)
        @test weights ≈ [1/6, 5/6, 5/6, 1/6]
    end

end

