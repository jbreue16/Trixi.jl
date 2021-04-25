@testset "barycentric_weights" begin
    @testset "N=0" begin
        @test Numerik4.barycentric_weights([0]) == [1]
        @test Numerik4.barycentric_weights([24.0]) == [1]
    end

    @testset "N=1" begin
        @test Numerik4.barycentric_weights([2, 1]) == [1/(2 - 1), 1/(1 - 2)]
        @test Numerik4.barycentric_weights([4.78, 9.12]) == [1/(4.78 - 9.12), 1/(9.12 - 4.78)]
    end

    @testset "N=2" begin
        @test Numerik4.barycentric_weights([-1.2, -0.12, 4.1]) == 
            [1/((-1.2 + 0.12)*(-1.2 - 4.1)), 1/((-0.12 + 1.2)*(-0.12 - 4.1)), 1/((4.1 + 1.2)*(4.1 + 0.12))]
    end
end

@testset "interpolation" begin
    @testset "N=0" begin
        p = @test_nowarn Numerik4.InterpolationPolynomial([0])
        @test p(0, [0]) == 0
        @test p(0, [3]) == 3
        @test p(0, [-6.34]) == -6.34
        p = @test_nowarn Numerik4.InterpolationPolynomial([321.2])
        @test p(34.2, [6.34]) ≈ 6.34
        @test p(-3.24, [-2.3]) == -2.3
    end

    @testset "N=1" begin
        p = @test_nowarn Numerik4.InterpolationPolynomial([-1.2, 3.8])
        @test p(-1.2, [38.1, 3]) == 38.1
        @test p(3.8, [38.1, 3]) == 3
        @test p(1.3, [1, 2]) == 1.5
        @test p(1.3, [2.1, 2.2]) == 2.15
        @test p(-3.1, [3.2, -1.0]) ≈ 4.796
    end

    @testset "N=3" begin
        f(x) = -1.1 * x^3 + 0.28*x - 1.289
        p = @test_nowarn Numerik4.InterpolationPolynomial([-1.23, 23.1, -28.1, 0.218])
        @test p(0, [f(-1.23), f(23.1), f(-28.1), f(0.218)]) ≈ f(0)
        @test p(-0.2823, [f(-1.23), f(23.1), f(-28.1), f(0.218)]) ≈ f(-0.2823)
    end

    @testset "invalid length" begin
        p = @test_nowarn Numerik4.InterpolationPolynomial([-1.2, 3.8])
        @test_throws AssertionError("length of f needs to be equal to the number of nodes") p(0, [23])
        @test_throws AssertionError("length of f needs to be equal to the number of nodes") p(0, [23, 213, 2])
    end
end

@testset "derivation_matrix" begin
    @testset "N=0" begin
        p = @test_nowarn Numerik4.InterpolationPolynomial([3.1])
        @test Numerik4.derivation_matrix(p) * [34] == [0]
        @test Numerik4.derivation_matrix(p) * [-2.1] == [0]
    end

    @testset "N=1" begin
        p = @test_nowarn Numerik4.InterpolationPolynomial([-1.2, 3.1])
        @test isapprox(Numerik4.derivation_matrix(p) * [0.4, 0.4], [0, 0], atol=1e-16)

        f(x) = 4.1*x - 0.123
        @test Numerik4.derivation_matrix(p) * [f(-1.2), f(3.1)] ≈ [4.1, 4.1]
    end

    @testset "N=3" begin
        f(x) = 1.23*x^3 - 0.123 * x^2 + 1.239 * x - 0.83
        g(x) = 3*1.23*x^2 - 2*0.123 * x + 1.239

        p = @test_nowarn Numerik4.InterpolationPolynomial([1.23, -8.123, -1.2, 3.1])
        @test Numerik4.derivation_matrix(p) * 
            [f(1.23), f(-8.123), f(-1.2), f(3.1)] ≈ [g(1.23), g(-8.123), g(-1.2), g(3.1)]
    end
end