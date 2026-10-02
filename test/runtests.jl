using Test, Random, LinearAlgebra, MonitoredXXZDomainWall

@testset "inputs and domain wall" begin
    @test_throws ArgumentError Model(3)
    @test_throws ArgumentError Model(2; p=1.2)
    @test_throws ArgumentError Model(2; period=0)
    @test_throws ArgumentError Model(2; J=NaN)
    m = Model(4)
    @test_throws ArgumentError trajectory(m, -1)
    @test_throws ArgumentError trajectory(m, 1; maxdim=0)
    @test_throws ArgumentError ensemble(m, 1, 0)
    start = trajectory(m, 0)
    @test vec(start.magnetization[1,:]) ≈ [1,1,-1,-1]
    @test start.transfer == [0]
    @test norm(start.state) ≈ 1 atol=1e-12
end

@testset "zero measurement unitary dynamics" begin
    m = Model(4; J=0.5, delta=0.3, p=0, period=0.04)
    exact = dense_channel(m, 10)
    sampled = trajectory(m, 10; cutoff=0, maxdim=16, rng=MersenneTwister(1))
    @test isempty(sampled.measurements)
    @test maximum(abs.(sampled.magnetization-exact.magnetization)) < 3e-4
    @test abs(sampled.transfer[end]-exact.transfer[end]) < 3e-4
    @test norm(sampled.state) ≈ 1 atol=1e-12
    @test real(tr(exact.rho^2)) ≈ 1 atol=1e-12
    @test sampled.transfer[end] > 0
    @test maximum(abs.(sum(sampled.magnetization; dims=2))) < 1e-10
end

@testset "projective measurement limits" begin
    frozen = Model(4; J=0, delta=0.8, p=1, period=0.15)
    sampled = trajectory(frozen, 4; rng=MersenneTwister(2))
    @test sampled.magnetization ≈ repeat([1.0 1.0 -1.0 -1.0], 5, 1) atol=1e-12
    @test length(sampled.measurements) == 16
    @test maximum(abs.(sampled.transfer)) < 1e-12
    exact = dense_channel(frozen, 4)
    @test exact.magnetization ≈ sampled.magnetization atol=1e-12
    @test real(tr(exact.rho)) ≈ 1 atol=1e-12
    @test minimum(eigvals(Hermitian(exact.rho))) > -1e-12
end

@testset "trajectory averages versus channel" begin
    m = Model(4; J=0.7, delta=0.35, p=0.4, period=0.1)
    exact = dense_channel(m, 5)
    sampled = ensemble(m, 5, 700; maxdim=16, rng=MersenneTwister(23))
    @test all(abs.(sampled.mean_magnetization-exact.magnetization) .<
              4 .* sampled.se_magnetization .+ 0.025)
    @test all(abs.(sampled.mean_transfer-exact.transfer) .<
              4 .* sampled.se_transfer .+ 0.025)
    @test maximum(abs.(sum(sampled.mean_magnetization; dims=2))) < 1e-10
    @test real(tr(exact.rho)) ≈ 1 atol=1e-12
    @test minimum(eigvals(Hermitian(exact.rho))) > -1e-12
    @test_throws ArgumentError dense_channel(Model(6), 1)
end

@testset "step and bond refinement" begin
    coarse = Model(4; J=0.7, delta=0.35, p=0, period=0.2)
    fine = Model(4; J=0.7, delta=0.35, p=0, period=0.025)
    reference = dense_channel(fine, 16).magnetization[end,:]
    coarse_result = trajectory(coarse, 2; maxdim=16, cutoff=0)
    fine_result = trajectory(fine, 16; maxdim=16, cutoff=0)
    narrow_result = trajectory(fine, 16; maxdim=1, cutoff=0)
    err = maximum(abs.(fine_result.magnetization[end,:]-reference))
    @test err < maximum(abs.(coarse_result.magnetization[end,:]-reference))/5
    @test err < maximum(abs.(narrow_result.magnetization[end,:]-reference))/5
    @test fine_result.max_bond > 1
end
