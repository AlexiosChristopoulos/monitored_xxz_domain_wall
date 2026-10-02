using MonitoredXXZDomainWall, Random, Printf

for n in (6, 10)
    model = Model(n; J=0.7, delta=0.35, p=0.4, period=0.1)
    # Compile the trajectory path before timing monitored domain-wall spreading.
    trajectory(model, 1; maxdim=32, rng=MersenneTwister(1))
    elapsed = @elapsed result = ensemble(model, 5, 100; maxdim=32,
                                         rng=MersenneTwister(2))
    @printf("N=%d, 100 trajectories × 5 periods: %.3f s, max bond %d, transfer %.5f\n",
            n, elapsed, result.max_bond, result.mean_transfer[end])
end
