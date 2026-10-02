using MonitoredXXZDomainWall, Random, Printf

model = Model(4; J=0.7, delta=0.35, p=0.4, period=0.1)
# The dense channel averages over unread measurement outcomes exactly.
exact = dense_channel(model, 5)
# The MPS estimate averages 700 individually recorded outcomes.
sampled = ensemble(model, 5, 700; maxdim=16, rng=MersenneTwister(23))
println("period  exact transfer  MPS mean ± SE")
# Transfer is the expected right-half up-spin count, initially zero.
for k in 0:5
    @printf("%6d  %14.8f  %8.5f ± %.5f\n", k, exact.transfer[k+1],
            sampled.mean_transfer[k+1], sampled.se_transfer[k+1])
end
@printf("maximum magnetization error: %.6f\n",
        maximum(abs.(sampled.mean_magnetization-exact.magnetization)))
@printf("total spin at final time: %.3e (channel), %.3e (MPS mean)\n",
        sum(exact.magnetization[end,:]), sum(sampled.mean_magnetization[end,:]))
