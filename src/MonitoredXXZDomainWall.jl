module MonitoredXXZDomainWall

using ITensors, ITensorMPS, LinearAlgebra, Random, Statistics

export Model, trajectory, ensemble, dense_channel, magnetizations, front_transfer

"""Open XXZ chain with periodic local Z measurements.

H = J sum(XᵢXᵢ₊₁ + YᵢYᵢ₊₁) + Δ sum(ZᵢZᵢ₊₁).
Each site is measured independently with probability `p` after every interval `period`.
"""
struct Model
    sites::Vector{Index}
    J::Float64
    delta::Float64
    p::Float64
    period::Float64
end

function finite_real(name, x)
    x isa Real && isfinite(x) || throw(ArgumentError("$name must be finite and real"))
    Float64(x)
end

function Model(n::Integer; J=0.5, delta=0.3, p=0.2, period=0.1)
    n >= 2 && iseven(n) || throw(ArgumentError("domain-wall chain length must be even and at least 2"))
    coupling = finite_real("J", J)
    interaction = finite_real("delta", delta)
    chance = finite_real("p", p)
    0 <= chance <= 1 || throw(ArgumentError("p must lie in [0,1]"))
    τ = finite_real("period", period)
    τ > 0 || throw(ArgumentError("period must be positive"))
    Model(siteinds("S=1/2", n), coupling, interaction, chance, τ)
end

# The left half starts with Z=+1, the right half with Z=-1.
domain_wall(model::Model) = MPS(model.sites, [i <= length(model.sites)÷2 ? "Up" : "Dn" for i in eachindex(model.sites)])
# ITensors' Sz has eigenvalues ±1/2; multiply by two for Pauli-Z magnetization.
magnetizations(state::MPS) = 2 .* real.(expect(state, "Sz"))
# (1+⟨Zᵢ⟩)/2 is the up-spin probability; the initial right-half value is zero.
front_transfer(z::AbstractVector) = sum((z[i]+1)/2 for i in length(z)÷2+1:length(z))

function bond_gate(model::Model, i, time)
    a, b = model.sites[i:i+1]
    # X X + Y Y = 2(S+ S- + S- S+); Z Z = 4 Sz Sz.
    h = 2model.J * (op("S+", a)*op("S-", b) + op("S-", a)*op("S+", b)) +
        4model.delta * op("Sz", a)*op("Sz", b)
    exp(-im*time*h)
end

function unitary_step(state, odd_half, even_full; cutoff, maxdim)
    # Odd/2, even, odd/2 is a symmetric split of one physical period τ.
    state = apply(odd_half, state; cutoff, maxdim)
    state = apply(even_full, state; cutoff, maxdim)
    apply(odd_half, state; cutoff, maxdim)
end

function controls(periods, cutoff, maxdim)
    periods isa Integer && periods >= 0 || throw(ArgumentError("periods must be a nonnegative integer"))
    c = finite_real("cutoff", cutoff)
    c >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    maxdim isa Integer && maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    c
end

function prepared_gates(model)
    n = length(model.sites)
    odd = [bond_gate(model, i, model.period/2) for i in 1:2:n-1]
    even = [bond_gate(model, i, model.period) for i in 2:2:n-1]
    # P± = (I ± Z)/2 project onto a recorded local measurement outcome.
    project_up = [0.5op("Id", s)+op("Sz", s) for s in model.sites]
    project_down = [0.5op("Id", s)-op("Sz", s) for s in model.sites]
    odd, even, project_up, project_down
end

function sample(model, periods, cutoff, maxdim, rng, gates)
    state = domain_wall(model)
    n = length(model.sites)
    history = Matrix{Float64}(undef, periods+1, n)
    history[1, :] = magnetizations(state)
    records = Tuple{Int,Int,Symbol}[]
    odd, even, project_up, project_down = gates
    largest_bond = maxlinkdim(state)
    for k in 1:periods
        state = unitary_step(state, odd, even; cutoff, maxdim)
        state /= norm(state)
        for i in 1:n
            rand(rng) < model.p || continue
            # Born probability q(up) = ⟨P+ᵢ⟩ in the current trajectory state.
            chance_up = clamp((magnetizations(state)[i]+1)/2, 0.0, 1.0)
            outcome = rand(rng) < chance_up ? :up : :down
            state = apply(outcome == :up ? project_up[i] : project_down[i],
                          state; cutoff, maxdim)
            norm(state) > 1e-12 || throw(ErrorException("sampled zero-probability outcome"))
            state /= norm(state)
            push!(records, (k, i, outcome))
        end
        history[k+1, :] = magnetizations(state)
        largest_bond = max(largest_bond, maxlinkdim(state))
    end
    (state=state, magnetization=history, transfer=[front_transfer(view(history,k,:)) for k in 1:periods+1],
     measurements=records, max_bond=largest_bond)
end

"""One MPS trajectory, including magnetization and right-half transferred spin after each period."""
function trajectory(model::Model, periods::Integer; cutoff=1e-12, maxdim=64,
                    rng=Random.default_rng())
    c = controls(periods, cutoff, maxdim)
    sample(model, periods, c, maxdim, rng, prepared_gates(model))
end

"""Average Pauli-Z magnetization and right-half up-spin transfer.

Standard errors describe trajectory sampling only; gate and bond errors remain.
"""
function ensemble(model::Model, periods::Integer, samples::Integer;
                  cutoff=1e-12, maxdim=64, rng=Random.default_rng())
    c = controls(periods, cutoff, maxdim)
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    n = length(model.sites)
    values = Array{Float64}(undef, samples, periods+1, n)
    transfer = Matrix{Float64}(undef, samples, periods+1)
    gate_set = prepared_gates(model)
    largest_bond = 1
    for k in 1:samples
        result = sample(model, periods, c, maxdim, rng, gate_set)
        values[k, :, :] = result.magnetization
        transfer[k, :] = result.transfer
        largest_bond = max(largest_bond, result.max_bond)
    end
    (mean_magnetization=dropdims(mean(values; dims=1); dims=1),
     se_magnetization=samples == 1 ? zeros(periods+1,n) :
         dropdims(std(values; dims=1, corrected=true); dims=1)/sqrt(samples),
     mean_transfer=vec(mean(transfer; dims=1)),
     se_transfer=samples == 1 ? zeros(periods+1) :
         vec(std(transfer; dims=1, corrected=true))/sqrt(samples),
     samples=samples, max_bond=largest_bond)
end

"""Exact small-chain unitary followed by the unread Z-measurement channel.

This evolves the density operator averaged over measurement selections and outcomes.
"""
function dense_channel(model::Model, periods::Integer)
    controls(periods, 0.0, 1)
    n = length(model.sites)
    n <= 4 || throw(ArgumentError("dense channel supports at most four spins"))
    dim = 1 << n
    h = zeros(ComplexF64, dim, dim)
    z = [zeros(ComplexF64, dim, dim) for _ in 1:n]
    for basis in 0:dim-1
        for i in 1:n
            z[i][basis+1,basis+1] = (basis & (1 << (i-1))) != 0 ? 1 : -1
        end
        for i in 1:n-1
            bit1, bit2 = 1 << (i-1), 1 << i
            h[basis+1,basis+1] += model.delta*z[i][basis+1,basis+1]*z[i+1][basis+1,basis+1]
            if ((basis & bit1) != 0) != ((basis & bit2) != 0)
                h[(basis ⊻ bit1 ⊻ bit2)+1,basis+1] += 2model.J
            end
        end
    end
    u = exp(-im*model.period*h)
    basis = sum(1 << (i-1) for i in 1:n÷2)
    rho = zeros(ComplexF64, dim, dim)
    rho[basis+1,basis+1] = 1
    history = Matrix{Float64}(undef, periods+1, n)
    history[1, :] = [real(tr(z[i]*rho)) for i in 1:n]
    for k in 1:periods
        rho = u*rho*u'
        for i in 1:n
            # Averaging over unrecorded outcomes leaves populations fixed;
            # coherences between opposite local Z sectors acquire 1-p.
            for a in 0:dim-1, b in 0:dim-1
                ((a ⊻ b) & (1 << (i-1))) != 0 && (rho[a+1,b+1] *= 1-model.p)
            end
        end
        history[k+1, :] = [real(tr(z[i]*rho)) for i in 1:n]
    end
    (rho=rho, magnetization=history,
     transfer=[front_transfer(view(history,k,:)) for k in 1:periods+1])
end

end
