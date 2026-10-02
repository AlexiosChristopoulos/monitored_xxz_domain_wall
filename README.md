# MonitoredXXZDomainWall.jl

A small Julia MPS package for finite XXZ chains that start with all left-half spins up and right-half spins down. At each period, each spin is independently selected for a projective `Z` measurement. The package tracks local magnetization and the expected number of up spins transferred into the initially down half.

## Model

With Pauli matrices, open ends, and ℏ = 1,

```text
H = J Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁) + Δ Σᵢ ZᵢZᵢ₊₁.
```

Every interval `period`, site `i` is measured in the `Z` basis with independent probability `p`. Outcome probabilities follow the current trajectory state; the state is projected and normalized. The Hamiltonian preserves total `Z`, including across measurement trajectories. A symmetric odd/even/odd bond-gate split approximates each unitary interval to second order. The dense reference instead exponentiates the full Hamiltonian and applies the exact unread-measurement channel after each period.

### Numerical quantities and physical meaning

| In the code | Physical meaning |
|---|---|
| `J`, `delta` (`Δ`) | Exchange strength and nearest-neighbor Ising interaction in the Pauli convention above. |
| `period` (`τ`), `p` | Time between monitoring rounds and per-site selection probability in each round. |
| `magnetizations(state)[i]` | `⟨Zᵢ⟩ = 2⟨Sᶻᵢ⟩`, ranging from −1 to +1. |
| `front_transfer(z)` | $Q_R=\sum_{i=N/2+1}^{N}(1+\langle Z_i\rangle)/2$, the expected number of up spins in the initially down half. |
| `measurements` | Recorded `(period, site, outcome)` projective measurements for one trajectory. |
| `mean_transfer`, `se_transfer` | Trajectory estimate of `Q_R` and its sampling standard error. |
| `dense_channel(...).rho` | Ensemble density matrix after exact unitary intervals and measurements whose outcomes are unread. |

`Q_R` starts at zero. Total `Z` is conserved, so an up spin gained on the
right corresponds to one leaving the left. `Q_R` is an integrated transfer
observable at a finite time, not an instantaneous current. For one selected
site, the Born probability of `up` is `(1+⟨Zᵢ⟩)/2`; averaging over unread
outcomes suppresses coherence between opposite local `Z` sectors by `1-p`.

This compact model is motivated by [Gunawardana, Moghaddam, and Ojanen (2026)](https://arxiv.org/abs/2605.27350). It is a finite-chain learning and comparison tool; it does not claim to reproduce that work's phase-transition results.

## Install and run

Julia 1.11+ and ITensors.jl / ITensorMPS.jl are required. In this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. example.jl
julia --project=. benchmark.jl
```

```julia
using MonitoredXXZDomainWall, Random
model = Model(4; J=0.7, delta=0.35, p=0.4, period=0.1)
one = trajectory(model, 5; rng=MersenneTwister(1))
many = ensemble(model, 5, 700; rng=MersenneTwister(23))
exact = dense_channel(model, 5)
println(many.mean_transfer[end], " ± ", many.se_transfer[end])
println(exact.transfer[end])
```

`trajectory` returns an MPS, `(periods+1) × N` magnetization history, right-half transfer history, `(period, site, outcome)` records, and largest observed bond. `ensemble` returns trajectory means and standard errors. `dense_channel` returns the exact small-chain density matrix and corresponding observable histories for `N ≤ 4`. A seeded RNG makes MPS samples repeatable.

## Checks and limits

Tests compare the zero-measurement MPS unitary with exact dense evolution, check static projective limits, total-spin conservation, trace and positivity, MPS trajectory averages against the dense channel, and time-step/bond refinement. See the [executed notebook](xxz_monitoring_tutorial.ipynb), [model notes](problem.tex), and [progress log](progress.tex).

Use smaller `period` to reduce Trotter error; this also changes the physical monitoring rate if `p` stays fixed. To refine while holding approximately fixed rate, set `p = 1-exp(-rate*period)`. Increase `maxdim` and decrease `cutoff` to assess MPS truncation. Monte Carlo standard errors do not include these systematic errors. The exact dense reference is capped at four sites. Finite-chain transfer curves cannot establish a ballistic or diffusive phase.

The local `noisy_transverse_ising` package supplied the finite-step MPS trajectory pattern, adapted here to projective monitoring. No connected GitHub code was reused. Existing dependencies retain their upstream licenses.
I also reviewed the owner's existing `Open-Systems-and-Current-Fluctuations`
and `Local_Temperature` repositories before publication. Their vectorized
density-operator code would complicate this pure-state trajectory package, so
no code was copied.
