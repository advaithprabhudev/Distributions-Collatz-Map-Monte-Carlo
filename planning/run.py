import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import chain
import convergence
import iterator
import mh_sampler
import parity


K = 3
N_HISTOGRAM = 10_000
N_EMPIRICAL_SAMPLES = 10_000
N_CHAIN_STEPS = 2_000
BURN_IN = 500
THINNING = 5
CHAIN_SEED_A = 42
CHAIN_SEED_B = 99


def main() -> None:
    hist = iterator.stopping_time_histogram(N_HISTOGRAM)
    print(f"Phase 1 — stopping times (n <= {N_HISTOGRAM})")
    print(f"  max stopping time : {len(hist) - 1}")
    print(f"  mode              : {int(np.argmax(hist))}")
    print(f"  mean              : {float(np.average(np.arange(len(hist)), weights=hist)):.2f}")

    part = parity.residue_partition(N_HISTOGRAM, K)
    print(f"\nPhase 1 — residue partition (k={K}, mod {2**K})")
    for r, ns in part.items():
        print(f"  residue {r}: {len(ns)} integers")

    P_exact = chain.exact_transition_matrix(K)
    P_empirical = chain.empirical_transition_matrix(N_EMPIRICAL_SAMPLES, K)
    print(f"\nPhase 2 — Markov chain (k={K}, state space size={2**K})")
    print(f"  exact P irreducible  : {chain.is_irreducible(P_exact)}")
    print(f"  exact P aperiodic    : {chain.is_aperiodic(P_exact)}")
    print(f"  empirical P max|diff|: {float(np.max(np.abs(P_exact - P_empirical))):.6f}")

    pi = chain.stationary_distribution(P_exact)
    print(f"  stationary pi        : {np.round(pi, 4)}")

    chain_a = mh_sampler.run_chain(1, pi, K, N_CHAIN_STEPS, BURN_IN, THINNING, CHAIN_SEED_A)
    chain_b = mh_sampler.run_chain(3, pi, K, N_CHAIN_STEPS, BURN_IN, THINNING, CHAIN_SEED_B)
    print(f"\nPhase 3 — MH sampler ({N_CHAIN_STEPS} steps, burn_in={BURN_IN}, thinning={THINNING})")
    print(f"  chain A unique states: {len(np.unique(chain_a))}")
    print(f"  chain B unique states: {len(np.unique(chain_b))}")

    chains_float = np.stack([chain_a.astype(np.float64), chain_b.astype(np.float64)])
    rhat = convergence.gelman_rubin(chains_float)
    ess_a = convergence.effective_sample_size(chain_a.astype(np.float64))
    ess_b = convergence.effective_sample_size(chain_b.astype(np.float64))
    print(f"\nPhase 4 — convergence diagnostics")
    print(f"  Gelman-Rubin R-hat   : {rhat:.4f}")
    print(f"  ESS chain A          : {ess_a:.1f}")
    print(f"  ESS chain B          : {ess_b:.1f}")


if __name__ == "__main__":
    main()
