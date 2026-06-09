import numpy as np


MIN_CHAINS = 2
MIN_CHAIN_LENGTH = 10
MAX_ACF_LAG = 100


def autocorrelation(chain: np.ndarray, max_lag: int) -> np.ndarray:
    if chain.ndim != 1:
        raise ValueError(f"chain must be 1D, got {chain.ndim}D")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    n = len(chain)
    if n <= max_lag:
        raise ValueError(f"chain length {n} must exceed max_lag {max_lag}")
    centered = chain - chain.mean()
    variance = np.var(chain)
    if variance == 0.0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    acf = np.array(
        [np.mean(centered[: n - lag] * centered[lag:]) / variance for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
    return acf


def effective_sample_size(chain: np.ndarray) -> float:
    if chain.ndim != 1:
        raise ValueError(f"chain must be 1D, got {chain.ndim}D")
    n = len(chain)
    if n < MIN_CHAIN_LENGTH:
        raise ValueError(f"chain length {n} must be >= {MIN_CHAIN_LENGTH}")
    max_lag = min(n // 2 - 1, MAX_ACF_LAG)
    acf = autocorrelation(chain, max_lag)
    rho_sum = 1.0 + 2.0 * float(np.sum(acf[1:]))
    rho_sum = max(rho_sum, 1.0)
    return float(n / rho_sum)


def gelman_rubin(chains: np.ndarray) -> float:
    if chains.ndim != 2:
        raise ValueError(
            f"chains must be 2D array of shape (num_chains, length), got {chains.ndim}D"
        )
    if chains.shape[0] < MIN_CHAINS:
        raise ValueError(f"need >= {MIN_CHAINS} chains, got {chains.shape[0]}")
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    B = n * np.var(chain_means, ddof=1)
    W = float(np.mean(np.var(chains, axis=1, ddof=1)))
    if W <= 0.0:
        return float("nan")
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))
