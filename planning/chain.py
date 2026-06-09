import functools
from math import gcd

import numpy as np


MAX_POWER_ITER = 10_000
CONVERGENCE_TOL = 1e-12


def _accelerated_k(n: int, k: int) -> int:
    for _ in range(k):
        n = n // 2 if n % 2 == 0 else (3 * n + 1) // 2
    return n


def exact_transition_matrix(k: int) -> np.ndarray:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    M = 2 ** k
    P = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        j = _accelerated_k(i, k) % M
        P[i, j] = 1.0
    assert np.allclose(P.sum(axis=1), 1.0), "Row sums must equal 1"
    return P


def empirical_transition_matrix(num_samples: int, k: int) -> np.ndarray:
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    M = 2 ** k
    counts = np.zeros((M, M), dtype=np.float64)
    for n in range(1, num_samples + 1):
        i = n % M
        j = _accelerated_k(n, k) % M
        counts[i, j] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return counts / row_sums


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    assert P.ndim == 2 and P.shape[1] == n, f"P must be square, got {P.shape}"
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "P must be row-stochastic"
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(MAX_POWER_ITER):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < CONVERGENCE_TOL:
            return pi_new / pi_new.sum()
        pi = pi_new
    return pi / pi.sum()


def is_irreducible(P: np.ndarray) -> bool:
    n = P.shape[0]
    assert P.ndim == 2 and P.shape[1] == n, f"P must be square, got {P.shape}"
    adj = P > 0
    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        stack = [start]
        while stack:
            node = stack.pop()
            for nbr in np.where(adj[node])[0]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(int(nbr))
        if not np.all(visited):
            return False
    return True


def is_aperiodic(P: np.ndarray) -> bool:
    n = P.shape[0]
    assert P.ndim == 2 and P.shape[1] == n, f"P must be square, got {P.shape}"
    if np.any(np.diag(P) > 0):
        return True
    Pk = np.eye(n, dtype=np.float64)
    return_times: list[int] = []
    for power in range(1, 2 * n + 1):
        Pk = Pk @ P
        if Pk[0, 0] > CONVERGENCE_TOL:
            return_times.append(power)
    if not return_times:
        return False
    return functools.reduce(gcd, return_times) == 1
