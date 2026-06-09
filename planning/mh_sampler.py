import numpy as np


def acceptance_ratio(x: int, y: int, pi: np.ndarray) -> float:
    assert 0 <= x < len(pi), f"x={x} out of range [0, {len(pi)})"
    assert 0 <= y < len(pi), f"y={y} out of range [0, {len(pi)})"
    if pi[x] <= 0.0:
        return 1.0
    return float(min(1.0, pi[y] / pi[x]))


def mh_step(x: int, pi: np.ndarray, k: int, rng: np.random.Generator) -> int:
    M = 2 ** k
    assert len(pi) == M, f"pi must have length {M}, got {len(pi)}"
    assert 0 <= x < M, f"x={x} out of range [0, {M})"
    y = int(rng.integers(0, M))
    alpha = acceptance_ratio(x, y, pi)
    return y if rng.random() < alpha else x


def run_chain(
    start: int,
    pi: np.ndarray,
    k: int,
    n_steps: int,
    burn_in: int,
    thinning: int,
    seed: int,
) -> np.ndarray:
    M = 2 ** k
    if not (0 <= start < M):
        raise ValueError(f"start={start} out of range [0, {M})")
    if len(pi) != M:
        raise ValueError(f"pi must have length {M}, got {len(pi)}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if thinning < 1:
        raise ValueError(f"thinning must be >= 1, got {thinning}")
    rng = np.random.default_rng(seed)
    x = start
    total_steps = burn_in + n_steps * thinning
    samples = np.empty(n_steps, dtype=np.int64)
    sample_idx = 0
    for step in range(total_steps):
        x = mh_step(x, pi, k, rng)
        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples[sample_idx] = x
            sample_idx += 1
    return samples
