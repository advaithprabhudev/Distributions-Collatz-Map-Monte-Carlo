import numpy as np


def accelerated_step(n: int) -> int:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return n // 2 if n % 2 == 0 else (3 * n + 1) // 2


def parity_vector(n: int, k: int) -> np.ndarray:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    v = np.empty(k, dtype=np.int64)
    current = n
    for i in range(k):
        v[i] = current % 2
        current = accelerated_step(current)
    return v


def closed_form_constant(v: np.ndarray) -> int:
    k = len(v)
    assert k >= 1, f"parity vector must have length >= 1, got {k}"
    c = 0
    for i in range(k):
        tail_sum = int(np.sum(v[i + 1:]))
        c += int(v[i]) * (3 ** tail_sum) * (2 ** i)
    return c


def accelerated_map_k(n: int, k: int) -> int:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    v = parity_vector(n, k)
    s = int(np.sum(v))
    c = closed_form_constant(v)
    TWO_K = 2 ** k
    numerator = (3 ** s) * n + c
    assert numerator % TWO_K == 0, (
        f"Division not exact: n={n}, k={k}, v={v}, numerator={numerator}"
    )
    return numerator // TWO_K


def residue_partition(upper: int, k: int) -> dict[int, np.ndarray]:
    if upper < 1:
        raise ValueError(f"upper must be >= 1, got {upper}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    TWO_K = 2 ** k
    ns = np.arange(1, upper + 1, dtype=np.int64)
    remainders = ns % TWO_K
    return {r: ns[remainders == r] for r in range(TWO_K)}
