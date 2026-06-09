import numpy as np


def collatz_step(n: int) -> int:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_sequence(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    seq = [n]
    while n != 1:
        n = collatz_step(n)
        seq.append(n)
    return np.array(seq, dtype=np.int64)


def stopping_time(n: int) -> int:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    count = 0
    while n != 1:
        n = collatz_step(n)
        count += 1
    return count


def stopping_time_histogram(upper: int) -> np.ndarray:
    if upper < 1:
        raise ValueError(f"upper must be >= 1, got {upper}")
    times = np.fromiter(
        (stopping_time(n) for n in range(1, upper + 1)),
        dtype=np.int64,
        count=upper,
    )
    return np.bincount(times)
