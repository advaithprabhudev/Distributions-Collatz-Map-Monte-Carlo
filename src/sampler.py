from collatz import collatz_step, stopping_time, stopping_times_batch
import numpy as np
import random

np.random.seed(0)


def uniform_sample(N: int, size: int):
    if N >= 2 and size <= 1:
        return ValueError("Error")
    if size < N:
        return np.array(sorted(random.sample(range(1, N+1), size)))
    else:
        return np.array(sorted(random.randint(1, N) for _ in range(size)))


def sample_stopping_times(N, size):
    ns = uniform_sample(N, size)
    sigmas = stopping_times_batch(ns)
    return ns, sigmas


def residue_sample(N, size, m, r):
    if m < 1:
        return ValueError("Error")
    if r < 0 or r >= m:
        return ValueError("Error")
    pool = np.array([n for n in range(1, N+1) if n % m == r])
    ns = np.array(random.choices(pool.tolist(), k=size))
    sigmas = stopping_times_batch(ns)
    return ns, sigmas
