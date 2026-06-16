import numpy as np


def collatz_step(x):
    if x % 2 == 0:
        result = x // 2
    else:
        result = (3 * x + 1) // 2
    return result


def stopping_time(n):
    counter = 0
    while n > 1:
        n = collatz_step(n)
        counter += 1
    return counter


def stopping_times_batch(arr):
    if arr.ndim != 1:
        return ValueError("Error")
    return np.vectorize(stopping_time)(arr)


def _collatz_sequence(n):
    arr = [n]
    while n > 1:
        n = collatz_step(n)
        arr.append(n)
    return np.array(arr)
