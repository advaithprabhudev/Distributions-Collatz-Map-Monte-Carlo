from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from analysis import _lognormal_pdf, empirical_cdf
from scipy import stats


def plot_histogram(sigma, s, mu, path):
    fig, ax = plt.subplots()
    ax.hist(sigma, bins=50, density=True, label="Empirical")
    x = np.linspace(sigma.min(), sigma.max(), 400).astype(float)
    ax.plot(x, _lognormal_pdf(x, s, mu), label="Lognormal")
    ax.set_xlabel("σ(n) Stopping Time")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cdf(sigma, s, mu, path, terras_curve):
    fig, ax = plt.subplots()
    x, p = empirical_cdf(sigma)
    ax.plot(x, p, label="Empirical CDF")
    x = np.linspace(x[0], x[-1], 600).astype(float)
    fit = stats.lognorm(s=s, scale=np.exp(mu))

    ax.plot(x, fit.cdf(x), label="Log-Normal CDF")
    ax.set_xscale("log")
    ax.set_xlabel("σ(n) Log Scale")
    ax.set_ylabel("P(σ ≤ x)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mean_by_residue(results, m, path):
    fig, ax = plt.subplots()
    rs = list(range(m))
    means = [float(np.mean(results[r])) for r in rs]
    sigma = np.concatenate(list(results.values()))
    unconditional_mean = float(np.mean(sigma))

    ax.bar(rs, means)
    ax.axhline(unconditional_mean, label="Unconditional Mean")
    ax.set_xticks(rs)
    ax.set_xlabel("Residue Class")
    ax.set_ylabel("Mean σ(n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
