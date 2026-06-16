from pathlib import Path

from analysis import fit_lognormal, ks_test, summary
from plots import plot_cdf, plot_histogram, plot_mean_by_residue
from sampler import residue_sample, sample_stopping_times

output = Path("outputs")
N = 1_000_000
SIZE = 50_000
M = 8


def run_experiment(N: int, size: int) -> None:
    output.mkdir(exist_ok=True)

    ns, sigma = sample_stopping_times(N, size)

    s, mu = fit_lognormal(sigma)
    result = ks_test(sigma, s, mu)
    stats = summary(sigma)

    print(f"N={N:,}  size={size:,}")
    print(f"log-normal fit:  s={s:.4f}  mu={mu}")
    print(f"KS statistic:    {result.statistic}")
    print(f"KS p-value:      {result.pvalue}")
    for key, val in stats.items():
        print(f"{key:<12} {val}")

    plot_histogram(sigma, s, mu, output / "histogram.png")
    plot_cdf(sigma, s, mu, output / "cdf.png", terras_curve=None)

    residue_results = {}
    for r in range(M):
        _, sigma_r = residue_sample(N, size // M, M, r)
        residue_results[r] = sigma_r

    plot_mean_by_residue(residue_results, M, output / "residue_means.png")

    print(f"Figures saved to {output}/")


if __name__ == "__main__":
    run_experiment(N, SIZE)
