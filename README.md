# Collatz Stopping-Time Distributions via Monte Carlo Sampling

An empirical study of stopping-time distributions in the Collatz map using uniform Monte Carlo sampling, distribution fitting, and hypothesis testing.

---

## Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Running the Tests](#running-the-tests)
- [Output](#output)
- [References](#references)

---

## Overview

This project estimates the empirical distribution of stopping times in the Collatz map by uniformly sampling starting integers from `[1, N]`, iterating the map to convergence, and fitting a log-normal model to the resulting sample. The core research question is whether the stopping-time distribution is consistent with a log-normal model, and whether this consistency differs across residue classes modulo small integers.

The codebase is structured as five independent Python modules with a strict dependency order. No Jupyter notebooks are used; all experiments run from `main.py`.

---

## Mathematical Background

### The Collatz Map

The standard Collatz function is defined on positive integers as:

$$T(n) = \begin{cases} n / 2 & \text{if } n \equiv 0 \pmod{2} \\ (3n + 1) / 2 & \text{if } n \equiv 1 \pmod{2} \end{cases}$$

This is the Syracuse (accelerated) form, which applies one full odd step rather than splitting `3n+1` and `n/2` into separate cases. The Collatz conjecture asserts that for all `n >= 1`, repeated application of `T` eventually reaches 1. This project does not attempt to address the conjecture; it treats convergence as an empirical assumption.

### Stopping Time

The stopping time of `n` is:

$$\sigma(n) = \min \{ k \geq 1 : T^k(n) = 1 \}$$

where `T^k` denotes the `k`-fold composition of `T`. The function `sigma : Z_{>=1} -> Z_{>=1}` is well-defined for all `n` for which the Collatz conjecture holds; empirically, this includes all integers tested to date (verified up to approximately `2^68` [2]).

### Monte Carlo Estimation

Let `n` be drawn uniformly at random from `{1, 2, ..., N}`. The object of study is the induced random variable `sigma(n)` and its distributional properties as `N -> infinity`. Since no closed form for the distribution of `sigma(n)` is known, this project estimates it by sampling: drawing `M` independent realisations of `n ~ Uniform[1, N]`, computing `sigma(n)` for each, and analysing the resulting empirical distribution.

The estimator for the mean stopping time is:

$$\hat{\mu}_N = \frac{1}{M} \sum_{i=1}^{M} \sigma(n_i), \quad n_i \overset{\text{i.i.d.}}{\sim} \text{Uniform}\{1, \ldots, N\}$$

Consistency of `hat_mu_N` as an estimator of `E[sigma(n)]` follows from the law of large numbers, provided `sigma` is integrable under the uniform measure on `[1, N]`.

### Log-Normal Model

The Terras–Lagarias random walk heuristic [3, 4] predicts that `log sigma(n)` is approximately normally distributed. Specifically, under a probabilistic model that treats each step of the Collatz iteration as an independent Bernoulli trial with equal probability of halving or applying the odd rule, the heuristic yields:

$$\log \sigma(n) \approx \mathcal{N}\!\left(\mu \log n,\ \sigma^2 \log n\right)$$

where `mu` and `sigma^2` are constants determined by the expected contraction rate of the map. The contraction factor per step is `(3/4)` in expectation (one multiplication by 3 and one division by 4 per two-step cycle), giving `mu = log(4/3) / log 2` approximately. This implies that `sigma(n)` itself follows a log-normal distribution with parameters scaling logarithmically in `n`.

This project tests this prediction empirically. A log-normal distribution with parameters `(s, mu)` has probability density function:

$$f(x; s, \mu) = \frac{1}{x \cdot s \sqrt{2\pi}} \exp\!\left( -\frac{(\log x - \mu)^2}{2s^2} \right), \quad x > 0$$

The parameters are estimated by maximum likelihood via `scipy.stats.lognorm.fit` with the location parameter fixed at zero (`floc=0`).

### Goodness-of-Fit Test

The Kolmogorov–Smirnov (KS) statistic measures the maximum absolute deviation between the empirical CDF `F_n` and the fitted theoretical CDF `F`:

$$D_n = \sup_{x} \left| F_n(x) - F(x) \right|$$

Under the null hypothesis that the sample is drawn from `F`, the distribution of `D_n * sqrt(n)` converges to the Kolmogorov distribution as `n -> infinity`. A large `p`-value does not confirm that the data are log-normally distributed; it only indicates insufficient evidence to reject that hypothesis at the chosen significance level. This distinction is maintained throughout the analysis.

### Residue Stratification

The Collatz map has non-trivial modular structure. The image of the map satisfies:

$$T(n) \equiv \begin{cases} n/2 \pmod{2} & n \equiv 0 \pmod{2} \\ (3n+1)/2 \pmod{4} & n \equiv 1 \pmod{2} \end{cases}$$

This means that the conditional distribution `sigma(n | n ≡ r mod m)` may differ across residue classes `r` for modulus `m`. The project tests this by comparing mean stopping times and distributional shape across residue classes modulo 4 and 8. The relevant theoretical motivation is provided by Niu [1], who establishes a closed-form analytic count of paradoxical parity sequences for each residue class.

---

## Repository Structure

```
collatz-montecarlo/
├── src/
│   ├── collatz.py        core algorithm — stopping time computation
│   ├── sampler.py        uniform Monte Carlo sampler
│   ├── analysis.py       distribution fitting and hypothesis testing
│   ├── plots.py          figure generation
├── outputs/              saved figures (generated at runtime)
├── plots/              mcmc approach plots (generated at runtime)
├── main.py               experiment entry point
├── requirements.txt
└── README.md
```

The dependency order is strict: `collatz.py` has no internal imports; `sampler.py` imports from `collatz.py`; `analysis.py` imports from `sampler.py`; `plots.py` imports from `analysis.py`; `main.py` imports all four.

---

## Requirements

- Python 3.11 or later
- numpy
- scipy
- matplotlib
- pytest (for tests only)

All dependencies are pinned in `requirements.txt`.

---

## Installation

Clone the repository and install dependencies into a virtual environment.

```bash
git clone https://github.com/<your-username>/collatz-montecarlo.git
cd collatz-montecarlo
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import numpy, scipy, matplotlib; print('Dependencies installed.')"
```

---

## Usage

Run the full experiment pipeline with default parameters (`N = 1_000_000`, `size = 50_000`):

```bash
python main.py
```

To modify `N` and sample size, edit the `run_experiment` call at the bottom of `main.py`:

```python
run_experiment(N=500_000, size=20_000)
```

The pipeline executes in the following order:

1. Draw `size` integers uniformly from `[1, N]`
2. Compute stopping times via vectorised batch iteration
3. Fit a log-normal distribution by maximum likelihood
4. Run a one-sample KS test against the fitted distribution
5. Print summary statistics and the KS result to stdout
6. Save all figures to `outputs/`

Expected runtime on a modern laptop: under 30 seconds for `N = 1_000_000`, `size = 50_000`.

---

## Running the Tests

Tests use `pytest` and cover correctness of the core algorithm against the OEIS A006577 sequence, sampler boundary conditions, and the parameter conversion in `fit_lognormal`.

```bash
pytest tests/ -v
```

To run a single test file:

```bash
pytest tests/test_collatz.py -v
```

All tests should pass with zero warnings on a clean install. If any test fails, verify that Python 3.11+ is active and that all dependencies are installed from `requirements.txt`.

---

## Output

Running `main.py` produces three files in `outputs/`:

**`histogram.png`** — density histogram of `sigma(n)` on linear axes with the fitted log-normal PDF overlaid as a dashed curve.

**`cdf.png`** — empirical CDF vs fitted log-normal CDF on log-linear axes (log x-axis). Deviation between the two curves is the visual counterpart to the KS statistic.

**`mean_by_residue.png`** (optional, Week 5 extension) — bar chart of mean stopping time per residue class `r = 0, ..., 7` for `n ≡ r (mod 8)`, with the unconditional mean drawn as a horizontal reference line.

Summary statistics and the KS result are printed to stdout in the format:

```
mean         : 152.34
variance     : 4821.07
median       : 131.00
p95          : 298.00
p99          : 412.00
KS statistic : 0.0082
KS p-value   : 0.3471
```

---

## References

[1] Niu, T. (2026). *Parity vectors and paradoxical sequences in the accelerated Collatz map*. arXiv:2605.13882 [math.NT].

[2] Barina, D. (2021). *Convergence verification of the Collatz problem*. *The Journal of Supercomputing*, 77, 2681–2688.

[3] Terras, R. (1976). A stopping time problem on the positive integers. *Acta Arithmetica*, 30(3), 241–252.

[4] Lagarias, J. C. (1985). The 3x+1 problem and its generalizations. *The American Mathematical Monthly*, 92(1), 3–23.

[5] Lagarias, J. C. (Ed.). (2010). *The Ultimate Challenge: The 3x+1 Problem*. American Mathematical Society.
