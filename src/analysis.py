import numpy as np
import numpy.typing as npt
from scipy import stats
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KSResult:
    statistic: float
    pvalue: float


def empirical_cdf(sigma):
    if sigma.size == 0:
        return ValueError("Error")
    x = np.sort(sigma).astype(float)
    p = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, p


def fit_lognormal(sigma):
    s, loc, scale = stats.lognorm.fit(data=sigma, floc=0)
    mu = math.log(scale)
    return s, mu


def ks_test(sigma, s, mu):
    dist = stats.lognorm(s=s, loc=0, scale=math.exp(mu))
    result = stats.kstest(sigma, dist.cdf)
    return KSResult(statistic=float(result.statistic), pvalue=float(result.pvalue))


def summary(sigma):
    return {
        "Mean": float(np.mean(sigma)),
        "Variance": float(np.var(sigma)),
        "Median": float(np.median(sigma)),
        "P95": float(np.percentile(sigma, 95)),
        "P99": float(np.percentile(sigma, 99))
    }


def _lognormal_pdf(x, s, mu):
    dist = stats.lognorm(s=s, loc=0, scale=math.exp(mu))
    return np.array(dist.pdf(x), dtype=float)


def terras_baseline(n_value, k):
    log_n = np.log(n_value.astype(float))
    log_ratio = math.log(3.0/4.0)
    mean_log = k * log_ratio * log_n
    return np.exp(mean_log)
