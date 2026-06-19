# Collatz Stopping-Time Distributions via Monte Carlo Sampling

An empirical study of stopping-time distributions in the Collatz map using uniform Monte Carlo sampling, distribution fitting, and hypothesis testing.


## Repository Structure

```
collatz-montecarlo/
├── src/
│   ├── collatz.py        
│   ├── sampler.py        
│   ├── analysis.py       
│   ├── plots.py          
├── outputs/              
├── plots/             
├── main.py               
├── requirements.txt
└── README.md
```

The dependency order is strict: `collatz.py` has no internal imports; `sampler.py` imports from `collatz.py`; `analysis.py` imports from `sampler.py`; `plots.py` imports from `analysis.py`; `main.py` imports all four.

## Output 

For a positive integer N, the collatz map is defined as:

$$T(n) = \begin{cases} n/2 & n \text{ even} \\ (3n+1)/2 & n \text{ odd} \end{cases}$$

The stopping time is the smallest such that $T^k(n) = 1$. This project samples $n \sim \text{Uniform}[1, N]$ and characterizes the empirical distributions of stopping times statistically using parity vectors.

## Mathematics Behind It

**Log-Normal Heuristic** Terras [1] showed that under a random parity, succesive applications of the collatz map in expectation of odd steps and even steps. Lagrias [2] formalized on this hypothesis:

$$\log \sigma(n) \approx \mathcal{N}\!\left(\mu \log n,\ \sigma^2 \log n\right)$$

This project tests whether the empirical distributions of stopping times is consistent with the Log-Normal model, using the KS-Test as the criterion. 


## References

[1] Niu, T. (2026). *Parity vectors and paradoxical sequences in the accelerated Collatz map*. arXiv:2605.13882 [math.NT].

[2] Barina, D. (2021). *Convergence verification of the Collatz problem*. *The Journal of Supercomputing*, 77, 2681–2688.

[3] Terras, R. (1976). A stopping time problem on the positive integers. *Acta Arithmetica*, 30(3), 241–252.

[4] Lagarias, J. C. (1985). The 3x+1 problem and its generalizations. *The American Mathematical Monthly*, 92(1), 3–23.

[5] Lagarias, J. C. (Ed.). (2010). *The Ultimate Challenge: The 3x+1 Problem*. American Mathematical Society.
