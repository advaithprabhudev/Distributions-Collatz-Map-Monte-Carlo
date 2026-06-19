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


## References

[1] Niu, T. (2026). *Parity vectors and paradoxical sequences in the accelerated Collatz map*. arXiv:2605.13882 [math.NT].

[2] Barina, D. (2021). *Convergence verification of the Collatz problem*. *The Journal of Supercomputing*, 77, 2681–2688.

[3] Terras, R. (1976). A stopping time problem on the positive integers. *Acta Arithmetica*, 30(3), 241–252.

[4] Lagarias, J. C. (1985). The 3x+1 problem and its generalizations. *The American Mathematical Monthly*, 92(1), 3–23.

[5] Lagarias, J. C. (Ed.). (2010). *The Ultimate Challenge: The 3x+1 Problem*. American Mathematical Society.
