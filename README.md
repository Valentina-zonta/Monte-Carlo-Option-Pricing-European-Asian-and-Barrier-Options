# Python Option Pricing Engine (Monte Carlo & Black-Scholes)

## Overview
This project provides a Python-based framework for pricing financial options using both analytical and numerical methods. It features:

- **Analytical Pricer:** Implements the closed-form Black-Scholes-Merton (BSM) formula for European options (price, Delta, Vega).
- **Monte Carlo (MC) Pricer:** A flexible simulation engine for pricing various option types by simulating the underlying asset's price paths using Geometric Brownian Motion (GBM).

The library is designed to price standard European options as well as path-dependent exotic options, including:

- Asian Options (arithmetic average)
- Barrier Options (Down-and-Out)

A key focus is on numerical efficiency and accuracy, incorporating Numba for JIT-compilation to accelerate simulations and employing variance reduction techniques (Antithetic Variates and Control Variates).

---

## Core Components

### `black_scholes.py`
Contains the analytical, closed-form solutions for European call/put prices and the Greeks (Delta, Vega). This serves as a benchmark for validating the Monte Carlo engine.

### `mc_pricer.py`
The main, high-performance Monte Carlo engine.

- Uses Numba (@njit) to significantly speed up the GBM path simulation loop.
- Implements pricing functions for European, Asian, and Barrier options.
- Calculates Greeks (Delta, Vega) for all option types using the "bump-and-revalue" (finite difference) method with common random numbers.

### `mc_pricer_slow.py`
A pure-NumPy reference implementation of the MC pricer. This file is used to benchmark the performance gains achieved by the Numba-optimized version.

### (Analysis Notebooks/Scripts)
The provided screenshots are taken from scripts used to run simulations, analyze performance, and visualize the results.

---

## Analysis of Results
The project's effectiveness is validated through several key analyses, as shown in the execution screenshots.

### European Option Validation (MC vs. Black-Scholes)
The European MC pricer is benchmarked against the exact Black-Scholes analytical price.

**Parameters:**  
S0=100, K=100, r=0.02, vol=0.20, T=1.0, n_paths=200,000

**Price Comparison:**
- MC Price: 8.9113 (± 0.0218)
- BS Price: 8.9160
- Relative Error: 0.053%

**Greeks Comparison:**
- Delta (MC / BS): 0.5795 / 0.5793
- Vega (MC / BS): 39.2288 / 39.1043

**Conclusion:** The Monte Carlo results are extremely close to the analytical BSM values for both the price and the first-order Greeks. The low relative error and tight standard error (SE) validate the correctness of the GBM simulation and the "bump-and-revalue" implementation.

---

### Monte Carlo Convergence
A log-log plot was generated to analyze the error as a function of the number of simulations (N).  
The plot compares the actual error (|MC Price - BS Price|) and the estimated standard error (MC SE) against the theoretical convergence rate of 1/sqrt(N).

**Conclusion:** Both error metrics are shown to follow the 1/sqrt(N) slope, confirming that the pricer behaves exactly as expected by central limit theorem. This demonstrates that simulation accuracy can be reliably increased by adding more paths.

---

### Exotic Options and Variance Reduction
The engine was used to price path-dependent Asian and Barrier options.

#### Asian Option (with Control Variate):
- Asian (Naive): 5.0333 ± 0.0120
- Asian (Control Variate): 5.0333 ± 0.0066

**Conclusion:** The use of a control variate (a European option) successfully reduced the variance by a factor of 1.8, achieving a much smaller standard error for the same computational effort. The price estimate remains consistent.

#### Barrier & Asian Greeks:
- Barrier D&O (H=90.0): 7.4983 ± 0.0215
- Asian Delta: 0.5473 | Asian Vega: 22.4781
- Barrier Delta: 0.7171 | Barrier Vega: 19.1029

**Conclusion:** The "bump-and-revalue" method is successfully extended to price the Greeks of complex, path-dependent options for which no simple analytical formula exists.

---

### Performance: Numba vs. Pure NumPy
The project includes two MC implementations: a pure-NumPy version (`mc_pricer_slow.py`) and a Numba-accelerated version (`mc_pricer.py`).  
The Numba version's "first run" timings (which include the one-time JIT compilation cost) are already competitive.

Subsequent runs (shown in the screenshot for 200k, 500k, and 1M paths) are significantly faster than the pure NumPy equivalent, especially as n_paths and steps increase. This highlights the value of numba for computationally-intensive simulation tasks.
