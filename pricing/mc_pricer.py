import numpy as np
from numba import njit

# =======================
# Simulazione percorsi
# =======================

# Geometric Brownian Motion (GBM) dynamics under the risk‑neutral measure:
# dS_t = r * S_t * dt + σ * S_t * dW_t
# Analytical discrete solution for one time step:
# S_{t+Δt} = S_t * exp( (r - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * Z ),
# where Z ~ N(0,1)
# in the next function the imput z is a matrix of standard normal variables with shape (n_paths, steps)


@njit
def _gbm_paths_from_Z(S0, r, vol, T, Z):
    n_paths, steps = Z.shape
    dt = T / steps
    drift = (r - 0.5 * vol**2) * dt
    diff = vol * np.sqrt(dt)
    S = np.empty((n_paths, steps+1))
    for i in range(n_paths):
        S[i, 0] = S0
        logS = np.log(S0)
        for j in range(steps):
            logS += drift + diff * Z[i, j]
            S[i, j+1] = np.exp(logS)
    return S

def simulate_paths_fast(S0, r, vol, T, steps, n_paths, antithetic=False, seed=None):
    """
    Versione veloce: genera Z (Gaussiane) in NumPy e fa evolvere con Numba.
    Ritorna array (n_paths[×2 se antithetic], steps+1).
    """
    #random number generator object
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    return _gbm_paths_from_Z(S0, r, vol, T, Z)


# =======================
# MC Pricer 
# =======================
# All Monte Carlo pricers (European, Asian, Barrier) follow the same theoretical principle:
# 
#     V0 = exp(-r * T) * E_Q[ payoff(contract) ]
#
# i.e., the discounted expected value of the option payoff under the risk‑neutral measure.
# The main difference between option types lies in how the payoff is defined:
#   - European: depends only on the final price S_T
#   - Asian: depends on the average price over time
#   - Barrier: depends on whether the price path hits a threshold (barrier)




# =======================
# Payoff base european
# =======================

def payoff_call_put_terminal(ST, K, call=True):
    if call:
        return np.maximum(ST - K, 0.0)
    else:
        return np.maximum(K - ST, 0.0)

# =======================
# Pricer europeo (MC)
# =======================

def price_european_mc(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                      antithetic=False, seed=None, return_paths=False):
    # QUI usiamo la versione veloce
    paths = simulate_paths_fast(S0, r, vol, T, steps, n_paths, antithetic, seed)
    ST = paths[:, -1]
    payoff = payoff_call_put_terminal(ST, K, call=call) #same shape of ST; call is the same call of the father function
    disc = np.exp(-r*T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    if return_paths:
        return price, stderr, paths
    return price, stderr

def delta_vega_bump(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                    eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42):
    """
    Delta e Vega via differenze centrate con bump relativo.
    Delta denom = 2*eps_S*S0 ; Vega denom = 2*eps_vol*vol
    """
    base_seed = seed # to ensure same paths for up/down bumps
    # Derivata numerica centrata con bump relativo:
    # f'(x) ≈ [ f(x*(1 + ε)) - f(x*(1 - ε)) ] / (2 * ε * x)

    p_up, _ = price_european_mc(S0*(1+eps_S), K, r, vol, T, steps, n_paths, call, antithetic, base_seed)
    p_dn, _ = price_european_mc(S0*(1-eps_S), K, r, vol, T, steps, n_paths, call, antithetic, base_seed)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)

    p_upv, _ = price_european_mc(S0, K, r, vol*(1+eps_vol), T, steps, n_paths, call, antithetic, base_seed)
    p_dnv, _ = price_european_mc(S0, K, r, vol*(1-eps_vol), T, steps, n_paths, call, antithetic, base_seed)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)

    return delta, vega

# =======================
# Asian (payoff e pricer) use the mean S_mean instead of S_T
# =======================

def payoff_asian_arith(paths, K, call=True):
    avg = paths.mean(axis=1)
    return np.maximum(avg - K, 0.0) if call else np.maximum(K - avg, 0.0)

def price_asian_mc(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                   antithetic=True, seed=None, control_variate=True):
    paths = simulate_paths_fast(S0, r, vol, T, steps, n_paths, antithetic, seed)
    payoff_A = payoff_asian_arith(paths, K, call=call)

    disc = np.exp(-r*T)
    price_naive = disc * payoff_A.mean()
    stderr_naive = disc * payoff_A.std(ddof=1) / np.sqrt(len(payoff_A))

    if not control_variate:
        return price_naive, stderr_naive

    # Control variate: call europea sullo stesso ST
    # Control variate method to reduce Monte Carlo variance.
    # Asian options have no closed-form Black–Scholes price (path-dependent payoff),
    # so we use a correlated European call as control variate:
    # adj = payoff_A - beta * (payoff_C - payoff_C.mean())
    # This keeps the same expected value but lowers variance, improving estimation. To minimize variance, beta is cov(S_A, S_C) / var(S_C).
    ST = paths[:, -1]
    payoff_C = payoff_call_put_terminal(ST, K, call=call)
    cov = np.cov(payoff_A, payoff_C, ddof=1)
    beta = cov[0,1] / cov[1,1] if cov[1,1] > 0 else 0.0
    adj = payoff_A - beta * (payoff_C - payoff_C.mean())

    price_cv = disc * adj.mean()
    stderr_cv = disc * adj.std(ddof=1) / np.sqrt(len(adj))
    return price_cv, stderr_cv

def delta_vega_bump_asian(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                          eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42,
                          control_variate=True):
    p_up, _ = price_asian_mc(S0*(1+eps_S), K, r, vol, T, steps, n_paths, call,
                             antithetic, seed, control_variate)
    p_dn, _ = price_asian_mc(S0*(1-eps_S), K, r, vol, T, steps, n_paths, call,
                             antithetic, seed, control_variate)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)

    p_upv, _ = price_asian_mc(S0, K, r, vol*(1+eps_vol), T, steps, n_paths, call,
                              antithetic, seed, control_variate)
    p_dnv, _ = price_asian_mc(S0, K, r, vol*(1-eps_vol), T, steps, n_paths, call,
                              antithetic, seed, control_variate)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)
    return delta, vega

# =======================
# Barrier D&O (payoff e pricer)
# =======================

def payoff_barrier_down_and_out(paths, K, H, call=True):
    knocked = (paths.min(axis=1) <= H)
    ST = paths[:, -1]
    vanilla = payoff_call_put_terminal(ST, K, call=call)
    vanilla[knocked] = 0.0
    return vanilla

def price_barrier_mc(S0, K, H, r, vol, T, steps=252, n_paths=100_000, call=True,
                     antithetic=True, seed=None):
    paths = simulate_paths_fast(S0, r, vol, T, steps, n_paths, antithetic, seed)
    payoff = payoff_barrier_down_and_out(paths, K, H, call=call)
    disc = np.exp(-r*T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, stderr

def delta_vega_bump_barrier(S0, K, H, r, vol, T, steps=252, n_paths=100_000, call=True,
                            eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42):
    p_up, _ = price_barrier_mc(S0*(1+eps_S), K, H, r, vol, T, steps, n_paths, call, antithetic, seed)
    p_dn, _ = price_barrier_mc(S0*(1-eps_S), K, H, r, vol, T, steps, n_paths, call, antithetic, seed)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)

    p_upv, _ = price_barrier_mc(S0, K, H, r, vol*(1+eps_vol), T, steps, n_paths, call, antithetic, seed)
    p_dnv, _ = price_barrier_mc(S0, K, H, r, vol*(1-eps_vol), T, steps, n_paths, call, antithetic, seed)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)
    return delta, vega