import numpy as np

def simulate_paths(S0, r, vol, T, steps, n_paths, antithetic=False, seed=None):
    """
    Simula traiettorie del prezzo con moto browniano geometrico (GBM).
    Ritorna array shape (n_paths, steps+1).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / steps
    # gaussiane standard per ogni step e path
    Z = rng.standard_normal(size=(n_paths, steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)  # raddoppia i path con opposti

    drift = (r - 0.5 * vol**2) * dt
    diff = vol * np.sqrt(dt) * Z

    # log S_t = log S0 + cumulata(drift + diff)
    log_S = np.log(S0) + np.cumsum(drift + diff, axis=1)
    S = np.empty((log_S.shape[0], steps+1))
    S[:, 0] = S0
    S[:, 1:] = np.exp(log_S)
    return S

def payoff_call_put_terminal(ST, K, call=True):
    if call:
        return np.maximum(ST - K, 0.0)
    else:
        return np.maximum(K - ST, 0.0)

def price_european_mc(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                      antithetic=False, seed=None, return_paths=False):
    paths = simulate_paths(S0, r, vol, T, steps, n_paths, antithetic, seed)
    ST = paths[:, -1]
    payoff = payoff_call_put_terminal(ST, K, call=call)
    disc = np.exp(-r*T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    if return_paths:
        return price, stderr, paths
    return price, stderr

def delta_vega_bump(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                    eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42):
    """
    Delta e Vega via differenze centrate.
    - Delta: bump relativo su S0 (±eps_S*S0) → denom 2*eps_S*S0
    - Vega:  bump relativo su vol (±eps_vol*vol) → denom 2*eps_vol*vol
    Riusa gli stessi random numbers (common random numbers).
    """
    base_seed = seed

    # Delta
    p_up, _ = price_european_mc(S0*(1+eps_S), K, r, vol, T, steps, n_paths, call, antithetic, base_seed)
    p_dn, _ = price_european_mc(S0*(1-eps_S), K, r, vol, T, steps, n_paths, call, antithetic, base_seed)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)

    # Vega (ATTENZIONE al denominatore: bump relativo → *vol)
    p_upv, _ = price_european_mc(S0, K, r, vol*(1+eps_vol), T, steps, n_paths, call, antithetic, base_seed)
    p_dnv, _ = price_european_mc(S0, K, r, vol*(1-eps_vol), T, steps, n_paths, call, antithetic, base_seed)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)

    return delta, vega

def payoff_asian_arith(paths, K, call=True):
    avg = paths.mean(axis=1)  # media lungo il percorso
    if call:
        return np.maximum(avg - K, 0.0)
    else:
        return np.maximum(K - avg, 0.0)

def price_asian_mc(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                   antithetic=True, seed=None, control_variate=True):
    # simula percorsi
    paths = simulate_paths(S0, r, vol, T, steps, n_paths, antithetic, seed)
    payoff_A = payoff_asian_arith(paths, K, call=call)

    disc = np.exp(-r*T)
    price_naive = disc * payoff_A.mean()
    stderr_naive = disc * payoff_A.std(ddof=1) / np.sqrt(len(payoff_A))

    if not control_variate:
        return price_naive, stderr_naive

    # Control variate: usa la call europea sullo stesso ST
    ST = paths[:, -1]
    payoff_C = payoff_call_put_terminal(ST, K, call=call)
    cov = np.cov(payoff_A, payoff_C, ddof=1)
    beta = cov[0,1] / cov[1,1] if cov[1,1] > 0 else 0.0

    adj = payoff_A - beta * (payoff_C - payoff_C.mean())
    price_cv = disc * adj.mean()
    stderr_cv = disc * adj.std(ddof=1) / np.sqrt(len(adj))
    return price_cv, stderr_cv

def payoff_barrier_down_and_out(paths, K, H, call=True):
    # toccare la barriera → payoff zero
    knocked = (paths.min(axis=1) <= H)
    ST = paths[:, -1]
    vanilla = payoff_call_put_terminal(ST, K, call=call)
    vanilla[knocked] = 0.0
    return vanilla

def price_barrier_mc(S0, K, H, r, vol, T, steps=252, n_paths=100_000, call=True,
                     antithetic=True, seed=None):
    paths = simulate_paths(S0, r, vol, T, steps, n_paths, antithetic, seed)
    payoff = payoff_barrier_down_and_out(paths, K, H, call=call)
    disc = np.exp(-r*T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, stderr


def delta_vega_bump_asian(S0, K, r, vol, T, steps=252, n_paths=100_000, call=True,
                          eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42,
                          control_variate=True):
    # Delta (bump relativo su S0)
    p_up, _ = price_asian_mc(S0*(1+eps_S), K, r, vol, T, steps, n_paths, call,
                             antithetic, seed, control_variate)
    p_dn, _ = price_asian_mc(S0*(1-eps_S), K, r, vol, T, steps, n_paths, call,
                             antithetic, seed, control_variate)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)
    # Vega (bump relativo su vol)
    p_upv, _ = price_asian_mc(S0, K, r, vol*(1+eps_vol), T, steps, n_paths, call,
                              antithetic, seed, control_variate)
    p_dnv, _ = price_asian_mc(S0, K, r, vol*(1-eps_vol), T, steps, n_paths, call,
                              antithetic, seed, control_variate)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)
    return delta, vega

def delta_vega_bump_barrier(S0, K, H, r, vol, T, steps=252, n_paths=100_000, call=True,
                            eps_S=1e-3, eps_vol=1e-3, antithetic=True, seed=42):
    # Delta
    p_up, _ = price_barrier_mc(S0*(1+eps_S), K, H, r, vol, T, steps, n_paths, call,
                               antithetic, seed)
    p_dn, _ = price_barrier_mc(S0*(1-eps_S), K, H, r, vol, T, steps, n_paths, call,
                               antithetic, seed)
    delta = (p_up - p_dn) / (2.0 * eps_S * S0)
    # Vega
    p_upv, _ = price_barrier_mc(S0, K, H, r, vol*(1+eps_vol), T, steps, n_paths, call,
                                antithetic, seed)
    p_dnv, _ = price_barrier_mc(S0, K, H, r, vol*(1-eps_vol), T, steps, n_paths, call,
                                antithetic, seed)
    vega = (p_upv - p_dnv) / (2.0 * eps_vol * vol)
    return delta, vega

