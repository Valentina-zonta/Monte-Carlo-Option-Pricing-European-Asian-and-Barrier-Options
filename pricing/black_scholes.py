import numpy as np
from scipy.stats import norm

def bs_price(S0, K, r, vol, T, call=True):
    """
    Compute the price of a European call or put option
    using the Black–Scholes–Merton analytical formula.

    Parameters
    ----------
    S0 : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Annualized risk-free interest rate.
    vol : float
        Annualized volatility (σ) of the underlying asset.
    T : float
        Time to maturity, in years.
    call : bool, optional
        True for a call option, False for a put (default is True).

    Returns
    -------
    float
        Present value of the option according to the Black–Scholes model.

    Notes
    -----
    - Formula:
        d1 = [ ln(S0/K) + (r + 0.5 * vol^2) * T ] / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        Call = S0 * N(d1) - K * exp(-r*T) * N(d2)
        Put  = K * exp(-r*T) * N(-d2) - S0 * N(-d1)

    - For degenerate cases (vol <= 0 or T <= 0):
        returns the discounted intrinsic value as a fallback:
        exp(-r*T) * max(S0 - K, 0)  for a call
        exp(-r*T) * max(K - S0, 0)  for a put
    """
    S0, K, r, vol, T = map(float, (S0, K, r, vol, T))
    if vol <= 0 or T <= 0:
        # payoff a scadenza scontato come fallback
        return np.exp(-r*T)*max((S0-K) if call else (K-S0), 0.0)
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    if call:
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

def bs_delta(S0, K, r, vol, T, call=True):
    """
    Compute the Delta of a European call or put option under
    the Black–Scholes–Merton model.

    Delta measures the sensitivity of the option price
    to changes in the underlying price.

    Formulas:
        d1 = [ ln(S0/K) + (r + 0.5*vol^2)*T ] / (vol*sqrt(T))
        Call: Δ = N(d1)
        Put : Δ = N(d1) - 1  = -N(-d1)
    """
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    return norm.cdf(d1) if call else (norm.cdf(d1) - 1.0)

def bs_vega(S0, K, r, vol, T):
    """
    Compute the Vega of a European option under
    the Black–Scholes–Merton model.

    Vega measures the sensitivity of the option price
    to changes in volatility (σ).

    Formulas:
        d1 = [ ln(S0/K) + (r + 0.5*vol^2)*T ] / (vol*sqrt(T))
        Vega = S0 * φ(d1) * sqrt(T)

    (φ(d1) is the standard normal probability density)
    """
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    # vega per 1 unità di vol (non in %)
    return S0 * norm.pdf(d1) * np.sqrt(T)