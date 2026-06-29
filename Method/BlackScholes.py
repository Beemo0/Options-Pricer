import numpy as np
from scipy.stats import norm


def _d1_d2(option):
    """Calcule d1 et d2 selon Black-Scholes-Merton (avec dividende y)."""
    S, K, t, r, sigma, y = option.S, option.K, option.t, option.r, option.sigma, option.y
    d1 = (np.log(S / K) + (r - y + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def BlackScholes(option, option_type: str, **kwargs):
    """
    Pricer Black-Scholes (analytique) — options européennes uniquement.

    Paramètres
    ----------
    option      : EuropeanOption
    option_type : 'call' ou 'put'

    Retourne
    --------
    float : prix de l'option
    """
    if getattr(option, "is_american", False):
        raise ValueError(
            "Black-Scholes analytique ne supporte pas les options américaines. "
            "Utilisez Binomial, Trinomial ou FiniteDifference."
        )

    d1, d2 = _d1_d2(option)
    S, K, t, r, y = option.S, option.K, option.t, option.r, option.y
    disc = np.exp(-r * t)
    fwd  = S * np.exp(-y * t)

    if option_type == "call":
        return fwd * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option_type == "put":
        return K * disc * norm.cdf(-d2) - fwd * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")