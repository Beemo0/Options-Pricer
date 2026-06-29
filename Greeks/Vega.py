import numpy as np
from scipy.stats import norm

def Vega(option):
    S = option.S
    K = option.K
    t = option.t
    r = option.r
    sigma = option.sigma

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    vega = S * np.sqrt(t/2 * np.pi) * np.exp(-d1 **2 * 0.5)

    # Calcul du prix de l'option
    """if is_american:
        raise ValueError("Uniquement les options européennes")
    else:
        if option_type == "call":
            call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
            return call_price
        elif option_type == "put":
            put_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put_price
        else:
            raise ValueError("Le type d'option doit être 'call' ou 'put'")"""