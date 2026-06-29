import numpy as np


def Binomial(option, option_type: str, N: int = 200, **kwargs):
    """
    Modèle binomial de Cox-Ross-Rubinstein.
    Supporte les options européennes et américaines.

    Paramètres
    ----------
    option      : EuropeanOption | AmericanOption
    option_type : 'call' ou 'put'
    N           : nombre de pas (défaut 200)

    Retourne
    --------
    float : prix de l'option
    """
    S     = option.S
    K     = option.K
    t     = option.t
    r     = option.r
    sigma = option.sigma
    y     = getattr(option, "y", 0.0)
    is_american = getattr(option, "is_american", False)

    dt = t / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - y) * dt) - d) / (u - d)     # probabilité risque-neutre

    if not (0 < p < 1):
        raise ValueError(f"Probabilité hors [0,1] : p={p:.4f}. Vérifiez les paramètres.")

    discount = np.exp(-r * dt)

    # ── Prix finaux du sous-jacent (vectorisé) ───────────────────────────────
    j = np.arange(N + 1)
    S_T = S * (u ** (N - j)) * (d ** j)           # shape (N+1,)

    # ── Payoffs à maturité ───────────────────────────────────────────────────
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # ── Backward induction (vectorisé) ──────────────────────────────────────
    for i in range(N - 1, -1, -1):
        # Prix du sous-jacent aux nœuds i
        j_i  = np.arange(i + 1)
        S_i  = S * (u ** (i - j_i)) * (d ** j_i)

        # Valeur de continuation
        V = (p * V[:i + 1] + (1 - p) * V[1:i + 2]) * discount

        if is_american:
            if option_type == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])