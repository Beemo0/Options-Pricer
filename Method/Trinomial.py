import numpy as np


def Trinomial(option, option_type: str, N: int = 100, **kwargs):
    """
    Modèle trinomial (Boyle, 1986).
    Converge plus vite que le binomial pour le même N.
    Supporte les options européennes et américaines.

    Paramètres
    ----------
    option      : EuropeanOption | AmericanOption
    option_type : 'call' ou 'put'
    N           : nombre de pas (défaut 100)

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
    u  = np.exp(sigma * np.sqrt(2 * dt))   # facteur haussier
    d  = 1.0 / u                             # facteur baissier

    # ── Probabilités risque-neutres (formule Boyle correcte) ────────────────
    drift = np.exp((r - y) * dt / 2)
    su    = np.exp(sigma * np.sqrt(dt / 2))
    sd    = 1.0 / su

    pu = ((drift - sd) / (su - sd)) ** 2
    pd = ((su - drift) / (su - sd)) ** 2
    pm = 1.0 - pu - pd

    if not (pu > 0 and pd > 0 and pm > 0):
        raise ValueError(f"Probabilités invalides : pu={pu:.4f}, pm={pm:.4f}, pd={pd:.4f}.")

    discount = np.exp(-r * dt)

    # ── Grille des prix à maturité (2N+1 nœuds) ────────────────────────────
    k = np.arange(-N, N + 1)               # indice centré, shape (2N+1,)
    S_T = S * (u ** (-k))                  # nœud k correspond à S*u^(-k)
                                            # (k=-N → S*u^N haut, k=N → S*u^{-N} bas)

    # ── Payoffs à maturité ───────────────────────────────────────────────────
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # ── Backward induction ───────────────────────────────────────────────────
    for i in range(N - 1, -1, -1):
        # À l'étape i, les nœuds valides sont k ∈ [-i, i] → 2i+1 valeurs
        n_nodes = 2 * i + 1
        V_new = np.empty(n_nodes)

        # indices dans le vecteur courant (taille 2*(i+1)+1)
        for idx in range(n_nodes):
            # Dans la grille à l'étape i+1 :
            # nœud up   = idx          (décalage -1 dans les indices)
            # nœud mid  = idx + 1
            # nœud down = idx + 2
            V_new[idx] = (pu * V[idx] + pm * V[idx + 1] + pd * V[idx + 2]) * discount

        if is_american:
            k_i  = np.arange(-i, i + 1)
            S_i  = S * (u ** (-k_i))
            if option_type == "call":
                V_new = np.maximum(V_new, S_i - K)
            else:
                V_new = np.maximum(V_new, K - S_i)

        V = V_new

    return float(V[0])