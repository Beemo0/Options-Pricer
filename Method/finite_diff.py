"""
pricing_methods/finite_diff.py
===============================
Méthode des différences finies (schéma de Crank-Nicolson).
Résout la PDE de Black-Scholes sur une grille (S, t).

Crank-Nicolson : stable inconditionnellement, ordre 2 en temps et espace.
"""

import numpy as np
from scipy.linalg import solve_banded


def FiniteDifference(option, option_type: str,
                     M: int = 200,
                     N: int = 200,
                     S_max_factor: float = 3.0,
                     **kwargs):
    """
    Pricer Crank-Nicolson.

    Paramètres
    ----------
    option          : EuropeanOption | AmericanOption
    option_type     : 'call' | 'put'
    M               : pas de temps
    N               : pas d'espace (prix)
    S_max_factor    : S_max = S_max_factor * K
    """
    S0    = option.S
    K     = option.K
    T     = option.t
    r     = option.r
    sigma = option.sigma
    y     = getattr(option, "y", 0.0)
    is_am = getattr(option, "is_american", False)

    S_max = S_max_factor * max(S0, K)
    dS    = S_max / N
    dt    = T / M

    # Grille des prix : j = 0 … N
    j_arr = np.arange(0, N + 1)
    S_arr = j_arr * dS                  # S[j] = j * dS

    # Condition terminale
    if option_type == "call":
        V = np.maximum(S_arr - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_arr, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # Noeuds intérieurs : j = 1 … N-1
    j_int = j_arr[1:N]                  # shape (N-1,)

    # Coefficients PDE de Black-Scholes (sur chaque nœud intérieur)
    #   ∂V/∂t + ½σ²S²∂²V/∂S² + (r-y)S∂V/∂S - rV = 0
    alpha = 0.25 * dt * (sigma**2 * j_int**2 - (r - y) * j_int)
    beta  = -0.5 * dt * (sigma**2 * j_int**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j_int**2 + (r - y) * j_int)

    n_int = N - 1

    # Matrice LHS (implicite) : I - L/2  (tridiagonale)
    # Matrice RHS (explicite) : I + L/2
    # au format banded (sous-diag, diag, sur-diag)
    lhs = np.zeros((3, n_int))
    lhs[2, :-1] = -alpha[1:]    # sous-diag (scipy : ligne 2)
    lhs[1,  : ] = 1 - beta      # diagonale
    lhs[0,  1:] = -gamma[:-1]   # sur-diag  (scipy : ligne 0)

    for step in range(M):
        # ── Conditions aux bords ─────────────────────────────────────────────
        tau = (step + 1) * dt          # temps restant APRÈS ce pas
        disc = np.exp(-r * tau)

        if option_type == "call":
            V[0]  = 0.0
            V[-1] = S_max - K * disc
        else:
            V[0]  = K * disc
            V[-1] = 0.0

        # ── RHS = (I + L/2) * V_int + corrections bords ─────────────────────
        V_int = V[1:N]

        rhs = (alpha * V[0:N-1] + (1 + beta) * V_int + gamma * V[2:N+1])
        # Ajout des termes de bord (alpha[0]*V[0] et gamma[-1]*V[N])
        rhs[0]  += alpha[0]  * V[0]
        rhs[-1] += gamma[-1] * V[N]

        # ── Résolution ───────────────────────────────────────────────────────
        V_new = solve_banded((1, 1), lhs, rhs)

        # ── Exercice anticipé (américain) ────────────────────────────────────
        if is_am:
            S_int = j_int * dS
            if option_type == "call":
                V_new = np.maximum(V_new, S_int - K)
            else:
                V_new = np.maximum(V_new, K - S_int)

        V[1:N] = V_new

    # ── Interpolation linéaire en S0 ─────────────────────────────────────────
    j0   = S0 / dS
    j_lo = int(j0)
    j_lo = min(max(j_lo, 0), N - 1)
    frac = j0 - j_lo
    return float((1 - frac) * V[j_lo] + frac * V[min(j_lo + 1, N)])
