import numpy as np
from Option_type.Exotic import AsianOption, BarrierOption, LookbackOption


def MonteCarlo(option, option_type: str,
               n_simulations: int = 100_000,
               n_steps: int = 252,
               seed: int = None,
               antithetic: bool = True,
               **kwargs):
    """
    Pricer Monte Carlo avec réduction de variance (variables antithétiques).
    Supporte : Européen, Américain (Longstaff-Schwartz), Asiatique,
               Barrière, Lookback.

    Paramètres
    ----------
    option        : BaseOption (et sous-classes)
    option_type   : 'call' ou 'put'
    n_simulations : nombre de trajectoires (défaut 100 000)
    n_steps       : nombre de pas de temps par trajectoire (défaut 252)
    seed          : graine aléatoire pour la reproductibilité
    antithetic    : variables antithétiques (réduit la variance, défaut True)

    Retourne
    --------
    dict : {'price': float, 'std_error': float, 'ci_95': (float, float)}
    """
    rng = np.random.default_rng(seed)

    S, K, t, r, sigma = option.S, option.K, option.t, option.r, option.sigma
    y = getattr(option, "y", 0.0)
    is_american = getattr(option, "is_american", False)

    dt   = t / n_steps
    disc = np.exp(-r * t)

    # ── Simulation des trajectoires ──────────────────────────────────────────
    n_half = n_simulations // 2 if antithetic else n_simulations

    Z = rng.standard_normal((n_half, n_steps))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)   # (n_simulations, n_steps)

    # Incréments log-normaux
    drift_step = (r - y - 0.5 * sigma ** 2) * dt
    diff_step  = sigma * np.sqrt(dt)

    log_returns = drift_step + diff_step * Z              # (n_sim, n_steps)
    path_log    = np.cumsum(log_returns, axis=1)
    paths       = S * np.exp(path_log)                    # (n_sim, n_steps)
    # Ajouter S0 en début de trajectoire
    S0_col      = np.full((paths.shape[0], 1), S)
    paths_full  = np.concatenate([S0_col, paths], axis=1) # (n_sim, n_steps+1)

    S_T = paths_full[:, -1]                               # prix finaux

    # ── Payoffs selon le type d'option ───────────────────────────────────────

    # --- Option asiatique ---
    if isinstance(option, AsianOption):
        if option.averaging == "arithmetic":
            avg = paths_full[:, 1:].mean(axis=1)
        else:
            avg = np.exp(np.log(paths_full[:, 1:]).mean(axis=1))

        if option.avg_type == "price":
            payoffs = np.maximum(avg - K, 0) if option_type == "call" else np.maximum(K - avg, 0)
        else:  # strike flottant
            payoffs = np.maximum(S_T - avg, 0) if option_type == "call" else np.maximum(avg - S_T, 0)

    # --- Option à barrière ---
    elif isinstance(option, BarrierOption):
        B  = option.barrier
        bt = option.barrier_type
        S_paths = paths_full[:, 1:]

        if bt == "down-and-out":
            crossed = (S_paths.min(axis=1) <= B)
        elif bt == "down-and-in":
            crossed = (S_paths.min(axis=1) <= B)
        elif bt == "up-and-out":
            crossed = (S_paths.max(axis=1) >= B)
        elif bt == "up-and-in":
            crossed = (S_paths.max(axis=1) >= B)
        else:
            raise ValueError(f"barrier_type inconnu : {bt}")

        base = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)

        if "out" in bt:
            payoffs = np.where(crossed, 0.0, base)
        else:  # "in"
            payoffs = np.where(crossed, base, 0.0)

    # --- Option lookback ---
    elif isinstance(option, LookbackOption):
        S_paths = paths_full[:, 1:]
        if option.lookback_type == "floating":
            # Call : S_T - min ; Put : max - S_T
            if option_type == "call":
                payoffs = S_T - S_paths.min(axis=1)
            else:
                payoffs = S_paths.max(axis=1) - S_T
        else:  # fixed strike
            if option_type == "call":
                payoffs = np.maximum(S_paths.max(axis=1) - K, 0)
            else:
                payoffs = np.maximum(K - S_paths.min(axis=1), 0)

    # --- Option américaine (Longstaff-Schwartz LSM) ---
    elif is_american:
        payoffs = _longstaff_schwartz(paths_full, K, r, dt, option_type)

    # --- Option européenne standard ---
    else:
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(K - S_T, 0.0)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'.")

    # ── Actualisation et statistiques ────────────────────────────────────────
    discounted = disc * payoffs
    price      = discounted.mean()
    std_err    = discounted.std(ddof=1) / np.sqrt(len(discounted))
    ci         = (price - 1.96 * std_err, price + 1.96 * std_err)

    return {"price": float(price), "std_error": float(std_err), "ci_95": ci}


# ── Longstaff-Schwartz (LSM) ────────────────────────────────────────────────

def _longstaff_schwartz(paths_full, K, r, dt, option_type):
    """
    Algorithme Longstaff-Schwartz pour les options américaines.
    Régression sur base polynomiale de Laguerre (degré 2).

    paths_full : (n_sim, n_steps+1) — inclut S0 à l'indice 0
    """
    n_sim, n_steps_p1 = paths_full.shape
    n_steps = n_steps_p1 - 1
    disc_step = np.exp(-r * dt)

    if option_type == "call":
        intrinsic = lambda S_: np.maximum(S_ - K, 0.0)
    else:
        intrinsic = lambda S_: np.maximum(K - S_, 0.0)

    # Cashflows : initialisés au payoff final
    cash_flows = intrinsic(paths_full[:, -1])

    for i in range(n_steps - 1, 0, -1):
        S_i = paths_full[:, i]
        itm = intrinsic(S_i) > 0          # masque in-the-money

        if itm.sum() == 0:
            cash_flows *= disc_step
            continue

        # Régression quadratique sur les trajectoires ITM
        X   = S_i[itm]
        Y   = cash_flows[itm] * disc_step   # valeur continuation actualisée
        A   = np.column_stack([np.ones_like(X), X, X ** 2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            continuation = A @ coeffs
        except np.linalg.LinAlgError:
            cash_flows *= disc_step
            continue

        # Exercice anticipé si intrinsic > continuation estimée
        exercise = intrinsic(X) >= continuation
        exercise_idx = np.where(itm)[0][exercise]

        cash_flows               *= disc_step
        cash_flows[exercise_idx]  = intrinsic(S_i[exercise_idx])

    return cash_flows * disc_step