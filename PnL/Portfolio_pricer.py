"""
pnl/portfolio_pricer.py
========================
Price un portefeuille de positions avec TOUTES les méthodes disponibles.
Calcule :
  - Prix consensus (moyenne pondérée des méthodes)
  - Spread d'incertitude (écart-type inter-méthodes)
  - Erreur vs prix marché
  - Grecs agrégés du portefeuille
  - Recommandations de hedge delta/vega/gamma
"""

import numpy as np
import pandas as pd
import copy, warnings
warnings.filterwarnings("ignore")

from Method.BlackScholes import BlackScholes
from Method.Binomial      import Binomial
from Method.Trinomial     import Trinomial
from Method.MonteCarlo    import MonteCarlo
from Method.finite_diff   import FiniteDifference
from Greeks.greeks                 import Greeks

from Option_type.European  import EuropeanOption
from Option_type.American  import AmericanOption
from Option_type.Exotic    import AsianOption, BarrierOption, LookbackOption

from data.Portfolio_generator import Position


# ─────────────────────────────────────────────────────────────────────────────
# Config des méthodes
# ─────────────────────────────────────────────────────────────────────────────

METHODS = {
    "BlackScholes" : (BlackScholes,    {"european_only": True},  1.0),
    "Binomial"     : (Binomial,         {"N": 150},               1.0),
    "Trinomial"    : (Trinomial,        {"N": 80},                1.0),
    "MonteCarlo"   : (MonteCarlo,       {"n_simulations": 20_000,
                                          "seed": 42},            0.8),
    "FiniteDiff"   : (FiniteDifference, {"M": 150, "N": 150},     0.9),
}

# Méthodes compatibles par style
STYLE_COMPAT = {
    "european" : ["BlackScholes", "Binomial", "Trinomial", "MonteCarlo", "FiniteDiff"],
    "american" : ["Binomial", "Trinomial", "MonteCarlo", "FiniteDiff"],
    "asian"    : ["MonteCarlo"],
    "barrier"  : ["MonteCarlo"],
    "lookback" : ["MonteCarlo"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Construction de l'objet option depuis une Position
# ─────────────────────────────────────────────────────────────────────────────

def _build_option(pos: Position):
    s = pos.style
    if s == "european":
        return EuropeanOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y)
    elif s == "american":
        return AmericanOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y)
    elif s == "asian":
        return AsianOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y,
                           averaging=pos.exotic_params.get("averaging", "arithmetic"),
                           avg_type=pos.exotic_params.get("avg_type", "price"))
    elif s == "barrier":
        return BarrierOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y,
                             barrier=pos.exotic_params.get("barrier", pos.S * 0.85),
                             barrier_type=pos.exotic_params.get("barrier_type", "down-and-out"))
    elif s == "lookback":
        return LookbackOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y,
                              lookback_type=pos.exotic_params.get("lookback_type", "floating"))
    raise ValueError(f"Style inconnu : {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Pricing d'une seule position
# ─────────────────────────────────────────────────────────────────────────────

def price_position(pos: Position) -> dict:
    """
    Price une position avec toutes les méthodes compatibles.

    Retourne
    --------
    dict avec :
      prices        : {method: price}
      consensus     : prix moyen pondéré
      uncertainty   : écart-type inter-méthodes
      market_price  : prix marché mid
      mispricing    : consensus - market_price
      greeks        : dict des grecs (sur l'option européenne BS si disponible)
      pnl_unit      : consensus - entry_price
      pnl_total     : pnl_unit * quantity
    """
    opt    = _build_option(pos)
    compat = STYLE_COMPAT.get(pos.style, ["MonteCarlo"])

    prices  = {}
    weights = []

    for name in compat:
        fn, kw, weight = METHODS[name]
        kw = {k: v for k, v in kw.items() if k != "european_only"}
        try:
            res = fn(opt, pos.option_type, **kw)
            p   = res["price"] if isinstance(res, dict) else float(res)
            if np.isfinite(p) and p >= 0:
                prices[name]  = round(p, 5)
                weights.append(weight)
        except Exception:
            pass

    if not prices:
        return _empty_result(pos)

    vals   = list(prices.values())
    w_arr  = np.array(weights[:len(vals)])
    w_arr /= w_arr.sum()

    consensus   = float(np.average(vals, weights=w_arr))
    uncertainty = float(np.std(vals)) if len(vals) > 1 else 0.0
    mispricing  = consensus - pos.market_price
    pnl_unit    = (consensus - pos.entry_price) * np.sign(pos.quantity)
    pnl_total   = pnl_unit * abs(pos.quantity)

    # Grecs (BS analytique pour européen, numérique sinon)
    greeks = _compute_greeks(pos, opt, prices)

    return {
        "pos_id"      : pos.pos_id,
        "strategy"    : pos.strategy,
        "option_type" : pos.option_type,
        "style"       : pos.style,
        "K"           : pos.K,
        "t"           : pos.t,
        "sigma"       : pos.sigma,
        "quantity"    : pos.quantity,
        "entry_price" : pos.entry_price,
        "market_price": pos.market_price,
        "prices"      : prices,
        "consensus"   : round(consensus, 5),
        "uncertainty" : round(uncertainty, 5),
        "mispricing"  : round(mispricing, 5),
        "pnl_unit"    : round(pnl_unit, 5),
        "pnl_total"   : round(pnl_total, 5),
        "greeks"      : greeks,
    }


def _compute_greeks(pos, opt, prices):
    try:
        # Pour les exotiques, on proxy avec une européenne même strike/maturity
        if pos.style not in ("european", "american"):
            opt_eu = EuropeanOption(pos.S, pos.K, pos.t, pos.r, pos.sigma, pos.y)
            g = Greeks(opt_eu, pos.option_type)
        elif pos.style == "american":
            g = Greeks(opt, pos.option_type, Binomial, N=100)
        else:
            g = Greeks(opt, pos.option_type)
        raw = g.all()
        # Pondérer par la quantité
        return {k: round(v * pos.quantity, 6) for k, v in raw.items()}
    except Exception:
        return {k: 0.0 for k in ["delta", "gamma", "vega", "theta", "rho"]}


def _empty_result(pos):
    return {
        "pos_id": pos.pos_id, "strategy": pos.strategy,
        "option_type": pos.option_type, "style": pos.style,
        "K": pos.K, "t": pos.t, "sigma": pos.sigma, "quantity": pos.quantity,
        "entry_price": pos.entry_price, "market_price": pos.market_price,
        "prices": {}, "consensus": 0.0, "uncertainty": 0.0,
        "mispricing": 0.0, "pnl_unit": 0.0, "pnl_total": 0.0,
        "greeks": {k: 0.0 for k in ["delta","gamma","vega","theta","rho"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pricing du portefeuille entier
# ─────────────────────────────────────────────────────────────────────────────

def price_portfolio(positions: list[Position],
                    progress_callback=None) -> pd.DataFrame:
    """
    Price toutes les positions. Retourne un DataFrame résultat.
    progress_callback(i, n) appelé à chaque position pour barre de progression.
    """
    results = []
    n = len(positions)
    for i, pos in enumerate(positions):
        r = price_position(pos)
        results.append(r)
        if progress_callback:
            progress_callback(i + 1, n)
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Agrégation des grecs du portefeuille
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_greeks(results_df: pd.DataFrame) -> dict:
    """Somme les grecs pondérés par la quantité de toutes les positions."""
    agg = {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]}
    for _, row in results_df.iterrows():
        g = row.get("greeks", {}) or {}
        for greek in agg:
            agg[greek] += g.get(greek, 0.0)
    return {k: round(v, 4) for k, v in agg.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Recommandations de hedge
# ─────────────────────────────────────────────────────────────────────────────

def hedge_recommendations(portfolio_greeks: dict,
                           spot: float,
                           sigma: float,
                           r: float,
                           y: float = 0.013) -> list[dict]:
    """
    Génère des recommandations de hedge pour neutraliser les grecs principaux.

    Stratégie :
    - Delta : hedge avec le sous-jacent (SPY shares)
    - Gamma/Vega : hedge avec options ATM (straddle ou strangle)
    - Theta : informatif (pas de hedge direct possible sans risque)
    """
    recs = []
    g    = portfolio_greeks

    # ── Delta hedge ──────────────────────────────────────────────────────────
    delta = g.get("delta", 0)
    if abs(delta) > 0.01:
        direction = "Vendre" if delta > 0 else "Acheter"
        shares    = abs(round(delta * 100, 0))   # en actions (contrat = 100)
        recs.append({
            "greek"   : "Delta",
            "valeur"  : round(delta, 4),
            "action"  : f"{direction} {shares:.0f} actions SPY",
            "raison"  : f"Neutralise le delta ({delta:+.4f}) via sous-jacent",
            "urgence" : "🔴 Haute" if abs(delta) > 5 else "🟡 Moyenne",
        })

    # ── Gamma hedge ──────────────────────────────────────────────────────────
    gamma = g.get("gamma", 0)
    if abs(gamma) > 0.001:
        # Option ATM court terme pour contrer le gamma
        opt_atm  = EuropeanOption(spot, spot, 30/365, r, sigma, y)
        g_atm    = Greeks(opt_atm, "call").gamma()
        contracts_needed = -gamma / (g_atm * 100) if abs(g_atm) > 1e-6 else 0
        direction = "Acheter" if contracts_needed > 0 else "Vendre"
        recs.append({
            "greek"   : "Gamma",
            "valeur"  : round(gamma, 4),
            "action"  : f"{direction} {abs(contracts_needed):.1f} contrats ATM 30j",
            "raison"  : f"Gamma portefeuille ({gamma:+.4f}) → options court terme",
            "urgence" : "🔴 Haute" if abs(gamma) > 0.1 else "🟡 Moyenne",
        })

    # ── Vega hedge ───────────────────────────────────────────────────────────
    vega = g.get("vega", 0)
    if abs(vega) > 0.01:
        # ATM 3 mois pour contrer le vega
        opt_3m   = EuropeanOption(spot, spot, 0.25, r, sigma, y)
        v_3m     = Greeks(opt_3m, "call").vega()
        contracts_needed = -vega / (v_3m * 100) if abs(v_3m) > 1e-6 else 0
        direction = "Acheter" if contracts_needed > 0 else "Vendre"
        recs.append({
            "greek"   : "Vega",
            "valeur"  : round(vega, 4),
            "action"  : f"{direction} {abs(contracts_needed):.1f} contrats ATM 3M",
            "raison"  : f"Vega portefeuille ({vega:+.4f}) → options 3 mois",
            "urgence" : "🟡 Moyenne" if abs(vega) < 1 else "🔴 Haute",
        })

    # ── Theta (informatif) ───────────────────────────────────────────────────
    theta = g.get("theta", 0)
    recs.append({
        "greek"   : "Theta",
        "valeur"  : round(theta, 4),
        "action"  : "Aucune action directe",
        "raison"  : (f"Gain de {abs(theta):.4f}/jour (Short vol)" if theta > 0
                     else f"Coût de {abs(theta):.4f}/jour (Long vol)"),
        "urgence" : "🟢 Info",
    })

    # ── Rho (informatif) ─────────────────────────────────────────────────────
    rho = g.get("rho", 0)
    recs.append({
        "greek"  : "Rho",
        "valeur" : round(rho, 4),
        "action" : "Surveiller les décisions Fed",
        "raison" : f"Sensibilité taux : {rho:+.4f} pour +1% de taux",
        "urgence": "🟢 Info",
    })

    return recs