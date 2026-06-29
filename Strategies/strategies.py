"""
strategies/strategies.py
=========================
Stratégies optionnelles classiques.

Chaque stratégie reçoit un pricer (ex. BlackScholes) et retourne :
  - le prix net de la stratégie
  - le détail de chaque jambe
  - le payoff à maturité sur une grille de S

Toutes les positions sont exprimées du point de vue de l'ACHETEUR (long).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """Une jambe de la stratégie."""
    option_type : str       # 'call' | 'put'
    strike      : float
    position    : int       # +1 long / -1 short
    price       : float = 0.0
    label       : str   = ""


@dataclass
class StrategyResult:
    name        : str
    legs        : List[Leg]
    net_premium : float                  # coût total de la stratégie (>0 = débit)
    breakevens  : List[float]
    max_profit  : float | str            # peut être "illimité"
    max_loss    : float | str
    S_range     : np.ndarray = field(repr=False, default=None)
    payoff      : np.ndarray = field(repr=False, default=None)

    def summary(self):
        lines = [
            f"{'─'*50}",
            f"Stratégie : {self.name}",
            f"Prime nette : {self.net_premium:+.4f}",
            f"Breakeven(s): {[round(b,4) for b in self.breakevens]}",
            f"Profit max  : {self.max_profit}",
            f"Perte max   : {self.max_loss}",
            f"{'─'*50}",
        ]
        for i, leg in enumerate(self.legs, 1):
            dir_ = "Long" if leg.position > 0 else "Short"
            lines.append(f"  Jambe {i} : {dir_} {leg.option_type.upper()} K={leg.strike}  prime={leg.price:.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper interne
# ─────────────────────────────────────────────────────────────────────────────

def _payoff_leg(S_arr, option_type, strike, position):
    if option_type == "call":
        intrinsic = np.maximum(S_arr - strike, 0)
    else:
        intrinsic = np.maximum(strike - S_arr, 0)
    return position * intrinsic


def _price_leg(option_template, strike, option_type, pricing_fn, **kw):
    """Clone l'option avec un strike différent et la price."""
    import copy
    opt = copy.copy(option_template)
    opt.K = strike
    result = pricing_fn(opt, option_type, **kw)
    return result["price"] if isinstance(result, dict) else result


def _build_S_range(S, width=2.0, n=500):
    low  = max(S * (1 - width / 2), 0.01)
    high = S * (1 + width / 2)
    return np.linspace(low, high, n)


# ─────────────────────────────────────────────────────────────────────────────
# Stratégies
# ─────────────────────────────────────────────────────────────────────────────

def call_spread(option, K1: float, K2: float, pricing_fn, **kw) -> StrategyResult:
    """
    Bull Call Spread : Long Call K1 + Short Call K2  (K1 < K2).
    Coût limité, gain limité, directionnel haussier.
    """
    assert K1 < K2, "K1 doit être < K2."
    p1 = _price_leg(option, K1, "call", pricing_fn, **kw)
    p2 = _price_leg(option, K2, "call", pricing_fn, **kw)
    net  = p1 - p2         # débit
    legs = [Leg("call", K1, +1, p1, "Long Call"), Leg("call", K2, -1, p2, "Short Call")]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, "call", K1, +1)
            + _payoff_leg(S_arr, "call", K2, -1) - net)

    return StrategyResult(
        name        = "Bull Call Spread",
        legs        = legs,
        net_premium = net,
        breakevens  = [K1 + net],
        max_profit  = round(K2 - K1 - net, 6),
        max_loss    = round(-net, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )


def put_spread(option, K1: float, K2: float, pricing_fn, **kw) -> StrategyResult:
    """
    Bear Put Spread : Long Put K2 + Short Put K1  (K1 < K2).
    Coût limité, gain limité, directionnel baissier.
    """
    assert K1 < K2, "K1 doit être < K2."
    p1 = _price_leg(option, K1, "put", pricing_fn, **kw)
    p2 = _price_leg(option, K2, "put", pricing_fn, **kw)
    net  = p2 - p1         # débit
    legs = [Leg("put", K2, +1, p2, "Long Put"), Leg("put", K1, -1, p1, "Short Put")]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, "put", K2, +1)
            + _payoff_leg(S_arr, "put", K1, -1) - net)

    return StrategyResult(
        name        = "Bear Put Spread",
        legs        = legs,
        net_premium = net,
        breakevens  = [K2 - net],
        max_profit  = round(K2 - K1 - net, 6),
        max_loss    = round(-net, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )


def straddle(option, K: float, pricing_fn, **kw) -> StrategyResult:
    """
    Straddle : Long Call K + Long Put K.
    Pari sur la volatilité (hausse OU baisse importante).
    """
    pc = _price_leg(option, K, "call", pricing_fn, **kw)
    pp = _price_leg(option, K, "put",  pricing_fn, **kw)
    net  = pc + pp
    legs = [Leg("call", K, +1, pc, "Long Call"), Leg("put", K, +1, pp, "Long Put")]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, "call", K, +1)
            + _payoff_leg(S_arr, "put",  K, +1) - net)

    return StrategyResult(
        name        = "Straddle",
        legs        = legs,
        net_premium = net,
        breakevens  = [K - net, K + net],
        max_profit  = "Illimité",
        max_loss    = round(-net, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )


def strangle(option, K_put: float, K_call: float, pricing_fn, **kw) -> StrategyResult:
    """
    Strangle : Long Put K_put + Long Call K_call  (K_put < K_call).
    Moins cher que le straddle, nécessite un mouvement plus important.
    """
    assert K_put < K_call, "K_put doit être < K_call."
    pp = _price_leg(option, K_put,  "put",  pricing_fn, **kw)
    pc = _price_leg(option, K_call, "call", pricing_fn, **kw)
    net  = pc + pp
    legs = [Leg("put",  K_put,  +1, pp, "Long Put"),
            Leg("call", K_call, +1, pc, "Long Call")]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, "put",  K_put,  +1)
            + _payoff_leg(S_arr, "call", K_call, +1) - net)

    return StrategyResult(
        name        = "Strangle",
        legs        = legs,
        net_premium = net,
        breakevens  = [K_put - net, K_call + net],
        max_profit  = "Illimité",
        max_loss    = round(-net, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )


def butterfly(option, K1: float, K2: float, K3: float,
              opt_type: str, pricing_fn, **kw) -> StrategyResult:
    """
    Long Butterfly : Long K1 + Short 2×K2 + Long K3  (K1 < K2 < K3, K2=(K1+K3)/2).
    Pari sur faible volatilité autour de K2.
    """
    assert K1 < K2 < K3, "K1 < K2 < K3 requis."
    p1 = _price_leg(option, K1, opt_type, pricing_fn, **kw)
    p2 = _price_leg(option, K2, opt_type, pricing_fn, **kw)
    p3 = _price_leg(option, K3, opt_type, pricing_fn, **kw)
    net  = p1 - 2 * p2 + p3
    legs = [
        Leg(opt_type, K1, +1, p1, f"Long {opt_type.upper()}"),
        Leg(opt_type, K2, -2, p2, f"Short 2× {opt_type.upper()}"),
        Leg(opt_type, K3, +1, p3, f"Long {opt_type.upper()}"),
    ]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, opt_type, K1, +1)
            + _payoff_leg(S_arr, opt_type, K2, -2)
            + _payoff_leg(S_arr, opt_type, K3, +1) - net)

    max_p = K2 - K1 - net

    return StrategyResult(
        name        = f"Long Butterfly ({opt_type.upper()})",
        legs        = legs,
        net_premium = net,
        breakevens  = [K1 + net, K3 - net],
        max_profit  = round(max_p, 6),
        max_loss    = round(-net, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )


def iron_condor(option, K1: float, K2: float, K3: float, K4: float,
                pricing_fn, **kw) -> StrategyResult:
    """
    Iron Condor : Short Put K2 + Long Put K1 + Short Call K3 + Long Call K4
    (K1 < K2 < K3 < K4).
    Pari sur faible volatilité dans un couloir [K2, K3].
    """
    assert K1 < K2 < K3 < K4, "K1 < K2 < K3 < K4 requis."
    p1 = _price_leg(option, K1, "put",  pricing_fn, **kw)
    p2 = _price_leg(option, K2, "put",  pricing_fn, **kw)
    p3 = _price_leg(option, K3, "call", pricing_fn, **kw)
    p4 = _price_leg(option, K4, "call", pricing_fn, **kw)

    credit = (p2 - p1) + (p3 - p4)   # crédit reçu (>0 idéalement)
    legs   = [
        Leg("put",  K1, +1, p1, "Long Put"),
        Leg("put",  K2, -1, p2, "Short Put"),
        Leg("call", K3, -1, p3, "Short Call"),
        Leg("call", K4, +1, p4, "Long Call"),
    ]

    S_arr  = _build_S_range(option.S)
    payoff = (_payoff_leg(S_arr, "put",  K1, +1)
            + _payoff_leg(S_arr, "put",  K2, -1)
            + _payoff_leg(S_arr, "call", K3, -1)
            + _payoff_leg(S_arr, "call", K4, +1) + credit)

    max_loss_val = min(K2 - K1, K4 - K3) - credit

    return StrategyResult(
        name        = "Iron Condor",
        legs        = legs,
        net_premium = -credit,             # on reçoit un crédit
        breakevens  = [K2 - credit, K3 + credit],
        max_profit  = round(credit, 6),
        max_loss    = round(-max_loss_val, 6),
        S_range     = S_arr,
        payoff      = payoff,
    )
