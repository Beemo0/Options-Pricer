"""
greeks/greeks.py
================
Calcul des 5 Grecs : Delta, Gamma, Vega, Theta, Rho.

- Formules analytiques Black-Scholes pour les options européennes.
- Approximation par différences finies (bump-and-reprice) pour
  les options américaines / exotiques (tout pricer compatible).

Usage
-----
from greeks.greeks import Greeks

g = Greeks(option, option_type="call", pricing_fn=BlackScholes)
print(g.delta())
print(g.all())          # dict complet
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _d1_d2(S, K, t, r, y, sigma):
    d1 = (np.log(S / K) + (r - y + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def _bs_price(S, K, t, r, y, sigma, option_type):
    """Prix Black-Scholes pur (sans objet option)."""
    if t <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, t, r, y, sigma)
    fwd  = S * np.exp(-y * t)
    disc = np.exp(-r * t)
    if option_type == "call":
        return fwd * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - fwd * norm.cdf(-d1)


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class Greeks:
    """
    Paramètres
    ----------
    option       : BaseOption (EuropeanOption, AmericanOption, …)
    option_type  : 'call' | 'put'
    pricing_fn   : fonction f(option, option_type, **kwargs) → float | dict
                   Si None, utilise Black-Scholes analytique.
                   Pour Monte Carlo, passer pricing_fn=MonteCarlo.
    kwargs       : transmis à pricing_fn (ex. N=200 pour Binomial)
    """

    def __init__(self, option, option_type: str, pricing_fn=None, **kwargs):
        self.option      = option
        self.option_type = option_type.lower()
        self._fn         = pricing_fn
        self._kw         = kwargs

        self._use_bs = (pricing_fn is None) and (not getattr(option, "is_american", False))

    # ── Appel interne au pricer ──────────────────────────────────────────────
    def _price(self, option=None):
        o = option or self.option
        if self._use_bs:
            return _bs_price(o.S, o.K, o.t, o.r, getattr(o, "y", 0.0),
                             o.sigma, self.option_type)
        result = self._fn(o, self.option_type, **self._kw)
        # Monte Carlo renvoie un dict
        return result["price"] if isinstance(result, dict) else result

    # ── Clonage d'option avec paramètre modifié ──────────────────────────────
    def _clone(self, **overrides):
        import copy
        o = copy.copy(self.option)
        for k, v in overrides.items():
            setattr(o, k, v)
        return o

    # ── bump size ────────────────────────────────────────────────────────────
    @staticmethod
    def _h(x, rel=1e-4):
        return max(abs(x) * rel, 1e-8)

    # ─────────────────────────────────────────────────────────────────────────
    # Delta  ∂V/∂S
    # ─────────────────────────────────────────────────────────────────────────
    def delta(self) -> float:
        """
        Sensibilité du prix à une variation du sous-jacent.
        BS analytique : Call → N(d1)·e^{-yT}, Put → (N(d1)-1)·e^{-yT}
        """
        o = self.option
        if self._use_bs:
            d1, _ = _d1_d2(o.S, o.K, o.t, o.r, getattr(o, "y", 0.0), o.sigma)
            factor = np.exp(-getattr(o, "y", 0.0) * o.t)
            if self.option_type == "call":
                return float(norm.cdf(d1) * factor)
            return float((norm.cdf(d1) - 1) * factor)

        # Différences centrées
        h = self._h(o.S)
        pu = self._price(self._clone(S=o.S + h))
        pd = self._price(self._clone(S=o.S - h))
        return float((pu - pd) / (2 * h))

    # ─────────────────────────────────────────────────────────────────────────
    # Gamma  ∂²V/∂S²
    # ─────────────────────────────────────────────────────────────────────────
    def gamma(self) -> float:
        """
        Convexité : variation du delta par unité de sous-jacent.
        BS analytique : N'(d1)·e^{-yT} / (S·σ·√T)
        """
        o = self.option
        if self._use_bs:
            d1, _ = _d1_d2(o.S, o.K, o.t, o.r, getattr(o, "y", 0.0), o.sigma)
            factor = np.exp(-getattr(o, "y", 0.0) * o.t)
            return float(norm.pdf(d1) * factor / (o.S * o.sigma * np.sqrt(o.t)))

        h = self._h(o.S)
        p0 = self._price()
        pu = self._price(self._clone(S=o.S + h))
        pd = self._price(self._clone(S=o.S - h))
        return float((pu - 2 * p0 + pd) / h ** 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Vega  ∂V/∂σ  (pour 1 pt de vol, i.e. /100)
    # ─────────────────────────────────────────────────────────────────────────
    def vega(self) -> float:
        """
        Sensibilité à la volatilité.
        BS analytique : S·e^{-yT}·N'(d1)·√T
        Retourné pour un mouvement de 1 % de volatilité (÷ 100).
        """
        o = self.option
        if self._use_bs:
            d1, _ = _d1_d2(o.S, o.K, o.t, o.r, getattr(o, "y", 0.0), o.sigma)
            factor = np.exp(-getattr(o, "y", 0.0) * o.t)
            raw = o.S * factor * norm.pdf(d1) * np.sqrt(o.t)
            return float(raw / 100)

        h = self._h(o.sigma, rel=1e-3)
        pu = self._price(self._clone(sigma=o.sigma + h))
        pd = self._price(self._clone(sigma=o.sigma - h))
        return float((pu - pd) / (2 * h) / 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Theta  ∂V/∂t  (par jour calendaire)
    # ─────────────────────────────────────────────────────────────────────────
    def theta(self) -> float:
        """
        Décroissance temporelle (exprimée par jour : ÷ 365).
        Valeur négative = perte de valeur avec le temps (normal pour acheteur).
        """
        o = self.option
        if self._use_bs:
            y = getattr(o, "y", 0.0)
            d1, d2 = _d1_d2(o.S, o.K, o.t, o.r, y, o.sigma)
            fwd  = o.S * np.exp(-y * o.t)
            disc = np.exp(-o.r * o.t)

            term1 = -(fwd * norm.pdf(d1) * o.sigma) / (2 * np.sqrt(o.t))
            if self.option_type == "call":
                term2 = -o.r * o.K * disc * norm.cdf(d2)
                term3 =  y   * fwd * norm.cdf(d1)
            else:
                term2 =  o.r * o.K * disc * norm.cdf(-d2)
                term3 = -y   * fwd * norm.cdf(-d1)
            return float((term1 + term2 + term3) / 365)

        # Différences finies forward (t ne peut pas être négatif)
        dt = 1 / 365
        if o.t <= dt:
            return 0.0
        p0 = self._price()
        pm = self._price(self._clone(t=o.t - dt))
        return float(pm - p0)          # variation pour 1 jour écoulé

    # ─────────────────────────────────────────────────────────────────────────
    # Rho  ∂V/∂r  (pour 1 bp = 0.01 %)
    # ─────────────────────────────────────────────────────────────────────────
    def rho(self) -> float:
        """
        Sensibilité au taux sans risque.
        BS analytique : K·T·e^{-rT}·N(±d2)
        Retourné pour un mouvement de 1 % (÷ 100).
        """
        o = self.option
        if self._use_bs:
            _, d2 = _d1_d2(o.S, o.K, o.t, o.r, getattr(o, "y", 0.0), o.sigma)
            disc = np.exp(-o.r * o.t)
            if self.option_type == "call":
                raw = o.K * o.t * disc * norm.cdf(d2)
            else:
                raw = -o.K * o.t * disc * norm.cdf(-d2)
            return float(raw / 100)

        h = 0.0001   # 1 bp
        pu = self._price(self._clone(r=o.r + h))
        pd = self._price(self._clone(r=o.r - h))
        return float((pu - pd) / (2 * h) / 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Tout d'un coup
    # ─────────────────────────────────────────────────────────────────────────
    def all(self) -> dict:
        """Retourne tous les grecs dans un dict."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }
