"""
pnl/pnl_calculator.py
======================
Calcul du P&L d'un portefeuille d'options.

Deux fichiers CSV :
  - data/input_data.csv  : paramètres d'entrée (snapshot à l'achat)
  - data/output_data.csv : résultats pricés + P&L

Usage minimal
-------------
from pnl.pnl_calculator import PnLCalculator
calc = PnLCalculator()
trade_id = calc.add_trade(S=100, K=100, t=0.5, r=0.05, sigma=0.2,
                           option_type="call", style="european",
                           quantity=10, entry_price=5.0)
calc.mark_to_market(pricing_fn=BlackScholes)
df = calc.pnl_report()
"""

import os
import uuid
import copy
from datetime import datetime

import numpy as np
import pandas as pd

# ── Chemins CSV ──────────────────────────────────────────────────────────────
_BASE  = os.path.dirname(os.path.abspath(__file__))
_DATA  = os.path.join(_BASE, "..", "data")
INPUT_CSV  = os.path.join(_DATA, "input_data.csv")
OUTPUT_CSV = os.path.join(_DATA, "output_data.csv")

# Colonnes attendues
_INPUT_COLS = [
    "trade_id", "timestamp", "style", "option_type",
    "S", "K", "t", "r", "sigma", "y",
    "quantity", "entry_price",
]
_OUTPUT_COLS = [
    "trade_id", "timestamp_mtm", "current_price",
    "delta", "gamma", "vega", "theta", "rho",
    "pnl_per_unit", "pnl_total",
]


# ─────────────────────────────────────────────────────────────────────────────

class PnLCalculator:
    """
    Gère un portefeuille d'options et calcule le P&L mark-to-market.
    """

    def __init__(self):
        os.makedirs(_DATA, exist_ok=True)
        self._input  = self._load(INPUT_CSV,  _INPUT_COLS)
        self._output = self._load(OUTPUT_CSV, _OUTPUT_COLS)

    # ── Persistance ──────────────────────────────────────────────────────────

    @staticmethod
    def _load(path, cols):
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=cols)

    def _save(self):
        self._input.to_csv(INPUT_CSV,   index=False)
        self._output.to_csv(OUTPUT_CSV, index=False)

    # ── Ajout d'un trade ─────────────────────────────────────────────────────

    def add_trade(self, S, K, t, r, sigma, option_type, style="european",
                  y=0.0, quantity=1, entry_price=None, pricing_fn=None, **kw):
        """
        Enregistre un nouveau trade dans input_data.csv.

        Paramètres
        ----------
        S, K, t, r, sigma, y : paramètres de l'option
        option_type           : 'call' | 'put'
        style                 : 'european' | 'american'
        quantity              : nombre de contrats (>0 long, <0 short)
        entry_price           : prime payée/reçue par unité
                                (si None → pricé automatiquement via pricing_fn)

        Retourne
        --------
        str : trade_id unique
        """
        if entry_price is None:
            if pricing_fn is None:
                raise ValueError("Fournissez entry_price ou pricing_fn.")
            opt = _make_option(style, S, K, t, r, sigma, y)
            res = pricing_fn(opt, option_type, **kw)
            entry_price = res["price"] if isinstance(res, dict) else res

        trade_id = str(uuid.uuid4())[:8]
        row = {
            "trade_id"    : trade_id,
            "timestamp"   : datetime.utcnow().isoformat(timespec="seconds"),
            "style"       : style,
            "option_type" : option_type,
            "S"           : S,  "K": K, "t": t, "r": r,
            "sigma"       : sigma, "y": y,
            "quantity"    : quantity,
            "entry_price" : entry_price,
        }
        self._input = pd.concat(
            [self._input, pd.DataFrame([row])], ignore_index=True
        )
        self._save()
        return trade_id

    # ── Mark-to-market ───────────────────────────────────────────────────────

    def mark_to_market(self, pricing_fn,
                       overrides: dict = None, **kw):
        """
        Reprice tous les trades avec les paramètres actuels (ou des overrides).

        overrides : dict {trade_id: {param: new_val}} pour changer S, sigma…
        Met à jour output_data.csv.
        """
        from Greeks.greeks import Greeks

        overrides = overrides or {}
        rows = []

        for _, tr in self._input.iterrows():
            tid = tr["trade_id"]
            # Paramètres courants (éventuellement modifiés)
            params = {
                "S": tr["S"], "K": tr["K"], "t": tr["t"],
                "r": tr["r"], "sigma": tr["sigma"], "y": tr["y"],
            }
            if tid in overrides:
                params.update(overrides[tid])

            opt = _make_option(tr["style"], **params)
            ot  = tr["option_type"]

            # Prix courant
            res   = pricing_fn(opt, ot, **kw)
            price = res["price"] if isinstance(res, dict) else res

            # Grecs
            g = Greeks(opt, ot, pricing_fn, **kw)
            all_g = g.all()

            pnl_unit  = (price - tr["entry_price"]) * np.sign(tr["quantity"])
            pnl_total = pnl_unit * abs(tr["quantity"])

            rows.append({
                "trade_id"      : tid,
                "timestamp_mtm" : datetime.utcnow().isoformat(timespec="seconds"),
                "current_price" : round(price,    6),
                **{k: round(v, 6) for k, v in all_g.items()},
                "pnl_per_unit"  : round(pnl_unit,  6),
                "pnl_total"     : round(pnl_total,  6),
            })

        self._output = pd.DataFrame(rows)
        self._save()
        return self._output

    # ── Rapport ──────────────────────────────────────────────────────────────

    def pnl_report(self) -> pd.DataFrame:
        """Fusionne input et output pour un rapport complet."""
        if self._output.empty:
            return pd.DataFrame()
        merged = pd.merge(self._input, self._output, on="trade_id", how="left")
        return merged

    def total_pnl(self) -> float:
        if "pnl_total" not in self._output.columns:
            return 0.0
        return float(self._output["pnl_total"].sum())

    def portfolio_greeks(self) -> dict:
        """Grecs agrégés du portefeuille (pondérés par quantity)."""
        if self._output.empty:
            return {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]}
        result = {}
        for greek in ["delta", "gamma", "vega", "theta", "rho"]:
            if greek in self._output.columns:
                # Multiply by signed quantity from input
                qtys = self._input.set_index("trade_id")["quantity"]
                vals = self._output.set_index("trade_id")[greek]
                agg  = (vals * qtys).sum()
                result[greek] = round(float(agg), 6)
        return result

    def clear(self):
        """Supprime tous les trades."""
        self._input  = pd.DataFrame(columns=_INPUT_COLS)
        self._output = pd.DataFrame(columns=_OUTPUT_COLS)
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# Helper interne
# ─────────────────────────────────────────────────────────────────────────────

def _make_option(style, S, K, t, r, sigma, y=0.0):
    if style == "american":
        from Option_type.American import AmericanOption
        return AmericanOption(S, K, t, r, sigma, y)
    from Option_type.European import EuropeanOption
    return EuropeanOption(S, K, t, r, sigma, y)
