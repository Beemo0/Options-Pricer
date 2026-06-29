"""
data/portfolio_generator.py
============================
Génère un portefeuille aléatoire et déséquilibré d'options
à partir de données de marché réelles (ou synthétiques).

Types de positions générées :
  - Vanilles long/short (call & put)
  - Straddles, strangles, spreads, butterfly, iron condor
  - Options exotiques (asiatique, barrière, lookback) via MC
"""

import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from data.Market_data import MarketDataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Structure d'une position du portefeuille
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    pos_id      : str
    strategy    : str           # 'vanilla', 'straddle', 'strangle', etc.
    option_type : str           # 'call' | 'put' | 'n/a'
    style       : str           # 'european' | 'american' | 'asian' | 'barrier' | 'lookback'
    S           : float
    K           : float         # strike principal (ou K1 pour spreads)
    K2          : Optional[float] = None   # strike secondaire si besoin
    K3          : Optional[float] = None
    K4          : Optional[float] = None
    t           : float = 1.0
    r           : float = 0.05
    sigma       : float = 0.20
    y           : float = 0.013
    quantity    : int   = 1     # >0 long, <0 short
    entry_price : float = 0.0
    market_price: float = 0.0  # prix mid du marché au moment de l'achat
    expiration  : str   = ""
    # Paramètres exotiques
    exotic_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.pos_id:
            import uuid
            self.pos_id = str(uuid.uuid4())[:6]

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "exotic_params"}
        d.update(self.exotic_params)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Générateur principal
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioGenerator:
    """
    Génère un portefeuille aléatoire et déséquilibré depuis la chaîne d'options.

    Paramètres
    ----------
    loader       : MarketDataLoader
    n_positions  : nombre total de positions à générer
    seed         : graine aléatoire pour reproductibilité
    weights      : dict optionnel pour pondérer les types de stratégies
    """

    STRATEGIES = [
        "vanilla_long", "vanilla_short",
        "straddle", "strangle",
        "call_spread", "put_spread",
        "butterfly", "iron_condor",
        "exotic_asian", "exotic_barrier", "exotic_lookback",
    ]

    DEFAULT_WEIGHTS = {
        "vanilla_long"  : 0.20,
        "vanilla_short" : 0.15,
        "straddle"      : 0.10,
        "strangle"      : 0.10,
        "call_spread"   : 0.10,
        "put_spread"    : 0.10,
        "butterfly"     : 0.08,
        "iron_condor"   : 0.07,
        "exotic_asian"  : 0.04,
        "exotic_barrier": 0.04,
        "exotic_lookback": 0.02,
    }

    def __init__(self, loader: MarketDataLoader,
                 n_positions: int = 20,
                 seed: int = 42,
                 weights: dict = None):
        self.loader     = loader
        self.n          = n_positions
        self.rng        = random.Random(seed)
        self.np_rng     = np.random.default_rng(seed)
        self.weights    = weights or self.DEFAULT_WEIGHTS
        self._df        = None

    # ── Entrée publique ───────────────────────────────────────────────────────

    def generate(self) -> list[Position]:
        """Génère le portefeuille. Retourne une liste de Position."""
        self._df = self.loader.get_options_chain()
        S        = self.loader.spot_price
        r        = self.loader.risk_free_rate
        y        = self.loader.div_yield

        strategy_names = list(self.weights.keys())
        strategy_probs = np.array(list(self.weights.values()))
        strategy_probs /= strategy_probs.sum()

        positions = []
        for i in range(self.n):
            strat = self.np_rng.choice(strategy_names, p=strategy_probs)
            try:
                pos_list = self._build(strat, S, r, y, i)
                positions.extend(pos_list)
            except Exception as e:
                continue  # skip si pas assez de strikes disponibles

        return positions

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _build(self, strat: str, S: float, r: float, y: float,
               idx: int) -> list[Position]:
        """Construit une ou plusieurs Position pour la stratégie donnée."""
        dispatch = {
            "vanilla_long"   : self._vanilla,
            "vanilla_short"  : self._vanilla,
            "straddle"       : self._straddle,
            "strangle"       : self._strangle,
            "call_spread"    : self._call_spread,
            "put_spread"     : self._put_spread,
            "butterfly"      : self._butterfly,
            "iron_condor"    : self._iron_condor,
            "exotic_asian"   : self._asian,
            "exotic_barrier" : self._barrier,
            "exotic_lookback": self._lookback,
        }
        fn = dispatch[strat]
        short = strat == "vanilla_short"
        return fn(S, r, y, idx, short=short) if "vanilla" in strat else fn(S, r, y, idx)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_row(self, option_type: str, moneyness_range=(0.85, 1.15)):
        """Choisit une option aléatoire dans la chaîne."""
        df = self._df
        mask = (
            (df["option_type"] == option_type) &
            (df["strike"] / self.loader.spot_price >= moneyness_range[0]) &
            (df["strike"] / self.loader.spot_price <= moneyness_range[1]) &
            (df["mid"] > 0.1)
        )
        sub = df[mask]
        if sub.empty:
            raise ValueError("Aucune option disponible dans cette plage")
        row = sub.sample(1, random_state=self.rng.randint(0, 9999)).iloc[0]
        return row

    def _pick_pair(self, option_type: str, K1_range, K2_range):
        """Choisit deux strikes distincts."""
        r1 = self._pick_row(option_type, K1_range)
        r2 = self._pick_row(option_type, K2_range)
        return r1, r2

    def _make_pos(self, strat, option_type, style, S, r, y,
                  row, quantity, idx, k2=None, k3=None, k4=None,
                  exotic_params=None):
        import uuid
        return Position(
            pos_id       = f"{strat[:4].upper()}-{idx:03d}-{str(uuid.uuid4())[:4]}",
            strategy     = strat,
            option_type  = option_type,
            style        = style,
            S            = S,
            K            = float(row["strike"]),
            K2           = k2, K3=k3, K4=k4,
            t            = float(row["t"]),
            r            = r,
            sigma        = float(row["iv"]),
            y            = y,
            quantity     = quantity,
            entry_price  = float(row["mid"]),
            market_price = float(row["mid"]),
            expiration   = str(row["expiration"]),
            exotic_params= exotic_params or {},
        )

    # ── Stratégies ────────────────────────────────────────────────────────────

    def _vanilla(self, S, r, y, idx, short=False):
        ot  = self.rng.choice(["call", "put"])
        row = self._pick_row(ot, (0.80, 1.20))
        qty = self.rng.choice([-5,-2,-1,1,2,5,10]) if not short else self.rng.choice([-10,-5,-2,-1])
        return [self._make_pos("vanilla", ot, "european", S, r, y, row, qty, idx)]

    def _straddle(self, S, r, y, idx):
        # ATM call + ATM put, même strike, même expiration
        atm = self.loader.spot_price
        df  = self._df
        # Trouver le strike le plus proche de l'ATM
        strikes = df["strike"].unique()
        atm_k   = min(strikes, key=lambda k: abs(k - atm))
        exp     = self.rng.choice(df["expiration"].unique())
        sub     = df[(df["strike"] == atm_k) & (df["expiration"] == exp)]

        call_row = sub[sub["option_type"] == "call"]
        put_row  = sub[sub["option_type"] == "put"]
        if call_row.empty or put_row.empty:
            raise ValueError("Straddle: pas de données")

        qty = self.rng.choice([1, 2, 5])
        return [
            self._make_pos("straddle", "call", "european", S, r, y,
                           call_row.iloc[0], qty, idx),
            self._make_pos("straddle", "put",  "european", S, r, y,
                           put_row.iloc[0],  qty, idx),
        ]

    def _strangle(self, S, r, y, idx):
        # OTM call + OTM put
        r_c = self._pick_row("call", (1.03, 1.20))
        r_p = self._pick_row("put",  (0.80, 0.97))
        # même expiration si possible
        qty = self.rng.choice([1, 2, 3])
        return [
            self._make_pos("strangle", "call", "european", S, r, y, r_c, qty, idx),
            self._make_pos("strangle", "put",  "european", S, r, y, r_p, qty, idx),
        ]

    def _call_spread(self, S, r, y, idx):
        r1, r2 = self._pick_pair("call", (0.90, 1.02), (1.02, 1.15))
        if r1["strike"] >= r2["strike"]:
            r1, r2 = r2, r1
        qty = self.rng.choice([1, 2, 5, 10])
        return [
            self._make_pos("call_spread", "call", "european", S, r, y, r1, +qty, idx),
            self._make_pos("call_spread", "call", "european", S, r, y, r2, -qty, idx),
        ]

    def _put_spread(self, S, r, y, idx):
        r1, r2 = self._pick_pair("put", (0.85, 0.98), (0.98, 1.10))
        if r1["strike"] > r2["strike"]:
            r1, r2 = r2, r1
        qty = self.rng.choice([1, 2, 5, 10])
        return [
            self._make_pos("put_spread", "put", "european", S, r, y, r2, +qty, idx),
            self._make_pos("put_spread", "put", "european", S, r, y, r1, -qty, idx),
        ]

    def _butterfly(self, S, r, y, idx):
        ot  = self.rng.choice(["call", "put"])
        r1  = self._pick_row(ot, (0.85, 0.95))
        r_atm = self._pick_row(ot, (0.97, 1.03))
        r3  = self._pick_row(ot, (1.05, 1.15))
        qty = self.rng.choice([1, 2])
        return [
            self._make_pos("butterfly", ot, "european", S, r, y, r1,    +qty, idx),
            self._make_pos("butterfly", ot, "european", S, r, y, r_atm, -2*qty, idx),
            self._make_pos("butterfly", ot, "european", S, r, y, r3,    +qty, idx),
        ]

    def _iron_condor(self, S, r, y, idx):
        r_lp = self._pick_row("put",  (0.80, 0.88))
        r_sp = self._pick_row("put",  (0.90, 0.98))
        r_sc = self._pick_row("call", (1.02, 1.10))
        r_lc = self._pick_row("call", (1.12, 1.20))
        qty  = self.rng.choice([1, 2])
        return [
            self._make_pos("iron_condor", "put",  "european", S, r, y, r_lp, +qty, idx),
            self._make_pos("iron_condor", "put",  "european", S, r, y, r_sp, -qty, idx),
            self._make_pos("iron_condor", "call", "european", S, r, y, r_sc, -qty, idx),
            self._make_pos("iron_condor", "call", "european", S, r, y, r_lc, +qty, idx),
        ]

    def _asian(self, S, r, y, idx):
        row = self._pick_row(self.rng.choice(["call","put"]), (0.90, 1.10))
        avg = self.rng.choice(["arithmetic", "geometric"])
        qty = self.rng.choice([-2, -1, 1, 2])
        return [self._make_pos("exotic_asian", row["option_type"], "asian",
                               S, r, y, row, qty, idx,
                               exotic_params={"averaging": avg, "avg_type": "price"})]

    def _barrier(self, S, r, y, idx):
        row  = self._pick_row(self.rng.choice(["call","put"]), (0.90, 1.10))
        btype = self.rng.choice(["down-and-out", "up-and-out", "down-and-in"])
        if "down" in btype:
            barrier = S * self.rng.uniform(0.75, 0.92)
        else:
            barrier = S * self.rng.uniform(1.08, 1.25)
        qty = self.rng.choice([-2, -1, 1, 2])
        return [self._make_pos("exotic_barrier", row["option_type"], "barrier",
                               S, r, y, row, qty, idx,
                               exotic_params={"barrier": round(barrier, 2),
                                              "barrier_type": btype})]

    def _lookback(self, S, r, y, idx):
        row   = self._pick_row(self.rng.choice(["call","put"]), (0.90, 1.10))
        ltype = self.rng.choice(["floating", "fixed"])
        qty   = self.rng.choice([-1, 1])
        return [self._make_pos("exotic_lookback", row["option_type"], "lookback",
                               S, r, y, row, qty, idx,
                               exotic_params={"lookback_type": ltype})]


# ─────────────────────────────────────────────────────────────────────────────
# Conversion en DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def positions_to_dataframe(positions: list[Position]) -> pd.DataFrame:
    rows = [p.to_dict() for p in positions]
    return pd.DataFrame(rows)