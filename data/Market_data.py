"""
data/market_data.py
====================
Récupère la chaîne d'options SPY/SPX depuis Yahoo Finance.
Fallback automatique sur des données synthétiques réalistes si pas de connexion.

Usage
-----
from data.market_data import MarketDataLoader

loader = MarketDataLoader(ticker="SPY")
df     = loader.get_options_chain()          # DataFrame unifié calls + puts
spot   = loader.spot_price
r      = loader.risk_free_rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0525          # taux Fed Funds approximatif
SPY_SPOT_REF   = 592.0           # niveau SPY ~juin 2025
SPY_DIV_YIELD  = 0.013           # rendement dividende SPY ~1.3%


# ─────────────────────────────────────────────────────────────────────────────
# Loader principal
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataLoader:
    """
    Charge la chaîne d'options SPY/SPX.
    Essaie yfinance en priorité, bascule sur données synthétiques si échec.
    """

    def __init__(self, ticker: str = "SPY", n_expirations: int = 4):
        self.ticker        = ticker
        self.n_expirations = n_expirations
        self.spot_price    = None
        self.risk_free_rate = RISK_FREE_RATE
        self.div_yield      = SPY_DIV_YIELD
        self._source        = None
        self._chain_cache   = None

    # ── Entrée publique ───────────────────────────────────────────────────────

    def get_options_chain(self, force_synthetic: bool = False) -> pd.DataFrame:
        """
        Retourne un DataFrame avec toutes les options chargées :
        colonnes : ticker, expiration, t, option_type, strike, mid,
                   bid, ask, iv, open_interest, volume, delta_mkt
        """
        if self._chain_cache is not None:
            return self._chain_cache

        if not force_synthetic:
            df = self._try_yfinance()
            if df is not None:
                self._chain_cache = df
                self._source = "yfinance"
                return df

        df = self._synthetic_chain()
        self._chain_cache = df
        self._source = "synthetic"
        return df

    @property
    def source(self) -> str:
        return self._source or "not_loaded"

    # ── Yahoo Finance ─────────────────────────────────────────────────────────

    def _try_yfinance(self) -> pd.DataFrame | None:
        try:
            import yfinance as yf
        except ImportError:
            return None

        try:
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period="1d")
            if hist.empty:
                return None
            self.spot_price = float(hist["Close"].iloc[-1])

            exps = ticker_obj.options
            if not exps:
                return None

            all_rows = []
            today = datetime.today()

            for exp_str in exps[:self.n_expirations]:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                t = max((exp_date - today).days / 365.0, 1 / 365)

                chain = ticker_obj.option_chain(exp_str)

                for opt_type, df_raw in [("call", chain.calls), ("put", chain.puts)]:
                    for _, row in df_raw.iterrows():
                        bid = row.get("bid", 0) or 0
                        ask = row.get("ask", 0) or 0
                        mid = (bid + ask) / 2 if ask > 0 else row.get("lastPrice", 0)
                        all_rows.append({
                            "ticker"        : self.ticker,
                            "expiration"    : exp_str,
                            "t"             : round(t, 4),
                            "option_type"   : opt_type,
                            "strike"        : float(row["strike"]),
                            "mid"           : round(float(mid), 4),
                            "bid"           : round(float(bid), 4),
                            "ask"           : round(float(ask), 4),
                            "iv"            : round(float(row.get("impliedVolatility", 0.20)), 4),
                            "open_interest" : int(row.get("openInterest", 0) or 0),
                            "volume"        : int(row.get("volume", 0) or 0),
                            "delta_mkt"     : None,
                        })

            return pd.DataFrame(all_rows) if all_rows else None

        except Exception:
            return None

    # ── Données synthétiques réalistes ───────────────────────────────────────

    def _synthetic_chain(self) -> pd.DataFrame:
        """
        Génère une chaîne d'options SPY réaliste :
        - Surface de volatilité avec skew (puts OTM plus chers)
        - Structure par terme (vol plus haute à court terme)
        - Bid/ask spread réaliste (plus large OTM et court terme)
        """
        self.spot_price = SPY_SPOT_REF
        S = self.spot_price
        r = self.risk_free_rate
        y = self.div_yield

        today = datetime.today()
        # Expirations : hebdo, mensuel, mensuel+2, mensuel+6
        expirations = [
            (today + timedelta(weeks=1)).strftime("%Y-%m-%d"),
            (today + timedelta(weeks=4)).strftime("%Y-%m-%d"),
            (today + timedelta(weeks=13)).strftime("%Y-%m-%d"),
            (today + timedelta(weeks=26)).strftime("%Y-%m-%d"),
        ]

        rows = []
        for exp_str in expirations[:self.n_expirations]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            t = max((exp_date - today).days / 365.0, 1 / 365)

            # Strikes : ±25% autour du spot, pas de 5
            strikes = np.arange(
                round(S * 0.75 / 5) * 5,
                round(S * 1.25 / 5) * 5 + 5,
                5
            )

            for K in strikes:
                moneyness = K / S

                # ── Surface de vol avec skew réaliste ──────────────────────
                # ATM vol diminue avec la maturité (term structure inversée court terme)
                atm_vol = 0.18 - 0.04 * np.log(t + 0.1)
                atm_vol = max(atm_vol, 0.10)

                # Skew : puts OTM plus chers, appel OTM moins chers
                log_m = np.log(moneyness)
                skew_factor = -0.15 * log_m           # négatif pour calls OTM
                smile_factor = 0.12 * log_m ** 2      # sourire symétrique
                iv = atm_vol * (1 + skew_factor + smile_factor)
                iv = max(iv, 0.08)

                # ── Prix BS ─────────────────────────────────────────────────
                from Method.BlackScholes import BlackScholes
                from Option_type.European  import EuropeanOption
                opt = EuropeanOption(S, K, t, r, iv, y)

                call_price = BlackScholes(opt, "call")
                put_price  = BlackScholes(opt, "put")

                # ── Bid/ask spread ──────────────────────────────────────────
                # Plus large pour OTM profond et court terme
                depth = abs(log_m)
                base_spread_pct = 0.04 + 0.10 * depth + 0.05 / (t * 52 + 1)

                for opt_type, mid_price in [("call", call_price), ("put", put_price)]:
                    if mid_price < 0.01:
                        continue
                    spread = max(mid_price * base_spread_pct, 0.05)
                    bid    = max(mid_price - spread / 2, 0.01)
                    ask    = mid_price + spread / 2

                    # Volume et OI corrélés à la liquidité
                    liq_score = np.exp(-3 * depth) * np.exp(-t * 2)
                    oi = int(np.random.lognormal(
                        mean=np.log(max(liq_score * 5000, 10)), sigma=0.8
                    ))
                    vol = int(oi * np.random.uniform(0.05, 0.3))

                    rows.append({
                        "ticker"        : self.ticker,
                        "expiration"    : exp_str,
                        "t"             : round(t, 4),
                        "option_type"   : opt_type,
                        "strike"        : float(K),
                        "mid"           : round(float(mid_price), 4),
                        "bid"           : round(float(bid), 4),
                        "ask"           : round(float(ask), 4),
                        "iv"            : round(float(iv), 4),
                        "open_interest" : oi,
                        "volume"        : vol,
                        "delta_mkt"     : None,
                    })

        df = pd.DataFrame(rows)
        # Calcul du delta de marché (approximation BS)
        df["delta_mkt"] = df.apply(self._compute_delta, axis=1)
        return df

    def _compute_delta(self, row) -> float:
        try:
            from option_types.european_option import EuropeanOption
            from greeks.greeks import Greeks
            opt = EuropeanOption(
                S=self.spot_price, K=row["strike"],
                t=row["t"], r=self.risk_free_rate,
                sigma=row["iv"], y=self.div_yield
            )
            g = Greeks(opt, row["option_type"])
            return round(g.delta(), 4)
        except Exception:
            return None

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def get_atm_options(self, option_type: str = "call", n: int = 4) -> pd.DataFrame:
        """Options les plus proches du spot par expiration."""
        df = self.get_options_chain()
        df = df[df["option_type"] == option_type].copy()
        df["dist"] = abs(df["strike"] - self.spot_price)
        return (df.sort_values("dist")
                  .groupby("expiration")
                  .first()
                  .reset_index()
                  .head(n))

    def get_vol_surface(self) -> pd.DataFrame:
        """Surface de volatilité : index = strike, colonnes = expiration."""
        df = self.get_options_chain()
        df = df[df["option_type"] == "call"]
        return df.pivot_table(index="strike", columns="expiration",
                              values="iv", aggfunc="mean")