"""
visuals/heatmap.py
==================
Heatmaps du prix et des grecs en fonction de (S, σ) ou (S, t).
Retourne des figures Plotly pour intégration Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import copy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_grid(option, pricing_fn, option_type, x_vals, y_vals,
               x_param, y_param, **kw):
    """Évalue pricing_fn sur une grille (x_param × y_param)."""
    Z = np.zeros((len(y_vals), len(x_vals)))
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            opt = copy.copy(option)
            setattr(opt, x_param, xv)
            setattr(opt, y_param, yv)
            try:
                res = pricing_fn(opt, option_type, **kw)
                Z[i, j] = res["price"] if isinstance(res, dict) else res
            except Exception:
                Z[i, j] = np.nan
    return Z


def _eval_greek_grid(option, pricing_fn, option_type, greek_name,
                     x_vals, y_vals, x_param, y_param, **kw):
    """Évalue un grec sur une grille."""
    from Greeks.greeks import Greeks
    Z = np.zeros((len(y_vals), len(x_vals)))
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            opt = copy.copy(option)
            setattr(opt, x_param, xv)
            setattr(opt, y_param, yv)
            try:
                g   = Greeks(opt, option_type, pricing_fn, **kw)
                Z[i, j] = getattr(g, greek_name)()
            except Exception:
                Z[i, j] = np.nan
    return Z


def _heatmap_fig(Z, x_vals, y_vals, x_label, y_label, title,
                 colorscale="RdYlGn", fmt=".3f"):
    text = np.vectorize(lambda v: f"{v:{fmt}}")(Z)
    fig  = go.Figure(go.Heatmap(
        z           = Z,
        x           = np.round(x_vals, 4).tolist(),
        y           = np.round(y_vals, 4).tolist(),
        colorscale  = colorscale,
        text        = text,
        texttemplate= "%{text}",
        textfont    = {"size": 9},
        hoverongaps = False,
    ))
    fig.update_layout(
        title       = title,
        xaxis_title = x_label,
        yaxis_title = y_label,
        height      = 480,
        margin      = dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def price_heatmap(option, pricing_fn, option_type: str,
                  x_param: str = "S", y_param: str = "sigma",
                  x_range: tuple = None, y_range: tuple = None,
                  n_x: int = 12, n_y: int = 10, **kw):
    """
    Heatmap du prix de l'option en fonction de deux paramètres.

    Paramètres par défaut : x=S (±40 %), y=sigma (0.05 → 0.80)
    """
    S0    = option.S
    sig0  = option.sigma

    if x_range is None:
        if x_param == "S":
            x_range = (S0 * 0.60, S0 * 1.40)
        elif x_param == "sigma":
            x_range = (0.05, 0.80)
        else:
            x_range = (getattr(option, x_param) * 0.5,
                       getattr(option, x_param) * 1.5)

    if y_range is None:
        if y_param == "sigma":
            y_range = (0.05, 0.80)
        elif y_param == "S":
            y_range = (S0 * 0.60, S0 * 1.40)
        elif y_param == "t":
            y_range = (1/52, option.t)
        else:
            y_range = (getattr(option, y_param) * 0.5,
                       getattr(option, y_param) * 1.5)

    x_vals = np.linspace(*x_range, n_x)
    y_vals = np.linspace(*y_range, n_y)

    labels = {"S": "Prix sous-jacent (S)", "sigma": "Volatilité (σ)",
              "t": "Maturité (T)", "r": "Taux (r)", "K": "Strike (K)"}

    Z = _eval_grid(option, pricing_fn, option_type,
                   x_vals, y_vals, x_param, y_param, **kw)

    color = "Blues" if option_type == "call" else "Oranges"
    return _heatmap_fig(
        Z, x_vals, y_vals,
        labels.get(x_param, x_param), labels.get(y_param, y_param),
        f"Prix {option_type.upper()} — {x_param} × {y_param}",
        colorscale=color,
    )


def greeks_heatmap(option, pricing_fn, option_type: str,
                   greek: str = "delta",
                   x_param: str = "S", y_param: str = "sigma",
                   x_range: tuple = None, y_range: tuple = None,
                   n_x: int = 12, n_y: int = 10, **kw):
    """
    Heatmap d'un grec en fonction de deux paramètres.
    greek : 'delta' | 'gamma' | 'vega' | 'theta' | 'rho'
    """
    S0 = option.S

    if x_range is None:
        x_range = (S0 * 0.60, S0 * 1.40) if x_param == "S" else (0.05, 0.80)
    if y_range is None:
        y_range = (0.05, 0.80) if y_param == "sigma" else (S0 * 0.60, S0 * 1.40)

    x_vals = np.linspace(*x_range, n_x)
    y_vals = np.linspace(*y_range, n_y)

    labels = {"S": "Prix sous-jacent (S)", "sigma": "Volatilité (σ)",
              "t": "Maturité (T)", "r": "Taux (r)"}

    greek_colors = {
        "delta": "RdYlGn", "gamma": "Viridis",
        "vega": "Plasma",  "theta": "RdBu",   "rho": "Cividis",
    }

    Z = _eval_greek_grid(option, pricing_fn, option_type, greek,
                         x_vals, y_vals, x_param, y_param, **kw)

    return _heatmap_fig(
        Z, x_vals, y_vals,
        labels.get(x_param, x_param), labels.get(y_param, y_param),
        f"{greek.capitalize()} ({option_type.upper()}) — {x_param} × {y_param}",
        colorscale=greek_colors.get(greek, "RdYlGn"),
    )


def all_greeks_heatmaps(option, pricing_fn, option_type: str,
                        x_param: str = "S", y_param: str = "sigma",
                        n_x: int = 10, n_y: int = 8, **kw):
    """
    Retourne un dict {greek_name: fig} avec les 5 heatmaps des grecs.
    Pratique pour afficher plusieurs onglets dans Streamlit.
    """
    results = {}
    for g in ["delta", "gamma", "vega", "theta", "rho"]:
        results[g] = greeks_heatmap(
            option, pricing_fn, option_type, greek=g,
            x_param=x_param, y_param=y_param, n_x=n_x, n_y=n_y, **kw
        )
    return results


def payoff_chart(strategy_result, title: str = None) -> go.Figure:
    """
    Courbe de payoff à maturité pour une stratégie.
    strategy_result : StrategyResult (de strategies/strategies.py)
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = strategy_result.S_range,
        y    = strategy_result.payoff,
        mode = "lines",
        line = dict(color="#2563EB", width=2.5),
        name = "Payoff net",
        fill = "tozeroy",
        fillcolor = "rgba(37,99,235,0.10)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    for be in strategy_result.breakevens:
        fig.add_vline(x=be, line_dash="dot", line_color="#F59E0B",
                      annotation_text=f"BE {be:.2f}",
                      annotation_position="top right")

    fig.update_layout(
        title       = title or strategy_result.name,
        xaxis_title = "Prix sous-jacent à maturité (S)",
        yaxis_title = "P&L",
        height      = 400,
        margin      = dict(l=50, r=20, t=50, b=50),
        hovermode   = "x unified",
    )
    return fig
