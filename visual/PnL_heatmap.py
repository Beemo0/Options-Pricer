"""
visuals/pnl_heatmap.py
=======================
Génère la heatmap de P&L en choquant S et σ autour des valeurs de base.

Logique
-------
- Axe X  : chocs sur S      (ex : -40% → +40%)
- Axe Y  : chocs sur σ      (ex : -20 pts vol → +20 pts vol)
- Cellule : P&L = prix_choqué - entry_price
            Vert  = profit (P&L > 0)
            Rouge = perte  (P&L < 0)
- Chaque calcul est persisté en base via data.database.save_calculation()
"""

import numpy as np
import plotly.graph_objects as go
import copy

from data.Database import save_calculation


# ─────────────────────────────────────────────────────────────────────────────

def compute_pnl_heatmap(
    option,
    pricing_fn,
    option_type:   str,
    entry_price:   float,
    style:         str         = "european",
    method_name:   str         = "BlackScholes",
    S_shocks:      np.ndarray  = None,     # ex: np.linspace(-0.40, 0.40, 13)
    sigma_shocks:  np.ndarray  = None,     # ex: np.linspace(-0.20, 0.20, 11)
    save_to_db:    bool        = True,
    **pricing_kw,
) -> dict:
    """
    Calcule la grille de P&L et optionnellement la persiste en DB.

    Retourne
    --------
    {
      "call_id"       : str,
      "S_shocks"      : array,     # chocs % sur S
      "sigma_shocks"  : array,     # chocs absolus sur σ
      "price_grid"    : 2D array,  # prix optionnel (n_sigma x n_S)
      "pnl_grid"      : 2D array,  # P&L  (n_sigma x n_S)
      "shock_results" : list[dict] # données brutes → DB
    }
    """
    if S_shocks is None:
        S_shocks = np.linspace(-0.40, 0.40, 13)
    if sigma_shocks is None:
        sigma_shocks = np.linspace(-0.20, 0.20, 11)

    n_s   = len(S_shocks)
    n_sig = len(sigma_shocks)

    price_grid = np.zeros((n_sig, n_s))
    pnl_grid   = np.zeros((n_sig, n_s))
    shock_results = []

    for i, dsig in enumerate(sigma_shocks):
        for j, dS_pct in enumerate(S_shocks):
            opt = copy.copy(option)
            opt.S     = option.S * (1 + dS_pct)
            opt.sigma = max(option.sigma + dsig, 1e-4)   # σ ne peut pas être négatif

            try:
                res = pricing_fn(opt, option_type, **pricing_kw)
                price = res["price"] if isinstance(res, dict) else float(res)
            except Exception:
                price = np.nan

            pnl = price - entry_price if not np.isnan(price) else np.nan

            price_grid[i, j] = price
            pnl_grid[i, j]   = pnl

            shock_results.append({
                "shock_S"      : float(dS_pct),
                "shock_sigma"  : float(dsig),
                "S_shocked"    : float(opt.S),
                "sigma_shocked": float(opt.sigma),
                "option_price" : float(price) if not np.isnan(price) else 0.0,
            })

    # Persistance DB
    calc_id = None
    if save_to_db:
        calc_id = save_calculation(
            style=style, option_type=option_type,
            S=option.S, K=option.K, t=option.t,
            r=option.r, sigma=option.sigma, y=getattr(option, "y", 0.0),
            entry_price=entry_price, method=method_name,
            shock_results=shock_results,
        )

    return {
        "calc_id"      : calc_id,
        "S_shocks"     : S_shocks,
        "sigma_shocks" : sigma_shocks,
        "price_grid"   : price_grid,
        "pnl_grid"     : pnl_grid,
        "shock_results": shock_results,
    }


# ─────────────────────────────────────────────────────────────────────────────

def pnl_heatmap_figure(
    result:      dict,
    option_type: str,
    S_base:      float,
    sigma_base:  float,
    mode:        str = "pnl",    # "pnl" | "price"
    show_values: bool = True,
) -> go.Figure:
    """
    Construit la figure Plotly de la heatmap P&L ou Prix.

    Paramètres
    ----------
    result      : sortie de compute_pnl_heatmap()
    mode        : 'pnl' → affiche le P&L (vert/rouge)
                  'price' → affiche le prix brut (bleu dégradé)
    show_values : affiche les valeurs dans les cellules
    """
    S_shocks     = result["S_shocks"]
    sigma_shocks = result["sigma_shocks"]
    grid         = result["pnl_grid"] if mode == "pnl" else result["price_grid"]

    # Labels axes
    x_labels = [f"{v:+.0%}" for v in S_shocks]
    y_labels = [f"{v:+.2f}" for v in sigma_shocks]

    # Texte des cellules
    fmt = ".3f"
    text_vals = np.vectorize(lambda v: f"{v:{fmt}}" if not np.isnan(v) else "—")(grid)

    # Colorscale
    if mode == "pnl":
        colorscale = [
            [0.0,  "#991B1B"],   # rouge foncé  (perte max)
            [0.35, "#EF4444"],   # rouge
            [0.48, "#FCA5A5"],   # rose pâle
            [0.50, "#F1F5F9"],   # blanc neutre  (P&L = 0)
            [0.52, "#86EFAC"],   # vert pâle
            [0.65, "#22C55E"],   # vert
            [1.0,  "#14532D"],   # vert foncé   (profit max)
        ]
        title_suffix = "P&L"
        colorbar_title = "P&L"
    else:
        colorscale = "Blues"
        title_suffix = "Prix"
        colorbar_title = "Prix"

    fig = go.Figure(go.Heatmap(
        z            = grid,
        x            = x_labels,
        y            = y_labels,
        colorscale   = colorscale,
        zmid         = 0 if mode == "pnl" else None,
        text         = text_vals if show_values else None,
        texttemplate = "%{text}" if show_values else None,
        textfont     = {"size": 10, "color": "#1E293B"},
        hoverongaps  = False,
        hovertemplate = (
            "Choc S : %{x}<br>"
            "Choc σ : %{y}<br>"
            f"{title_suffix} : %{{z:.4f}}<extra></extra>"
        ),
        colorbar=dict(title=colorbar_title, thickness=14, len=0.85),
    ))

    # Marqueur position actuelle (choc = 0 sur les deux axes)
    try:
        x0 = x_labels.index("+0%")
        y0 = y_labels.index("+0.00")
        fig.add_shape(type="rect",
            x0=x0 - 0.5, x1=x0 + 0.5,
            y0=y0 - 0.5, y1=y0 + 0.5,
            line=dict(color="#F59E0B", width=2.5),
        )
    except ValueError:
        pass

    fig.update_layout(
        title       = f"Heatmap {title_suffix} — {option_type.upper()}  (S={S_base}, σ={sigma_base:.0%})",
        xaxis_title = "Choc sur S (%)",
        yaxis_title = "Choc sur σ (pts de vol)",
        height      = 520,
        margin      = dict(l=70, r=20, t=55, b=55),
        xaxis       = dict(side="bottom"),
    )
    return fig