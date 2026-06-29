"""
interface/app.py
================
Interface Streamlit du pricer d'options.

Lancement :
    cd Pricer_Project
    streamlit run interface/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Imports internes ─────────────────────────────────────────────────────────
from Option_type.European import EuropeanOption
from Option_type.American import AmericanOption

from Method.BlackScholes  import BlackScholes
from Method.Binomial       import Binomial
from Method.Trinomial      import Trinomial
from Method.MonteCarlo     import MonteCarlo
from Method.finite_diff    import FiniteDifference

from Greeks.greeks         import Greeks
from visual.heatmap       import price_heatmap, greeks_heatmap, payoff_chart
from visual.PnL_heatmap   import compute_pnl_heatmap, pnl_heatmap_figure
from Strategies.strategies import (call_spread, put_spread, straddle,
                                    strangle, butterfly, iron_condor)
from PnL.pnl_calculator    import PnLCalculator
from PnL.Portfolio_pricer import aggregate_greeks, price_portfolio, price_position, _build_option, _compute_greeks, _empty_result, hedge_recommendations
from data.Market_data import MarketDataLoader
from data.Portfolio_generator import Position, PortfolioGenerator, positions_to_dataframe
from data.Database         import (save_calculation, list_calculations, get_calculation, delete_calculation, clear_all)

# ─────────────────────────────────────────────────────────────────────────────
# Config Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Options Pricer",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Styles CSS personnalisés ─────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --accent: #2563EB; --accent2: #F59E0B; }

    .stApp { background: #0F172A; color: #E2E8F0; }

    section[data-testid="stSidebar"] {
        background: #1E293B;
        border-right: 1px solid #334155;
    }

    h1, h2, h3 { color: #F1F5F9; font-family: 'Inter', sans-serif; }

    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;
                    letter-spacing: .06em; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #F1F5F9; }
    .metric-sub   { font-size: 0.85rem; color: #64748B; margin-top: 2px; }

    .greek-card   { background: #1E293B; border: 1px solid #334155;
                    border-radius: 10px; padding: 14px 18px; }
    .greek-name   { font-size: .75rem; color: #94A3B8; text-transform: uppercase; }
    .greek-val    { font-size: 1.4rem; font-weight: 600; color: #F1F5F9; }
    .greek-desc   { font-size: .72rem; color: #64748B; margin-top: 3px; }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94A3B8;
        border-bottom: 2px solid transparent;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #F1F5F9 !important;
        border-bottom: 2px solid #2563EB !important;
    }

    div[data-testid="metric-container"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
    }
    label[data-testid="stMetricLabel"] { color: #94A3B8 !important; }
    div[data-testid="stMetricValue"]   { color: #F1F5F9 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — paramètres communs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

    st.markdown("### Sous-jacent")
    S     = st.number_input("Prix sous-jacent (S)",  min_value=0.01, value=100.0, step=1.0)
    K     = st.number_input("Strike (K)",             min_value=0.01, value=100.0, step=1.0)

    st.markdown("### Marché")
    r     = st.number_input("Taux sans risque (r)",  min_value=0.0, max_value=0.5,  value=0.05, step=0.005, format="%.4f")
    y     = st.number_input("Dividende continu (y)", min_value=0.0, max_value=0.5,  value=0.0,  step=0.005, format="%.4f")
    sigma = st.number_input("Volatilité (σ)",        min_value=0.01, max_value=5.0, value=0.20, step=0.01,  format="%.3f")
    t     = st.number_input("Maturité (T, années)",  min_value=0.01, max_value=10.0,value=1.0,  step=0.05,  format="%.3f")

    st.markdown("### Option")
    style       = st.selectbox("Style",        ["Européenne", "Américaine"])
    option_type = st.selectbox("Type",         ["call", "put"])
    method      = st.selectbox("Méthode",      ["Black-Scholes", "Binomial", "Trinomial", "Monte Carlo", "Différences finies"])

    st.markdown("### Méthode numérique")
    if method in ["Binomial", "Trinomial"]:
        N_steps = st.slider("Nombre de pas (N)", 10, 500, 100, step=10)
    elif method == "Monte Carlo":
        n_sim    = st.slider("Simulations",   1000, 200000, 50000, step=1000)
        n_steps_ = st.slider("Pas par sim.", 50, 500, 252, step=50)
        mc_seed  = st.number_input("Seed", value=42, step=1)
    elif method == "Différences finies":
        fd_M = st.slider("Pas de temps (M)", 50, 500, 100, step=50)
        fd_N = st.slider("Pas d'espace (N)", 50, 500, 100, step=50)


# ─────────────────────────────────────────────────────────────────────────────
# Construction de l'option et du pricer
# ─────────────────────────────────────────────────────────────────────────────

if style == "Européenne":
    option = EuropeanOption(S, K, t, r, sigma, y)
else:
    option = AmericanOption(S, K, t, r, sigma, y)

_PRICERS = {
    "Black-Scholes"     : (BlackScholes,    {}),
    "Binomial"          : (Binomial,         {"N": N_steps if method == "Binomial" else 100}),
    "Trinomial"         : (Trinomial,        {"N": N_steps if method == "Trinomial" else 100}),
    "Monte Carlo"       : (MonteCarlo,       {"n_simulations": n_sim if method=="Monte Carlo" else 50000,
                                               "n_steps": n_steps_ if method=="Monte Carlo" else 252,
                                               "seed": int(mc_seed) if method=="Monte Carlo" else 42}),
    "Différences finies": (FiniteDifference, {"M": fd_M if method=="Différences finies" else 100,
                                               "N": fd_N if method=="Différences finies" else 100}),
}

pricing_fn, pricing_kw = _PRICERS[method]


def get_price():
    try:
        res = pricing_fn(option, option_type, **pricing_kw)
        if isinstance(res, dict):
            return res["price"], res.get("std_error"), res.get("ci_95")
        return float(res), None, None
    except ValueError as e:
        return None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📈 Options Pricer")
st.markdown(
    f"`{style}` · `{option_type.upper()}` · `{method}` · "
    f"S={S} K={K} σ={sigma:.0%} T={t}y r={r:.1%}"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Onglets principaux
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs(["💰 Pricing", "🔢 Grecs", "🗺️ Heatmaps", "📐 Stratégies", "🔥 P&L Heatmap", "🗄️ Historique DB", "📊 Portefeuille"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    price, se, ci = get_price()
    if price is None:
        st.error(f"Erreur : {ci}")
        st.stop()

    # KPIs principaux
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prix {option_type.upper()}</div>
            <div class="metric-value">{price:.4f}</div>
            <div class="metric-sub">{method}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        # Parité call-put (européen BS seulement)
        parity = None
        if not option.is_american:
            try:
                other_type = "put" if option_type == "call" else "call"
                other_res  = pricing_fn(option, other_type, **pricing_kw)
                other_price = other_res["price"] if isinstance(other_res, dict) else other_res
                parity = other_price
            except Exception:
                pass
        if parity is not None:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prix {other_type.upper()}</div>
                <div class="metric-value">{parity:.4f}</div>
                <div class="metric-sub">Parité put-call</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        if se is not None:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Erreur std (MC)</div>
                <div class="metric-value">{se:.4f}</div>
                <div class="metric-sub">IC 95% [{ci[0]:.4f}, {ci[1]:.4f}]</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Valeur intrinsèque
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            time_val  = price - intrinsic
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Valeur intrinsèque</div>
                <div class="metric-value">{intrinsic:.4f}</div>
                <div class="metric-sub">Valeur temps : {time_val:.4f}</div>
            </div>""", unsafe_allow_html=True)
    with c4:
        moneyness = S / K
        label = "ITM" if (moneyness > 1 and option_type == "call") or \
                         (moneyness < 1 and option_type == "put") else \
                "OTM" if (moneyness < 1 and option_type == "call") or \
                         (moneyness > 1 and option_type == "put") else "ATM"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Moneyness</div>
            <div class="metric-value">{moneyness:.3f}</div>
            <div class="metric-sub">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Comparaison toutes méthodes (européen uniquement)
    if not option.is_american:
        st.markdown("### Comparaison des méthodes")
        methods_to_compare = {
            "Black-Scholes"     : (BlackScholes,     {}),
            "Binomial (N=200)"  : (Binomial,          {"N": 200}),
            "Trinomial (N=100)" : (Trinomial,         {"N": 100}),
            "Monte Carlo"       : (MonteCarlo,        {"n_simulations": 50000, "seed": 42}),
            "Diff. Finies"      : (FiniteDifference,  {"M": 100, "N": 200}),
        }
        rows = []
        bs_ref = None
        for name, (fn, kw) in methods_to_compare.items():
            try:
                res = fn(option, option_type, **kw)
                p   = res["price"] if isinstance(res, dict) else float(res)
                se_ = res.get("std_error") if isinstance(res, dict) else None
                if bs_ref is None:
                    bs_ref = p
                rows.append({"Méthode": name, "Prix": f"{p:.5f}",
                             "Écart vs BS": f"{p - bs_ref:+.5f}",
                             "Std Err": f"{se_:.5f}" if se_ else "—"})
            except Exception as e:
                rows.append({"Méthode": name, "Prix": "—", "Écart vs BS": "—",
                             "Std Err": str(e)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Payoff à maturité
    st.markdown("### Payoff à maturité")
    S_arr = np.linspace(max(S * 0.5, 0.01), S * 1.5, 400)
    payoff_vals = (np.maximum(S_arr - K, 0) if option_type == "call"
                   else np.maximum(K - S_arr, 0))

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=S_arr, y=payoff_vals,
        mode="lines", name="Payoff brut",
        line=dict(color="#2563EB", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.10)"))
    fig_payoff.add_vline(x=S, line_dash="dash", line_color="#F59E0B",
                         annotation_text=f"S actuel = {S}")
    fig_payoff.add_vline(x=K, line_dash="dot", line_color="#94A3B8",
                         annotation_text=f"Strike = {K}")
    fig_payoff.update_layout(
        xaxis_title="Prix sous-jacent", yaxis_title="Payoff",
        height=350, paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font_color="#E2E8F0",
    )
    st.plotly_chart(fig_payoff, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — GRECS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Grecs de l'option")

    try:
        g_obj = Greeks(option, option_type, pricing_fn, **pricing_kw)
        all_g = g_obj.all()
    except Exception as e:
        st.error(f"Erreur calcul des grecs : {e}")
        st.stop()

    _GREEK_META = {
        "delta": ("Δ Delta",  "Sensibilité à S",    "≈ Proba ITM (call)"),
        "gamma": ("Γ Gamma",  "Convexité (Δ²/S²)",  "Risque de re-hedge"),
        "vega":  ("ν Vega",   "Sensibilité à σ",    "Pour 1% de vol"),
        "theta": ("Θ Theta",  "Décroissance temps", "Par jour calendaire"),
        "rho":   ("ρ Rho",    "Sensibilité à r",    "Pour 1% de taux"),
    }

    cols = st.columns(5)
    for i, (greek, (symbol, sens, note)) in enumerate(_GREEK_META.items()):
        val = all_g[greek]
        color = "#34D399" if val >= 0 else "#F87171"
        with cols[i]:
            st.markdown(f"""
            <div class="greek-card">
                <div class="greek-name">{symbol}</div>
                <div class="greek-val" style="color:{color}">{val:.5f}</div>
                <div class="greek-desc">{sens}</div>
                <div class="greek-desc">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Évolution des grecs vs S
    st.markdown("### Grecs en fonction de S")
    S_arr = np.linspace(max(S * 0.5, 0.01), S * 1.5, 100)
    selected_greek = st.selectbox("Choisir un grec", list(_GREEK_META.keys()),
                                   format_func=lambda x: _GREEK_META[x][0])

    greek_vals = []
    for s_val in S_arr:
        import copy as _copy
        opt_tmp = _copy.copy(option)
        opt_tmp.S = s_val
        try:
            g_tmp = Greeks(opt_tmp, option_type, pricing_fn, **pricing_kw)
            greek_vals.append(getattr(g_tmp, selected_greek)())
        except Exception:
            greek_vals.append(np.nan)

    fig_greek = go.Figure()
    fig_greek.add_trace(go.Scatter(
        x=S_arr, y=greek_vals, mode="lines",
        line=dict(color="#A78BFA", width=2.5), name=selected_greek,
    ))
    fig_greek.add_vline(x=S, line_dash="dash", line_color="#F59E0B",
                        annotation_text=f"S = {S}")
    fig_greek.update_layout(
        xaxis_title="Prix sous-jacent (S)",
        yaxis_title=_GREEK_META[selected_greek][0],
        height=350, paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font_color="#E2E8F0",
    )
    st.plotly_chart(fig_greek, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Heatmaps")

    # ── Contrôles axes & résolution ─────────────────────────────────────────
    hm_c1, hm_c2, hm_c3 = st.columns(3)
    with hm_c1:
        x_ax = st.selectbox("Axe X", ["S", "sigma", "t", "r"], index=0, key="hm_x")
    with hm_c2:
        y_ax = st.selectbox("Axe Y", ["sigma", "S", "t", "r"], index=0, key="hm_y")
    with hm_c3:
        hm_n = st.slider("Résolution", 6, 20, 10, key="hm_n")

    if x_ax == y_ax:
        st.warning("Choisissez deux axes différents.")
        st.stop()

    _HM_THEME = dict(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font_color="#E2E8F0")

    # ── Sous-onglets : Prix | Δ | Γ | ν | Θ | ρ | Tous les grecs ────────────
    sub_tabs = st.tabs(["💲 Prix", "Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho", "📋 Tous les grecs"])

    def _render_hm(fig):
        fig.update_layout(**_HM_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Prix
    with sub_tabs[0]:
        with st.spinner("Calcul du prix…"):
            try:
                fig = price_heatmap(option, pricing_fn, option_type,
                                    x_param=x_ax, y_param=y_ax,
                                    n_x=hm_n, n_y=hm_n, **pricing_kw)
                _render_hm(fig)
            except Exception as e:
                st.error(f"Erreur : {e}")

    # Un onglet par grec
    for idx, (greek, label) in enumerate([
        ("delta", "Delta — Δ = ∂V/∂S"),
        ("gamma", "Gamma — Γ = ∂²V/∂S²"),
        ("vega",  "Vega  — ν = ∂V/∂σ  (pour 1% de vol)"),
        ("theta", "Theta — Θ = ∂V/∂t  (par jour)"),
        ("rho",   "Rho   — ρ = ∂V/∂r  (pour 1% de taux)"),
    ]):
        with sub_tabs[idx + 1]:
            st.caption(label)
            with st.spinner(f"Calcul {greek}…"):
                try:
                    fig = greeks_heatmap(option, pricing_fn, option_type,
                                         greek=greek,
                                         x_param=x_ax, y_param=y_ax,
                                         n_x=hm_n, n_y=hm_n, **pricing_kw)
                    _render_hm(fig)
                except Exception as e:
                    st.error(f"Erreur {greek} : {e}")

    # Tous les grecs côte à côte (2 colonnes)
    with sub_tabs[6]:
        st.markdown("#### Les 5 grecs — vue d'ensemble")
        st.caption(f"Axes : **{x_ax}** × **{y_ax}** · Résolution {hm_n}×{hm_n}")

        greeks_list = [
            ("delta", "Δ Delta"),
            ("gamma", "Γ Gamma"),
            ("vega",  "ν Vega"),
            ("theta", "Θ Theta"),
            ("rho",   "ρ Rho"),
        ]

        with st.spinner("Calcul des 5 grecs…"):
            for i in range(0, len(greeks_list), 2):
                col_a, col_b = st.columns(2)
                for col, (greek, title) in zip(
                    [col_a, col_b], greeks_list[i:i+2]
                ):
                    with col:
                        try:
                            fig = greeks_heatmap(
                                option, pricing_fn, option_type,
                                greek=greek,
                                x_param=x_ax, y_param=y_ax,
                                n_x=hm_n, n_y=hm_n, **pricing_kw,
                            )
                            fig.update_layout(
                                **_HM_THEME,
                                title=title,
                                height=360,
                                margin=dict(l=40, r=10, t=40, b=40),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"{greek} : {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATÉGIES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Stratégies optionnelles")

    strat_name = st.selectbox("Stratégie", [
        "Call Spread", "Put Spread", "Straddle", "Strangle", "Butterfly", "Iron Condor"
    ])

    # ── Paramètres spécifiques ───────────────────────────────────────────────
    result = None
    try:
        if strat_name == "Call Spread":
            sc1, sc2 = st.columns(2)
            K1s = sc1.number_input("Strike bas (K1)", value=K * 0.95, step=1.0)
            K2s = sc2.number_input("Strike haut (K2)", value=K * 1.05, step=1.0)
            result = call_spread(option, K1s, K2s, pricing_fn, **pricing_kw)

        elif strat_name == "Put Spread":
            sc1, sc2 = st.columns(2)
            K1s = sc1.number_input("Strike bas (K1)", value=K * 0.95, step=1.0)
            K2s = sc2.number_input("Strike haut (K2)", value=K * 1.05, step=1.0)
            result = put_spread(option, K1s, K2s, pricing_fn, **pricing_kw)

        elif strat_name == "Straddle":
            Ks = st.number_input("Strike (K)", value=float(K), step=1.0)
            result = straddle(option, Ks, pricing_fn, **pricing_kw)

        elif strat_name == "Strangle":
            sc1, sc2 = st.columns(2)
            Kp = sc1.number_input("Strike Put (K_put)",   value=K * 0.90, step=1.0)
            Kc = sc2.number_input("Strike Call (K_call)", value=K * 1.10, step=1.0)
            result = strangle(option, Kp, Kc, pricing_fn, **pricing_kw)

        elif strat_name == "Butterfly":
            sc1, sc2, sc3 = st.columns(3)
            K1b = sc1.number_input("K1 (bas)",    value=K * 0.90, step=1.0)
            K2b = sc2.number_input("K2 (milieu)", value=float(K), step=1.0)
            K3b = sc3.number_input("K3 (haut)",   value=K * 1.10, step=1.0)
            fly_type = st.radio("Type", ["call", "put"], horizontal=True)
            result = butterfly(option, K1b, K2b, K3b, fly_type, pricing_fn, **pricing_kw)

        elif strat_name == "Iron Condor":
            sc1, sc2, sc3, sc4 = st.columns(4)
            K1c = sc1.number_input("K1 (Long Put)",   value=K * 0.85, step=1.0)
            K2c = sc2.number_input("K2 (Short Put)",  value=K * 0.95, step=1.0)
            K3c = sc3.number_input("K3 (Short Call)", value=K * 1.05, step=1.0)
            K4c = sc4.number_input("K4 (Long Call)",  value=K * 1.15, step=1.0)
            result = iron_condor(option, K1c, K2c, K3c, K4c, pricing_fn, **pricing_kw)

    except Exception as e:
        st.error(f"Erreur stratégie : {e}")

    if result is not None:
        # KPIs
        ks1, ks2, ks3, ks4 = st.columns(4)
        ks1.metric("Prime nette", f"{result.net_premium:+.4f}")
        ks2.metric("Profit max",  str(result.max_profit) if isinstance(result.max_profit, str)
                                   else f"{result.max_profit:.4f}")
        ks3.metric("Perte max",   str(result.max_loss) if isinstance(result.max_loss, str)
                                   else f"{result.max_loss:.4f}")
        ks4.metric("Breakeven(s)", "  /  ".join(f"{b:.2f}" for b in result.breakevens))

        # Tableau des jambes
        legs_df = pd.DataFrame([{
            "Jambe": leg.label,
            "Type": leg.option_type.upper(),
            "Strike": leg.strike,
            "Position": "Long" if leg.position > 0 else f"Short ×{abs(leg.position)}",
            "Prime": f"{leg.price:.4f}",
        } for leg in result.legs])
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

        # Graphique payoff
        fig_strat = payoff_chart(result)
        fig_strat.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font_color="#E2E8F0"
        )
        st.plotly_chart(fig_strat, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — P&L HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🔥 Heatmap de P&L — Stress test S × σ")
    st.caption(
        "Choque le prix sous-jacent et la volatilité autour de vos paramètres de base. "
        "**Vert = profit, Rouge = perte** par rapport au prix d'entrée."
    )

    # ── Prix d'entrée ────────────────────────────────────────────────────────
    col_ep1, col_ep2 = st.columns([2, 2])
    with col_ep1:
        entry_price_hm = st.number_input(
            "💰 Prix d'entrée de l'option (entry price)",
            min_value=0.001, value=float(round(get_price()[0] or 5.0, 4)),
            step=0.01, format="%.4f",
            help="La valeur actuelle de l'option est pré-remplie. Modifiez selon votre prix d'achat réel."
        )
    with col_ep2:
        hm_mode = st.radio("Afficher", ["P&L", "Prix brut"], horizontal=True,
                            help="P&L = prix choqué − entrée. Prix brut = valeur absolue.")

    # ── Amplitude des chocs ──────────────────────────────────────────────────
    st.markdown("#### Amplitude des chocs")
    cc1, cc2, cc3, cc4 = st.columns(4)
    S_shock_min  = cc1.number_input("Choc S min (%)", value=-40, step=5) / 100
    S_shock_max  = cc2.number_input("Choc S max (%)", value=+40, step=5) / 100
    sig_shock_min = cc3.number_input("Choc σ min (pts)", value=-20, step=5) / 100
    sig_shock_max = cc4.number_input("Choc σ max (pts)", value=+20, step=5) / 100

    cc5, cc6 = st.columns(2)
    n_S_pts   = cc5.slider("Nb de points S",     5, 21, 13, step=2)
    n_sig_pts = cc6.slider("Nb de points σ",     5, 21, 11, step=2)

    save_db = st.checkbox("💾 Sauvegarder ce calcul en base de données", value=True)

    if st.button("⚡ Calculer la heatmap", use_container_width=True, type="primary"):
        S_shocks     = np.linspace(S_shock_min,   S_shock_max,   n_S_pts)
        sigma_shocks = np.linspace(sig_shock_min, sig_shock_max, n_sig_pts)

        with st.spinner(f"Calcul de {n_S_pts * n_sig_pts} scénarios…"):
            try:
                result = compute_pnl_heatmap(
                    option       = option,
                    pricing_fn   = pricing_fn,
                    option_type  = option_type,
                    entry_price  = entry_price_hm,
                    style        = "american" if option.is_american else "european",
                    method_name  = method,
                    S_shocks     = S_shocks,
                    sigma_shocks = sigma_shocks,
                    save_to_db   = save_db,
                    **pricing_kw,
                )

                mode_key = "pnl" if hm_mode == "P&L" else "price"
                fig_pnl  = pnl_heatmap_figure(
                    result, option_type, S, sigma,
                    mode=mode_key, show_values=True,
                )
                fig_pnl.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                                       font_color="#E2E8F0")
                st.plotly_chart(fig_pnl, use_container_width=True)

                # ── Métriques clés ───────────────────────────────────────────
                grid = result["pnl_grid"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("P&L max (meilleur scénario)", f"{np.nanmax(grid):+.4f}")
                m2.metric("P&L min (pire scénario)",     f"{np.nanmin(grid):+.4f}")
                m3.metric("P&L médian",                  f"{np.nanmedian(grid):+.4f}")
                pct_profit = 100 * (grid > 0).sum() / grid.size
                m4.metric("% scénarios profitables",     f"{pct_profit:.1f}%")

                if save_db and result["calc_id"]:
                    st.success(f"✅ Calcul sauvegardé en base — `calc_id = {result['calc_id']}`")

                # ── Tableau des scénarios extrêmes ───────────────────────────
                with st.expander("📋 Voir tous les scénarios"):
                    rows = []
                    for r in result["shock_results"]:
                        pnl_val = r["option_price"] - entry_price_hm
                        rows.append({
                            "Choc S":    f"{r['shock_S']:+.0%}",
                            "Choc σ":    f"{r['shock_sigma']:+.2f}",
                            "S choqué":  f"{r['S_shocked']:.2f}",
                            "σ choquée": f"{r['sigma_shocked']:.3f}",
                            "Prix":      f"{r['option_price']:.4f}",
                            "P&L":       f"{pnl_val:+.4f}",
                        })
                    df_sc = pd.DataFrame(rows)
                    st.dataframe(
                        df_sc.style.applymap(
                            lambda v: "color:#34D399" if isinstance(v, str) and v.startswith("+") and v != "+0.00%"
                                      else ("color:#F87171" if isinstance(v, str) and v.startswith("-") else ""),
                            subset=["P&L"]
                        ),
                        use_container_width=True, hide_index=True
                    )

            except Exception as e:
                st.error(f"Erreur : {e}")
                import traceback; st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — HISTORIQUE DB
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🗄️ Historique des calculs")
    st.caption("Tous les calculs sauvegardés en base SQLite (`data/pricer.db`).")

    col_r, col_c = st.columns([3, 1])
    if col_r.button("🔄 Rafraîchir", use_container_width=True):
        st.rerun()
    if col_c.button("🗑️ Vider la base", use_container_width=True):
        clear_all()
        st.success("Base vidée.")
        st.rerun()

    history = list_calculations(limit=100)
    if not history:
        st.info("Aucun calcul en base. Lancez une heatmap P&L avec la sauvegarde activée.")
    else:
        df_hist = pd.DataFrame(history)
        cols_show = ["calc_id", "timestamp", "style", "option_type",
                     "S", "K", "t", "r", "sigma", "entry_price", "method"]
        cols_show = [c for c in cols_show if c in df_hist.columns]
        st.dataframe(df_hist[cols_show], use_container_width=True, hide_index=True)

        # ── Détail d'un calcul ───────────────────────────────────────────────
        st.markdown("#### 🔍 Détail d'un calcul")
        selected_id = st.selectbox(
            "Sélectionner un calc_id",
            options=[r["calc_id"] for r in history],
            format_func=lambda cid: f"{cid}  —  {next(r['timestamp'] for r in history if r['calc_id']==cid)}"
        )

        if selected_id:
            detail = get_calculation(selected_id)
            if detail:
                inp = detail["inputs"]
                st.markdown(f"**Paramètres :** S={inp['S']} K={inp['K']} T={inp['t']} "
                            f"r={inp['r']} σ={inp['sigma']} entry={inp['entry_price']}  "
                            f"méthode=`{inp['method']}`")

                outputs_df = pd.DataFrame(detail["outputs"])
                if not outputs_df.empty:
                    # Reconstruit la grille de P&L pour afficher la heatmap
                    pivot_pnl = outputs_df.pivot_table(
                        index="shock_sigma", columns="shock_S", values="pnl"
                    )
                    pivot_price = outputs_df.pivot_table(
                        index="shock_sigma", columns="shock_S", values="option_price"
                    )

                    tab_pnl, tab_price, tab_raw = st.tabs(["P&L", "Prix", "Données brutes"])

                    def _db_heatmap(pivot, title, zmid=None, colorscale="RdYlGn"):
                        x_lbl = [f"{v:+.0%}" for v in pivot.columns]
                        y_lbl = [f"{v:+.2f}" for v in pivot.index]
                        fig = go.Figure(go.Heatmap(
                            z=pivot.values, x=x_lbl, y=y_lbl,
                            colorscale=colorscale, zmid=zmid,
                            text=np.vectorize(lambda v: f"{v:.3f}")(pivot.values),
                            texttemplate="%{text}", textfont={"size": 9},
                        ))
                        fig.update_layout(
                            title=title, height=420,
                            xaxis_title="Choc S", yaxis_title="Choc σ",
                            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                            font_color="#E2E8F0",
                        )
                        return fig

                    with tab_pnl:
                        st.plotly_chart(
                            _db_heatmap(pivot_pnl, f"P&L — {selected_id}", zmid=0,
                                        colorscale=[[0,"#991B1B"],[0.5,"#F1F5F9"],[1,"#14532D"]]),
                            use_container_width=True
                        )
                    with tab_price:
                        st.plotly_chart(
                            _db_heatmap(pivot_price, f"Prix — {selected_id}", colorscale="Blues"),
                            use_container_width=True
                        )
                    with tab_raw:
                        st.dataframe(outputs_df, use_container_width=True, hide_index=True)

                    col_dl1, col_dl2 = st.columns(2)
                    col_dl1.download_button(
                        "⬇️ Télécharger outputs CSV",
                        data=outputs_df.to_csv(index=False),
                        file_name=f"outputs_{selected_id}.csv",
                        mime="text/csv",
                    )
                    col_dl2.download_button(
                        "⬇️ Télécharger inputs CSV",
                        data=pd.DataFrame([inp]).to_csv(index=False),
                        file_name=f"inputs_{selected_id}.csv",
                        mime="text/csv",
                    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 📊 Portefeuille")

    # ── Session state ────────────────────────────────────────────────────────
    for _k, _v in [("pf_positions", []), ("pf_results", None), ("pf_greeks", None),
                   ("pf_loader", None), ("pf_hedge_log", [])]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Ajouter des positions
    # ════════════════════════════════════════════════════════════════════════
    add_tab, gen_tab = st.tabs(["✍️ Ajout manuel", "🎲 Génération aléatoire SPY"])

    # ── Ajout manuel ─────────────────────────────────────────────────────────
    with add_tab:
        with st.form("pf_manual_form"):
            fc1, fc2 = st.columns(2)
            m_S     = fc1.number_input("S",    value=float(S),     step=1.0)
            m_K     = fc2.number_input("K",    value=float(K),     step=1.0)
            m_t     = fc1.number_input("T (ans)", value=float(t),  step=0.1)
            m_r     = fc2.number_input("r",    value=float(r),     step=0.005, format="%.4f")
            m_sig   = fc1.number_input("σ",    value=float(sigma), step=0.01,  format="%.3f")
            m_y     = fc2.number_input("y",    value=float(y),     step=0.005, format="%.4f")
            m_ot    = fc1.selectbox("Type",  ["call", "put"])
            m_style = fc2.selectbox("Style", ["european", "american",
                                              "asian_arithmetic", "asian_geometric",
                                              "barrier_down_out", "barrier_up_out",
                                              "barrier_down_in",  "lookback_floating",
                                              "lookback_fixed"])
            m_qty   = fc1.number_input("Quantité (>0 long, <0 short)", value=1, step=1)
            m_entry = fc2.number_input("Prix d'entrée (0 = auto)", value=0.0, step=0.01, format="%.4f")

            # Paramètres exotiques conditionnels
            m_barrier = None
            if "barrier" in m_style:
                m_barrier = fc1.number_input(
                    "Niveau barrière",
                    value=float(S * 0.85 if "down" in m_style else S * 1.15),
                    step=1.0
                )

            m_submit = st.form_submit_button("➕ Ajouter au portefeuille")

        if m_submit:
            from data.Portfolio_generator import Position
            import uuid
            # Mapper style → exotic_params
            style_map = {
                "european":          ("european", {}),
                "american":          ("american", {}),
                "asian_arithmetic":  ("asian",    {"averaging":"arithmetic","avg_type":"price"}),
                "asian_geometric":   ("asian",    {"averaging":"geometric", "avg_type":"price"}),
                "barrier_down_out":  ("barrier",  {"barrier": m_barrier, "barrier_type":"down-and-out"}),
                "barrier_up_out":    ("barrier",  {"barrier": m_barrier, "barrier_type":"up-and-out"}),
                "barrier_down_in":   ("barrier",  {"barrier": m_barrier, "barrier_type":"down-and-in"}),
                "lookback_floating": ("lookback", {"lookback_type":"floating"}),
                "lookback_fixed":    ("lookback", {"lookback_type":"fixed"}),
            }
            real_style, exotic_p = style_map.get(m_style, ("european", {}))

            pos = Position(
                pos_id=str(uuid.uuid4())[:6], strategy="manual",
                option_type=m_ot, style=real_style,
                S=m_S, K=m_K, t=m_t, r=m_r, sigma=m_sig, y=m_y,
                quantity=int(m_qty), entry_price=m_entry if m_entry > 0 else 0.0,
                market_price=0.0, expiration="manual",
                exotic_params=exotic_p,
            )
            st.session_state.pf_positions.append(pos)
            st.session_state.pf_results = None
            st.success(f"Position ajoutée : {pos.pos_id}")
            st.rerun()

    # ── Génération aléatoire ──────────────────────────────────────────────────
    with gen_tab:
        gc1, gc2 = st.columns(2)
        gn_pos      = gc1.slider("Nb de trades", 5, 40, 12)
        gn_seed     = gc2.number_input("Seed", value=42, step=1)
        gn_force_syn= gc2.checkbox("Données synthétiques", value=True)

        gw1, gw2, gw3, gw4 = st.columns(4)
        gw_van  = gw1.slider("Vanilles (%)",   0, 50, 35, step=5, key="gw_van")
        gw_spr  = gw2.slider("Spreads (%)",    0, 50, 20, step=5, key="gw_spr")
        gw_sta  = gw3.slider("Vol strats (%)", 0, 50, 30, step=5, key="gw_sta")
        gw_exo  = gw4.slider("Exotiques (%)",  0, 30, 10, step=5, key="gw_exo")

        if st.button("🎲 Générer & ajouter au portefeuille", type="primary", use_container_width=True):
            total_w = gw_van + gw_spr + gw_sta + gw_exo or 1
            van, spr, sta, exo = gw_van/total_w, gw_spr/total_w, gw_sta/total_w, gw_exo/total_w

            with st.spinner("Chargement données SPY…"):
                loader = MarketDataLoader("SPY")
                loader.get_options_chain(force_synthetic=gn_force_syn)
                st.session_state.pf_loader = loader

            custom_w = {
                "vanilla_long": van*.55, "vanilla_short": van*.45,
                "call_spread": spr*.5, "put_spread": spr*.5,
                "straddle": sta*.35, "strangle": sta*.25,
                "butterfly": sta*.25, "iron_condor": sta*.15,
                "exotic_asian": exo*.40, "exotic_barrier": exo*.40,
                "exotic_lookback": exo*.20,
            }

            with st.spinner(f"Génération de {gn_pos} trades…"):
                gen = PortfolioGenerator(loader, n_positions=gn_pos,
                                          seed=int(gn_seed), weights=custom_w)
                new_pos = gen.generate()
                st.session_state.pf_positions.extend(new_pos)
                st.session_state.pf_results = None

            st.success(f"✅ {len(new_pos)} positions ajoutées (source: {loader.source}, SPY={loader.spot_price:.1f})")
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Tableau du portefeuille + actions
    # ════════════════════════════════════════════════════════════════════════
    positions = st.session_state.pf_positions
    st.divider()

    if not positions:
        st.info("📭 Portefeuille vide — ajoutez des positions ci-dessus.")
    else:
        hdr1, hdr2, hdr3 = st.columns([4, 1, 1])
        hdr1.markdown(f"**{len(positions)} positions** dans le portefeuille")
        if hdr2.button("🗑️ Vider", use_container_width=True):
            st.session_state.pf_positions = []
            st.session_state.pf_results   = None
            st.session_state.pf_greeks    = None
            st.rerun()

        df_pos = positions_to_dataframe(positions)
        # Vue résumée par stratégie
        with st.expander("📋 Détail des positions", expanded=False):
            st.dataframe(
                df_pos[["pos_id","strategy","option_type","style","K","t","sigma","quantity","entry_price","expiration"]],
                use_container_width=True, hide_index=True,
            )

        # ── Pricing ──────────────────────────────────────────────────────────
        st.markdown("#### ⚡ Pricing multi-méthodes")
        if st.button("🔄 Pricer toutes les positions", use_container_width=True, type="primary"):
            prog = st.progress(0, "Pricing…")
            results_df = price_portfolio(positions,
                                          progress_callback=lambda i,n: prog.progress(i/n, f"{i}/{n}"))
            prog.empty()
            st.session_state.pf_results  = results_df
            st.session_state.pf_greeks   = aggregate_greeks(results_df)

            # ── Sauvegarde DB ─────────────────────────────────────────────────
            loader_ref = st.session_state.get("pf_loader")
            spot_ref   = loader_ref.spot_price if loader_ref else S
            sig_ref    = float(results_df["sigma"].mean()) if len(results_df) else sigma
            shock_ref  = [{"shock_S":0.0,"shock_sigma":0.0,
                           "S_shocked":spot_ref,"sigma_shocked":sig_ref,
                           "option_price":float(row["consensus"])}
                          for _,row in results_df.iterrows()]
            try:
                save_calculation(
                    style="portfolio", option_type="mixed",
                    S=spot_ref, K=0, t=0, r=r, sigma=sig_ref, y=y,
                    entry_price=None, method="multi",
                    shock_results=shock_ref,
                )
            except Exception:
                pass
            st.success("✅ Pricing terminé et sauvegardé en base.")
            st.rerun()

        # ════════════════════════════════════════════════════════════════════
        # SECTION 3 — Résultats & visualisations P&L
        # ════════════════════════════════════════════════════════════════════
        results_df = st.session_state.pf_results
        agg_g      = st.session_state.pf_greeks

        if results_df is not None and not results_df.empty:
            total_pnl = results_df["pnl_total"].sum()
            n_long    = (results_df["quantity"] > 0).sum()
            n_short   = (results_df["quantity"] < 0).sum()
            n_profit  = (results_df["pnl_total"] > 0).sum()

            # ── KPIs ─────────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            color_pnl = "#34D399" if total_pnl >= 0 else "#F87171"
            k1.markdown(f"""<div class="metric-card">
                <div class="metric-label">P&L Total</div>
                <div class="metric-value" style="color:{color_pnl}">{total_pnl:+.2f}</div>
                <div class="metric-sub">{n_profit}/{len(results_df)} positifs</div>
            </div>""", unsafe_allow_html=True)
            k2.markdown(f"""<div class="metric-card">
                <div class="metric-label">Positions</div>
                <div class="metric-value">{len(positions)}</div>
                <div class="metric-sub">Long {n_long} / Short {n_short}</div>
            </div>""", unsafe_allow_html=True)
            k3.markdown(f"""<div class="metric-card">
                <div class="metric-label">Mispricing moyen</div>
                <div class="metric-value">{results_df["mispricing"].abs().mean():.4f}</div>
                <div class="metric-sub">|consensus − marché|</div>
            </div>""", unsafe_allow_html=True)
            k4.markdown(f"""<div class="metric-card">
                <div class="metric-label">Incertitude σ</div>
                <div class="metric-value">{results_df["uncertainty"].mean():.4f}</div>
                <div class="metric-sub">Écart-type inter-méthodes</div>
            </div>""", unsafe_allow_html=True)
            k5.markdown(f"""<div class="metric-card">
                <div class="metric-label">Exposition nette Δ</div>
                <div class="metric-value">{agg_g.get("delta",0):+.3f}</div>
                <div class="metric-sub">Delta portefeuille</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Graphiques P&L ────────────────────────────────────────────────
            v1, v2 = st.columns(2)

            # Barres P&L par position
            with v1:
                st.markdown("##### P&L par position")
                sorted_res = results_df.sort_values("pnl_total")
                labels = sorted_res.apply(
                    lambda row: f"{row['pos_id']} ({row['strategy'][:4].upper()})", axis=1
                )
                fig_bar = go.Figure(go.Bar(
                    x=sorted_res["pnl_total"],
                    y=labels,
                    orientation="h",
                    marker_color=["#34D399" if v >= 0 else "#F87171"
                                  for v in sorted_res["pnl_total"]],
                    text=[f"{v:+.3f}" for v in sorted_res["pnl_total"]],
                    textposition="outside",
                ))
                fig_bar.add_vline(x=0, line_color="#64748B", line_width=1)
                fig_bar.update_layout(
                    height=max(300, len(results_df) * 28),
                    margin=dict(l=10, r=60, t=10, b=30),
                    paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                    font_color="#E2E8F0", xaxis_title="P&L",
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Donut par stratégie
            with v2:
                st.markdown("##### Répartition par stratégie")
                strat_pnl = results_df.groupby("strategy")["pnl_total"].sum().reset_index()
                strat_cnt = results_df.groupby("strategy")["pos_id"].count().reset_index()
                strat_merged = strat_pnl.merge(strat_cnt, on="strategy")
                strat_merged.columns = ["strategy","pnl","count"]

                colors = ["#34D399" if v >= 0 else "#F87171" for v in strat_merged["pnl"]]
                fig_donut = go.Figure(go.Pie(
                    labels=strat_merged["strategy"],
                    values=strat_merged["count"],
                    hole=0.55,
                    marker_colors=colors,
                    textinfo="label+percent",
                    textfont_size=10,
                ))
                fig_donut.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                    font_color="#E2E8F0",
                    annotations=[dict(text=f"{total_pnl:+.1f}", x=0.5, y=0.5,
                                      font_size=18, showarrow=False,
                                      font_color=color_pnl)],
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # Waterfall P&L cumulé
            st.markdown("##### P&L cumulé")
            sorted_by_strat = results_df.sort_values(["strategy","pnl_total"])
            cumulative = sorted_by_strat["pnl_total"].cumsum()
            fig_water = go.Figure(go.Scatter(
                x=list(range(len(cumulative))),
                y=cumulative.values,
                mode="lines+markers",
                line=dict(color="#2563EB", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.12)",
                marker=dict(
                    size=8,
                    color=["#34D399" if v>=0 else "#F87171" for v in cumulative],
                ),
                text=sorted_by_strat["pos_id"].values,
                hovertemplate="<b>%{text}</b><br>P&L cumulé : %{y:+.4f}<extra></extra>",
            ))
            fig_water.add_hline(y=0, line_color="#64748B", line_dash="dash", line_width=1)
            fig_water.update_layout(
                xaxis_title="Positions (triées par stratégie)",
                yaxis_title="P&L cumulé",
                height=280,
                margin=dict(l=50, r=10, t=10, b=40),
                paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                font_color="#E2E8F0",
            )
            st.plotly_chart(fig_water, use_container_width=True)

            # Consensus vs marché
            st.markdown("##### Consensus (multi-méthodes) vs Prix marché")
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=results_df["market_price"], y=results_df["consensus"],
                mode="markers",
                marker=dict(size=10, color=results_df["pnl_total"],
                            colorscale="RdYlGn", showscale=True,
                            colorbar=dict(title="P&L", thickness=12)),
                text=results_df.apply(
                    lambda r: f"{r['pos_id']} | {r['strategy']}<br>K={r['K']}", axis=1),
                hovertemplate="%{text}<br>Mkt: %{x:.4f} | Consensus: %{y:.4f}<extra></extra>",
            ))
            mn = min(results_df[["market_price","consensus"]].min())
            mx = max(results_df[["market_price","consensus"]].max())
            fig_scat.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                                line=dict(color="#F59E0B", dash="dash", width=1.5))
            fig_scat.update_layout(
                xaxis_title="Prix marché (mid)",
                yaxis_title="Prix consensus",
                height=350,
                margin=dict(l=50, r=10, t=10, b=50),
                paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                font_color="#E2E8F0",
            )
            st.plotly_chart(fig_scat, use_container_width=True)

            # ════════════════════════════════════════════════════════════════
            # SECTION 4 — Grecs + Hedging
            # ════════════════════════════════════════════════════════════════
            st.divider()
            st.markdown("### 🛡️ Grecs & Hedging")

            # Grecs portefeuille
            g_cols = st.columns(5)
            g_meta = {"delta":("Δ","#60A5FA"), "gamma":("Γ","#A78BFA"),
                      "vega":("ν","#34D399"),  "theta":("Θ","#F59E0B"),
                      "rho":("ρ","#94A3B8")}
            for col, (gk, (sym, col_color)) in zip(g_cols, g_meta.items()):
                val = agg_g.get(gk, 0)
                c   = "#34D399" if val >= 0 else "#F87171"
                col.markdown(f"""<div class="greek-card">
                    <div class="greek-name">{sym} {gk.upper()}</div>
                    <div class="greek-val" style="color:{c}">{val:+.4f}</div>
                    <div class="greek-desc">Agrégé portefeuille</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Recommandations
            loader_ref = st.session_state.get("pf_loader")
            spot_ref   = loader_ref.spot_price if loader_ref else S
            sig_ref    = float(results_df["sigma"].mean())
            recs       = hedge_recommendations(agg_g, spot=spot_ref, sigma=sig_ref, r=r)

            st.markdown("#### 📌 Recommandations de hedge")
            st.dataframe(
                pd.DataFrame(recs)[["greek","valeur","urgence","action","raison"]],
                use_container_width=True, hide_index=True,
            )

            # Stress test delta-gamma-vega
            st.markdown("#### 🔥 Stress Test portefeuille (Δ-Γ-ν)")
            n_stress = 11
            S_sh   = np.linspace(-0.30, 0.30, n_stress)
            sig_sh = np.linspace(-0.15, 0.15, n_stress)
            dp, gp, vp = agg_g.get("delta",0), agg_g.get("gamma",0), agg_g.get("vega",0)
            PnL_grid = np.array([[dp*(spot_ref*ds) + .5*gp*(spot_ref*ds)**2 + vp*dsig*100
                                   for ds in S_sh] for dsig in sig_sh])
            fig_stress = go.Figure(go.Heatmap(
                z=PnL_grid,
                x=[f"{v:+.0%}" for v in S_sh],
                y=[f"{v:+.2f}" for v in sig_sh],
                colorscale=[[0,"#991B1B"],[0.5,"#F1F5F9"],[1,"#14532D"]],
                zmid=0,
                text=np.vectorize(lambda v: f"{v:+.2f}")(PnL_grid),
                texttemplate="%{text}", textfont={"size":9},
                colorbar=dict(title="ΔP&L", thickness=12),
            ))
            fig_stress.update_layout(
                xaxis_title="Choc S (%)", yaxis_title="Choc σ (pts vol)",
                height=400, margin=dict(l=60,r=10,t=10,b=50),
                paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font_color="#E2E8F0",
            )
            st.plotly_chart(fig_stress, use_container_width=True)

            # Simulateur hedge interactif
            st.markdown("#### 🎮 Simulateur de hedge")
            hh1, hh2, hh3 = st.columns(3)
            h_type  = hh1.selectbox("Instrument", ["SPY shares","Call ATM","Put ATM","Straddle ATM"])
            h_qty_h = hh2.number_input("Quantité", value=1, step=1, key="h_qty_pf")
            h_mat   = hh3.selectbox("Maturité", ["30j","60j","90j"])
            h_t_v   = {"30j":30/365,"60j":60/365,"90j":90/365}[h_mat]

            if st.button("➕ Appliquer ce hedge", key="apply_hedge_pf"):
                opt_h = EuropeanOption(spot_ref, spot_ref, h_t_v, r, sig_ref, y)
                if h_type == "SPY shares":
                    hg = {"delta": h_qty_h/100, "gamma":0, "vega":0, "theta":0, "rho":0}
                elif h_type == "Call ATM":
                    hg = {k: v*h_qty_h for k,v in Greeks(opt_h,"call").all().items()}
                elif h_type == "Put ATM":
                    hg = {k: v*h_qty_h for k,v in Greeks(opt_h,"put").all().items()}
                else:
                    gc = Greeks(opt_h,"call").all()
                    gp = Greeks(opt_h,"put").all()
                    hg = {k:(gc[k]+gp[k])*h_qty_h for k in gc}
                st.session_state.pf_hedge_log.append(
                    {"instrument":h_type,"qty":h_qty_h,"mat":h_mat,"greeks":hg}
                )
                st.rerun()

            if st.session_state.pf_hedge_log:
                hedged = dict(agg_g)
                for hh in st.session_state.pf_hedge_log:
                    for k,v in hh["greeks"].items():
                        hedged[k] = hedged.get(k,0) + v

                ba_df = pd.DataFrame({
                    "Grec" : list(agg_g.keys()),
                    "Avant": [f"{v:+.4f}" for v in agg_g.values()],
                    "Après": [f"{hedged.get(k,0):+.4f}" for k in agg_g],
                    "Réduction %": [
                        f"{100*(1-abs(hedged.get(k,0))/max(abs(v),1e-9)):.1f}%"
                        if abs(v)>1e-6 else "—" for k,v in agg_g.items()
                    ],
                })
                st.dataframe(ba_df, use_container_width=True, hide_index=True)

                hedge_df = pd.DataFrame([{
                    "Instrument": h["instrument"], "Qté": h["qty"], "Maturité": h["mat"],
                    "Δ delta": f"{h['greeks'].get('delta',0):+.4f}",
                    "Δ vega":  f"{h['greeks'].get('vega',0):+.4f}",
                } for h in st.session_state.pf_hedge_log])
                st.dataframe(hedge_df, use_container_width=True, hide_index=True)

                if st.button("🗑️ Reset hedges", key="reset_pf_hedges"):
                    st.session_state.pf_hedge_log = []
                    st.rerun()


# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:.8rem'>"
    "Options Pricer · Black-Scholes · Binomial · Trinomial · Monte Carlo · Différences finies"
    "</div>",
    unsafe_allow_html=True
)