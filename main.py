"""
main.py — point d'entrée CLI pour tester le pricer.
Pour l'interface graphique : streamlit run interface/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from Option_type.European import EuropeanOption
from Option_type.American import AmericanOption
from Option_type.Exotic import AsianOption, BarrierOption, LookbackOption


from Method.BlackScholes  import BlackScholes
from Method.Binomial       import Binomial
from Method.Trinomial      import Trinomial
from Method.MonteCarlo     import MonteCarlo
from Method.finite_diff    import FiniteDifference

from Greeks.greeks         import Greeks
from Strategies.strategies import call_spread, straddle, butterfly
from PnL.pnl_calculator    import PnLCalculator

# ── Paramètres ───────────────────────────────────────────────────────────────
S, K, t, r, sigma, y = 100, 100, 1.0, 0.05, 0.20, 0.0
N = 200

print("=" * 60)
print("OPTIONS PRICER — test de toutes les méthodes")
print("=" * 60)

euro = EuropeanOption(S, K, t, r, sigma, y)
amer = AmericanOption(S, K, t, r, sigma, y)

print("\n── Européen Call ──────────────────────────────────────────")
methods = {
    "Black-Scholes"     : (BlackScholes,    {}),
    "Binomial"          : (Binomial,         {"N": N}),
    "Trinomial"         : (Trinomial,        {"N": N // 2}),
    "Monte Carlo"       : (MonteCarlo,       {"n_simulations": 100_000, "seed": 42}),
    "Diff. Finies"      : (FiniteDifference, {"M": 200, "N": 200}),
}

for name, (fn, kw) in methods.items():
    res = fn(euro, "call", **kw)
    if isinstance(res, dict):
        print(f"  {name:<20} {res['price']:.5f}  (±{res['std_error']:.5f})")
    else:
        print(f"  {name:<20} {res:.5f}")

print("\n── Grecs (Black-Scholes, Call Européen) ───────────────────")
g = Greeks(euro, "call")
for k_name, v in g.all().items():
    print(f"  {k_name:<8} {v:+.6f}")

print("\n── Américain Put (Binomial vs Trinomial vs Diff. Finies) ──")
for name, fn, kw in [
    ("Binomial",    Binomial,         {"N": N}),
    ("Trinomial",   Trinomial,        {"N": N // 2}),
    ("Diff. Finies",FiniteDifference, {"M": 200, "N": 200}),
]:
    res = fn(amer, "put", **kw)
    print(f"  {name:<20} {res:.5f}")

print("\n── Stratégies ─────────────────────────────────────────────")
cs = call_spread(euro, 95, 105, BlackScholes)
print(cs.summary())

std = straddle(euro, K, BlackScholes)
print(std.summary())

fly = butterfly(euro, 90, 100, 110, "call", BlackScholes)
print(fly.summary())

print("\n── P&L Demo ───────────────────────────────────────────────")
calc = PnLCalculator()
calc.clear()
tid = calc.add_trade(S=S, K=K, t=t, r=r, sigma=sigma, option_type="call",
                     style="european", quantity=10, pricing_fn=BlackScholes)
print(f"  Trade ajouté : {tid}")
calc.mark_to_market(BlackScholes)
print(f"  P&L total    : {calc.total_pnl():+.4f}")
print(f"  Grecs portef.: {calc.portfolio_greeks()}")

print("\n✅ Tous les modules fonctionnent correctement.")
print("   Interface : streamlit run interface/app.py")
