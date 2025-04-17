from ClassMarket import Underlying, MarketData
from ClassProduit import AutocallNote, CapitalProtectedNote, CapitalProtectedNoteWithBarrier
from ClassPricing import PricingEngine
from base.ClassRate import RateModel
from base.ClassVolatility import VolatilityModel
from base.ClassOption import Option
from datetime import datetime
from typing import Dict, Optional, Literal



market_data = MarketData(
    spot_price=100.0,
    volatility=0.20,
    risk_free_rate=0.02,
    dividend_yield=0.01
)

underlying = Underlying("EURO STOXX 50", market_data)

# Créer les différents produits
barrier_note = CapitalProtectedNoteWithBarrier(
    underlying=underlying,
    maturity=5.0,
    nominal=1000.0,
    strike=100.0,
    barrier=130.0,
    participation_rate=1.0,
    capital_protection=1.0,
    rebate=0.05
)
# Créer le sous-jacent
underlying = Underlying("EURO STOXX 50", market_data)

"""""
# A l'heure actuelle il faut réadapter le code de pricing engine
# Créer la note à capital protégé
capital_protected_note = CapitalProtectedNote(
    underlying=underlying,
    maturity=5.0,  # 5 ans
    nominal=1000.0,
    strike=100.0,
    participation_rate=1.0,  # 100% de participation
    capital_protection=1.0  # 100% du capital protégé
)

# Créer le moteur de pricing
pricing_engine = PricingEngine(capital_protected_note, nb_simulations=10000)
"""
# Calculer le prix
barrier_engine = PricingEngine(barrier_note, nb_simulations=10000)
price = barrier_engine.calculate_price()
print(f"Prix de la note avec barrière: {price:.2f}")


# Créer une option vanille
option = Option(
    underlying_id="AAPL",
    maturity=1,
    strike=100.0,
    option_type="call",
    nominal=1000.0
)

# Créer le moteur de pricing
pricing_engine = PricingEngine(
    spot_price=105.0,
    domestic_rate=RateModel(0.05),
    volatility=VolatilityModel(0.2),
    dividend=0.01,
    num_paths=10000,
    num_steps=252,
    seed=42
)

# Calculer le prix avec Black-Scholes
price_bs = pricing_engine.price(option, method="black_scholes")
print(f"Prix Black-Scholes: {price_bs:.2f}")

# Calculer les grecques avec Black-Scholes
greeks_bs = pricing_engine.calculate_greeks(option, method="black_scholes")
print(f"Grecques Black-Scholes: {greeks_bs}")

# Calculer le prix avec Monte Carlo
price_mc = pricing_engine.price(option, method="monte_carlo")
print(f"Prix Monte Carlo: {price_mc:.2f}")

# Calculer les grecques avec Monte Carlo
greeks_mc = pricing_engine.calculate_greeks(option, method="monte_carlo")
print(f"Grecques Monte Carlo: {greeks_mc}")