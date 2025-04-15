from ClassMarket import Underlying, MarketData
from ClassProduit import AutocallNote, CapitalProtectedNote, CapitalProtectedNoteWithBarrier
from ClassPricing import PricingEngine



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

