from ClassMarket import Underlying, MarketData
from ClassProduit import AutocallNote, CapitalProtectedNote
from ClassPricing import PricingEngine

# Créer les données de marché
market_data = MarketData(
    spot_price=100.0,
    volatility=0.20,
    risk_free_rate=0.02,
    dividend_yield=0.01
)

# Créer le sous-jacent
underlying = Underlying("EURO STOXX 50", market_data)

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

# Calculer le prix
price = pricing_engine.calculate_price()
print(f"Prix de la note à capital protégé: {price:.2f}")

# Calculer les Greeks
greeks = pricing_engine.calculate_greeks()
print(f"Delta: {greeks['delta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")