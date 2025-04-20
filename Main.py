from base.ClassMaturity import Maturity
from structuration.ClassMarket import Underlying, MarketData
from structuration.Produits.ProtectedCapital import AutocallNote, CapitalProtectedNote, CapitalProtectedNoteWithBarrier
from structuration.Produits.ProductBase import DecomposableProduct
from structuration.ClassPricing import PricingEngine
from base.ClassRate import Rate, RateModel, VasicekModel
from structuration.ClassVolatility import SSVIModel, VolatilityModel
from base.ClassOption import Option
from datetime import datetime
from typing import Dict, Optional, Literal
import numpy as np


### Données de Marché ###
spot = 100.0
strikes = np.linspace(80, 120, 9)  # Strikes de 80 à 120
maturities = np.array([0.5, 1.0, 2.0])  # En années

# Simuler des prix (à remplacer par des vraies données si tu en as)
market_prices = np.random.uniform(3, 15, size=(len(maturities), len(strikes)))

# Construire l’objet MarketData
market_data = MarketData(
    spot=spot,
    maturities=maturities,
    strikes=strikes,
    market_prices=market_prices,
    option_type='call', 
    risk_free_rate=0.03,
    dividend_yield=0.0
)

#### Modèle de vol ####
ssvi_model = SSVIModel(market_data=market_data)
params = ssvi_model.calibrate()  # Calibration en deux étapes

### Modèe de taux ###
vasicek_model = VasicekModel(a=0.1, b=0.03, sigma=0.01, r0=0.03)

#### Produit structuré :  note à capital protégé ###
note = CapitalProtectedNote(
    underlying_id="SPX",
    maturity=Maturity(5),
    nominal=1000,
    strike=100,
    participation_rate=1.5,
    capital_protection=0.9,
    rate_model=vasicek_model,
    spot_price=100.0,
    volatility_model=ssvi_model,
    dividend=0.0
)

#### Pricing ### 
engine = PricingEngine(
    spot_price=spot,
    domestic_rate=vasicek_model,
    volatility=ssvi_model,
    dividend=0.0,
    num_paths=10000,
    num_steps=252,
    seed=42
)

price = engine.price(note, method="decomposition")
print(f" Prix de la note à capital protégé : {price:.2f} €")

# Étape 7 : Grecques
greeks = engine.calculate_greeks(note, method="decomposition")
print(" Grecques :", greeks)