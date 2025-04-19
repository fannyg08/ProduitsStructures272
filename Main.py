from base.ClassMaturity import Maturity
from structuration.ClassMarket import Underlying, MarketData
from structuration.ClassProduit import AutocallNote, CapitalProtectedNote, CapitalProtectedNoteWithBarrier, DecomposableProduct
from structuration.ClassPricing import PricingEngine
from base.ClassRate import Rate, RateModel
from structuration.ClassVolatility import VolatilityModel
from base.ClassOption import Option
from datetime import datetime
from typing import Dict, Optional, Literal



# Création d'une note à capital protégé
capital_protected_note = CapitalProtectedNote(
    underlying_id="SPX",
    maturity=Maturity(maturity_in_years=5.0),
    nominal=1000.0,
    strike=100.0,
    participation_rate=1.5,
    capital_protection=0.9,
    rate_model=Rate(rate=0.03)
)

# Pricing par décomposition
price = PricingEngine.price(capital_protected_note, method="decomposition")

# Calcul des grecques par décomposition
greeks = PricingEngine.calculate_greeks(capital_protected_note, method="decomposition")
