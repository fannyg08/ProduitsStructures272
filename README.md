# ProduitsStructures272
Projet de Produits Structurés M2 Quant 272 - Année 2024-2025

Ce projet implémente une bibliothèque Python pour la modélisation, la valorisation et l'analyse de produits structurés. Il comprend des modèles de volatilité, des moteurs de pricing, des classes pour les produits dérivés et les obligations.

## Structure du Projet

Le projet est organisé en deux packages principaux :

### 1. Base
Contient les classes fondamentales nécessaires à la modélisation financière :
- **ClassMaturity.py** : Modélisation des échéances
- **ClassOption.py** : Classes pour les options
- **ClassRate.py** : Modèles de taux d'intérêt

### 2. Structuration
Contient les classes pour la structuration et la valorisation des produits :
- **ClassDerivs.py** : Regroupe les différents types d'options
- **ClassFixedIncome.py** : Produits de taux
- **ClassMarket.py** : Données de marché
- **ClassPricing.py** : Moteurs de pricing
- **ClassProduit.py** : Classes de base pour les produits
- **ClassVolatility.py** : Modèles de volatilité

## Flux de Travail

#### 1. Données de Marché
Le point de départ est la classe `MarketData`, qui stocke les informations de marché nécessaires à la calibration :
- Prix du sous-jacent (spot)
- Taux sans risque
- Prix d'options pour différents strikes et maturités
- Taux de dividende

#### 2. Modèles de Volatilité
Les modèles de volatilité (comme `SSVIModel`) utilisent les données de marché pour leur calibration :
- Ils implémentent la classe abstraite `VolatilityModel`
- Ils permettent de calculer la volatilité implicite pour un strike et une maturité donnés

#### 3. Moteur de Pricing
Le `PricingEngine` utilise les modèles calibrés pour valoriser les produits :
- Il prend en entrée les modèles de taux et de volatilité
- Il supporte différentes méthodes de pricing (Black-Scholes, Monte Carlo)
- Il peut calculer les grecques

#### 4. Produits Financiers
Le système modélise différents types de produits :
- **Options** : Call, Put, options exotiques
- **Obligations** : Zéro-coupon, coupons fixes
- **Produits structurés** : Notes à capital protégé, barrières, autocall etc.

## Diagramme des Relations

```
MarketData <---- VolatilityModel (SSVIModel, etc.)
     ^               ^
     |               |
     |               |
Underlying        RateModel (Rate, etc.)
     ^               ^
     |               |
     v               v
  Product <---- PricingEngine
     |               |
     v               v
Option, ABCBond, ZeroCouponBond, Bond, CapitalProtectedNoteWithBarrier, etc.
```

## Utilisation

### Exemple de Valorisation d'une Option

```python
# Création des données de marché
market_data = MarketData(spot=100, risk_free_rate=0.05, strikes=np.array([...]), ...)

# Calibration d'un modèle de volatilité
vol_model = SSVIModel(market_data)

# Création d'un modèle de taux
rate_model = Rate(rate=0.05, rate_type="continuous")

# Configuration du moteur de pricing
pricing_engine = PricingEngine(
    spot_price=market_data.spot, 
    domestic_rate=rate_model, 
    volatility=vol_model, 
    dividend=market_data.dividend_yield
)

# Création d'une option
maturity = Maturity(maturity_in_years=1.0)
option = Option(
    spot_price=market_data.spot,
    strike_price=100,
    maturity=maturity,
    domestic_rate=rate_model,
    volatility=vol_model.get_implied_volatility(100, 1.0),
    option_type="call"
)

# Valorisation
price = pricing_engine.price(option, method="black_scholes")
greeks = pricing_engine.calculate_greeks(option, method="black_scholes")
```

### Exemple de Valorisation d'un Produit Structuré

```python
# Création d'un produit structuré
maturity = Maturity(maturity_in_years=3.0)
barrier_note = CapitalProtectedNoteWithBarrier(
    underlying_id="SPX",
    maturity=maturity,
    nominal=1000.0,
    strike=100.0,
    barrier=120.0,
    participation_rate=1.5,
    capital_protection=0.9,
    barrier_direction="up",
    barrier_type="ko"
)

# Valorisation par Monte Carlo
price = pricing_engine.price(barrier_note, method="monte_carlo")
```


## Prérequis

- Python 3.7+
- NumPy
- SciPy
- Matplotlib (pour les visualisations)
- pandas (pour le chargement des données)
