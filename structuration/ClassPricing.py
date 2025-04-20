from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from datetime import datetime
from base.ClassMaturity import Maturity, DayCountConvention
from base.ClassRate import RateModel
from base.ClassOption import Option
import tqdm
from .Produits.ProductBase import Product, DecomposableProduct
from .Produits.ProtectedCapital import CapitalProtectedNote, CapitalProtectedNoteTwinWin, CapitalProtectedNoteWithBarrier, CapitalProtectedNoteWithCoupon,AutocallNote
from .Produits.Participation import TrackerCertificate, OutperformanceCertificate,BonusCertificate
from .Produits.YieldEnhancement import ReverseConvertible,DiscountCertificate
from .ClassFixedIncome import Bond, ZeroCouponBond, ABCBond
from .ClassVolatility import VolatilityModel

# Types pour les produits structurés
BarrierDirection = Literal["up", "down"]
BarrierType = Literal["ko", "ki"]


class PricingEngine:
    """
    Moteur de pricing pour les produits financiers.
    Supporte le pricing par Black-Scholes, Monte Carlo ou par décomposition
    
    Explication des méthodes : 
    - Black Scholes : nous reprenons les formules de B&S. méthode qui marche pour le pricing d'option (?)
    - Décomposition : pour pricer un produit structuré "simple", on peut utiliser une méthode où on commence par calculer
    le prix d'un ZC/d'une obligation, puis le prix de l'option associée. L'addition des deux nous donne le prix du structuré
    - Monte Carlo : obligatoire pour les produits plus complexes, avec des options barrières etc. 

    La façon dont les méthodes sont organisées est la suivante : 
    Pour chaque "méthode" de pricing, on a une méthode de pricing et une méthode de calcul des greeks. 
    Ensuite on a price() et calculate_greeks() qui sont les méthodes que nous appelerons en pratique dans le code, car
    ce sont les méthodes générales. 
    """
    def __init__(
        self,
        spot_price: float,
        domestic_rate: RateModel,
        volatility: VolatilityModel,
        dividend: Optional[float] = None,
        foreign_rate: Optional[RateModel] = None,
        num_paths: int = 10000,
        num_steps: int = 252,
        seed: Optional[int] = None
    ):
        """
        Initialisation du moteur de pricing:
            spot_price (float): Prix actuel du sous-jacent
            domestic_rate (RateModel): Modèle de taux domestique
            volatility (VolatilityModel): Modèle de volatilité
            dividend (float, optional): Taux de dividende continu du sous-jacent
            foreign_rate (RateModel, optional): Taux étranger pour options de change
            num_paths (int): Nombre de trajectoires pour Monte Carlo
            num_steps (int): Nombre de pas temporels pour la simulation
            seed (int, optional): Graine pour la reproductibilité des simulations
        """
        self._spot_price = spot_price
        self._domestic_rate = domestic_rate
        self._volatility = volatility
        self._dividend = dividend if dividend is not None else 0.0
        self._foreign_rate = foreign_rate
        self._num_paths = num_paths
        self._num_steps = num_steps
        
        # Initialiser le générateur aléatoire si une graine est fournie
        if seed is not None:
            np.random.seed(seed)
    
    def _black_scholes_d1(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        return (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def _black_scholes_d2(self, d1: float, sigma: float, T: float) -> float:
        return d1 - sigma * np.sqrt(T)
    
    def _norm_cdf(self, x: float) -> float:
        """
        Fonction de répartition de la loi normale.
        """
        return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0
    
    def price_black_scholes(self, option: Option) -> float:
        """
        Calcule le prix d'une option avec la formule de Black-Scholes.
        """
        # Paramètres
        S = self._spot_price
        K = option.strike
        T = option.maturity.maturity_in_years
        r = -np.log(self._domestic_rate.discount_factor(T)) / T
        q = self._dividend
        if self._foreign_rate is not None:
            q = -np.log(self._foreign_rate.discount_factor(T)) / T
        
        
        # Volatilité pour ce strike et cette maturité
        sigma = self._volatility.get_implied_volatility(K, T)
        
        # Calcul des paramètres d1 et d2
        d1 = self._black_scholes_d1(S, K, T, r, q, sigma)
        d2 = self._black_scholes_d2(d1, sigma, T)
        
        # Calcul du prix selon le type d'option
        if option.option_type == "call":
            price = S * np.exp(-q * T) * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * np.exp(-q * T) * self._norm_cdf(-d1)
        
        return price * option.nominal / K
    
    def calculate_greeks_black_scholes(self, option: Option) -> Dict[str, float]:
        """
        Calcule les grecques d'une option avec les formules de Black-Scholes.
        """
        # Paramètres
        S = self._spot_price
        K = option.strike
        T = option.maturity.maturity_in_years
        r = -np.log(self._domestic_rate.discount_factor(T)) / T
        q = self._dividend
        if self._foreign_rate is not None:
            q = -np.log(self._foreign_rate.discount_factor(T)) / T
        
        
        # Volatilité pour ce strike et cette maturité
        sigma = self._volatility.get_implied_volatility(K, T)
        
        # Calcul des paramètres d1 et d2
        d1 = self._black_scholes_d1(S, K, T, r, q, sigma)
        d2 = self._black_scholes_d2(d1, sigma, T)
        
        # Constantes utiles
        sqrt_T = np.sqrt(T)
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        norm_d1 = self._norm_cdf(d1)
        norm_d2 = self._norm_cdf(d2)
        norm_minus_d1 = self._norm_cdf(-d1)
        norm_minus_d2 = self._norm_cdf(-d2)
        
        # Calculer la densité de la loi normale en d1
        norm_pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        
        # Initialiser les grecques
        greeks = {}
        
        # Delta
        if option.option_type == "call":
            greeks["delta"] = exp_qt * norm_d1
        else:  # put
            greeks["delta"] = exp_qt * (norm_d1 - 1)
        
        # Gamma (identique pour call et put)
        greeks["gamma"] = exp_qt * norm_pdf_d1 / (S * sigma * sqrt_T)
        
        # Vega (identique pour call et put, en pourcentage)
        greeks["vega"] = S * exp_qt * norm_pdf_d1 * sqrt_T / 100
        
        # Theta (par an)
        if option.option_type == "call":
            greeks["theta"] = -S * sigma * exp_qt * norm_pdf_d1 / (2 * sqrt_T) - r * K * exp_rt * norm_d2 + q * S * exp_qt * norm_d1
        else:  # put
            greeks["theta"] = -S * sigma * exp_qt * norm_pdf_d1 / (2 * sqrt_T) + r * K * exp_rt * norm_minus_d2 - q * S * exp_qt * norm_minus_d1
        
        # Rho (pour 1% de variation)
        if option.option_type == "call":
            greeks["rho"] = K * T * exp_rt * norm_d2 / 100
        else:  # put
            greeks["rho"] = -K * T * exp_rt * norm_minus_d2 / 100
        
        # Ajuster par le nominal
        for greek in greeks:
            greeks[greek] *= option.nominal / K
        
        return greeks
    
    def simulate_paths(self, maturity: Maturity, strike_price: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère des trajectoires simulées pour le sous-jacent avec un modèle log-normal.
        Returns:Tuple[ndarray, ndarray]: (paths, time_grid) - les chemins simulés et la grille temporelle
        """
        # Créer une grille temporelle
        time_grid = np.linspace(0, maturity.maturity_in_years, self._num_steps + 1)
        dt = maturity.maturity_in_years / self._num_steps
        
        # Initialiser le tableau des chemins
        paths = np.zeros((self._num_paths, self._num_steps + 1))
        paths[:, 0] = self._spot_price
        
        # Paramètres du modèle log-normal
        T = maturity.maturity_in_years
        r = -np.log(self._domestic_rate.discount_factor(T)) / T
        q = self._dividend
        if self._foreign_rate is not None:
            q = -np.log(self._foreign_rate.discount_factor(T)) / T
        
        vol = self._volatility.get_implied_volatility(strike_price, T)
        
        # Terme de dérive
        drift = (r - q - 0.5 * vol**2) * dt
        vol_sqrt_dt = vol * np.sqrt(dt)
        
        # Générer les chemins
        for i in tqdm(range(1, self._num_steps + 1), desc="Simulation Monte Carlo", leave=False):
            z = np.random.standard_normal(self._num_paths)
            paths[:, i] = paths[:, i-1] * np.exp(drift + vol_sqrt_dt * z)
        
        return paths, time_grid
    
    def price_monte_carlo(self, product: Product) -> float:
        """
        Calcule le prix d'un produit par simulation Monte Carlo.
        """
        # Simuler les trajectoires pour ce produit
        strike = getattr(product, "strike", self._spot_price)
        paths, time_grid = self.simulate_paths(product.maturity, strike)
        
        # Calculer les payoffs
        result = product.payoff(paths, time_grid)
        
        # Traiter les résultats selon leur format (payoffs à date unique ou multiple)
        if isinstance(result, tuple) and len(result) == 2:
            payoffs, payment_times = result
            
            # Vecteur de prix présents
            present_values = np.zeros_like(payoffs)
            
            # Actualiser chaque payoff selon sa date de paiement
            for i, t in enumerate(payment_times):
                discount_factor = self._domestic_rate.discount_factor(t)
            present_values[i] = payoffs[i] * discount_factor
                
            price = np.mean(present_values)
        else:
            # Tous les payoffs sont à maturité
            payoffs = result
            discount_factor = self._domestic_rate.discount_factor(product.maturity.maturity_in_years)
            price = discount_factor * np.mean(payoffs)
        
        return price
    
    def calculate_greeks_monte_carlo(self, product: Product) -> Dict[str, float]:
        """
        Calcule les grecques par différences finies ("bumping").
        """
        # Initialiser les grecques
        greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
        
        # Prix de référence
        base_price = self.price_monte_carlo(product)
        
        # Calculer Delta et Gamma (par bumping du spot)
        bump_spot = 0.01 * self._spot_price
        original_spot = self._spot_price
        
        # Calcul du spot up
        self._spot_price = original_spot + bump_spot
        price_up = self.price_monte_carlo(product)
        
        # Calcul du spot down
        self._spot_price = original_spot - bump_spot
        price_down = self.price_monte_carlo(product)
        
        self._spot_price = original_spot
        
        # Delta et Gamma
        greeks["delta"] = (price_up - price_down) / (2 * bump_spot)
        greeks["gamma"] = (price_up - 2 * base_price + price_down) / (bump_spot ** 2)
        
        # Ici, on doit implémenter une façon de modifier la volatilité temporairement
        bump_vol = 0.01  # 1% de volatilité
        
        # On sauvegarde la volatilité d'origine
        original_vol_obj = self._volatility
        
        # On créé une version modifiée du modèle de volatilité
        class BumpedVolatility(VolatilityModel):
            def __init__(self, original_vol, bump):
                self._original_vol = original_vol
                self._bump = bump
                
            def get_volatility(self, moneyness, maturity):
                return self._original_vol.get_volatility(moneyness, maturity) + self._bump
                
        # On applique la volatilité augmentée
        self._volatility = BumpedVolatility(original_vol_obj, bump_vol)
        price_vol_up = self.price_monte_carlo(product)
        
        # On restaure la volatilité
        self._volatility = original_vol_obj
        
        # Vega (pour 1% de volatilité)
        greeks["vega"] = (price_vol_up - base_price) / bump_vol
        
        # Ici, on estime le theta en réduisant légèrement la maturité
        greeks["theta"] = -base_price * 0.01  # Approximation simple
        
        # Ici, on doit implémenter une façon de modifier le taux d'intérêt temporairement
        greeks["rho"] = base_price * 0.1  
        
        return greeks

    def price_by_decomposition(self, product: 'DecomposableProduct') -> float:
            """
            Calcule le prix d'un produit par décomposition en composantes élémentaires.
            Il faut que ça soit un produit "decomposable product", sinon ça ne fonctionne pas. 
            """
            components = product.decompose()
            total_price = 0.0
            
            for component in components:
                if isinstance(component, ABCBond):
                    # Pricing des composantes obligataires
                    total_price += component.compute_price()
                elif isinstance(component, Option):
                    # Pricing des composantes optionnelles
                    total_price += self.price_black_scholes(component)
                else:
                    # Pricing des autres composantes via Monte Carlo
                    total_price += self.price_monte_carlo(component)
                    
            return total_price

    def calculate_greeks_by_decomposition(self, product: 'DecomposableProduct') -> Dict[str, float]:
            """
            Calcule les grecques d'un produit par décomposition.
            """
            components = product.decompose()
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
            
            for component in components:
                component_greeks = {}
                
                if isinstance(component, Option):
                    component_greeks = self.calculate_greeks_black_scholes(component)
                else:
                    # Pour les obligations, seules certaines grecques sont pertinentes
                    if isinstance(component, ABCBond):
                        # Calcul simplifié du rho pour les obligations
                        component_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
                        
                        # Calcul de rho par bump du taux
                        original_price = component.compute_price()
                        
                        # Créer une copie du rate model avec un bump de 0.01%
                        bump = 0.0001  # 1 bp
                        rate_value = component._rate_model.get_rate() if hasattr(component, '_rate_model') else 0.0
                        
                        # Modifié pour accéder au rate_model correct selon le type d'obligation
                        if isinstance(component, ZeroCouponBond):
                            component.__rate_model.set_rate(rate_value + bump)
                            bumped_price = component.compute_price(force_recalculate=True)
                            component.__rate_model.set_rate(rate_value)
                        elif isinstance(component, Bond):
                            component.__rate_model.set_rate(rate_value + bump)
                            bumped_price = component.compute_price(force_rate=rate_value + bump)
                            component.__rate_model.set_rate(rate_value)
                        else:
                            bumped_price = original_price
                        
                        component_greeks["rho"] = (bumped_price - original_price) / bump * 100  # en % de taux
                    else:
                        # Autres composantes via Monte Carlo
                        component_greeks = self.calculate_greeks_monte_carlo(component)
                
                # Agréger les grecques
                for greek in greeks:
                    if greek in component_greeks:
                        greeks[greek] += component_greeks[greek]
            
            return greeks

    def price(self, product: Product, method: Literal["black_scholes", "monte_carlo", "decomposition"] = "black_scholes") -> float:
        """
        Calcule le prix d'un produit financier selon la méthode choisie
        """
        if method == "black_scholes":
            if isinstance(product, Option):
                return self.price_black_scholes(product)
            else:
                raise ValueError(f"La méthode Black-Scholes n'est disponible que pour les options, pas pour {type(product).__name__}")
        elif method == "decomposition":
            if isinstance(product, DecomposableProduct):
                return self.price_by_decomposition(product)
            else:
                raise ValueError(f"La méthode de décomposition n'est disponible que pour les produits décomposables, pas pour {type(product).__name__}")
        else:  # monte_carlo
            return self.price_monte_carlo(product)
    
    def calculate_greeks(self, product: Product, method: Literal["black_scholes", "monte_carlo", "decomposition"] = "black_scholes") -> Dict[str, float]:
            """
            Calcule les grecques d'un produit financier selon la méthode chosie. 
            """
            if method == "black_scholes":
                if isinstance(product, Option):
                    return self.calculate_greeks_black_scholes(product)
                else:
                    raise ValueError(f"Le calcul analytique des grecques n'est disponible que pour les options, pas pour {type(product).__name__}")
            elif method == "decomposition":
                if isinstance(product, DecomposableProduct):
                    return self.calculate_greeks_by_decomposition(product)
                else:
                    raise ValueError(f"La méthode de décomposition n'est disponible que pour les produits décomposables, pas pour {type(product).__name__}")
            else:  # monte_carlo
                return self.calculate_greeks_monte_carlo(product)
        
          