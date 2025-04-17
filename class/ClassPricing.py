from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass
from ClassProduit import Product,CapitalProtectedNoteTwinWin,CapitalProtectedNote, CapitalProtectedNoteWithBarrier, CapitalProtectedNoteWithCoupon
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from datetime import datetime
from base.ClassMaturity import Maturity, DayCountConvention
from base.ClassRate import RateModel
from base.ClassVolatility import VolatilityModel


# Types pour les produits structurés
BarrierDirection = Literal["up", "down"]
BarrierType = Literal["ko", "ki"]
class PricingEngine:
    """
    Moteur de pricing pour les produits structurés.
    Cette classe gère les calculs de prix et de risques pour différents produits.
    Supporte les méthodes analytiques (Black-Scholes) et numériques (Monte Carlo).
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
        seed: Optional[int] = None,
        pricing_method: Literal["monte_carlo", "analytic", "auto"] = "auto"
    ):
        """
        Initialisation du moteur de pricing.
        
        Args:
            spot_price (float): Prix actuel du sous-jacent
            domestic_rate (RateModel): Modèle de taux domestique
            volatility (VolatilityModel): Modèle de volatilité
            dividend (float, optional): Taux de dividende continu du sous-jacent
            foreign_rate (RateModel, optional): Taux étranger pour options de change
            num_paths (int): Nombre de trajectoires pour Monte Carlo
            num_steps (int): Nombre de pas temporels pour la simulation
            seed (int, optional): Graine pour la reproductibilité des simulations
            pricing_method (str): Méthode de pricing à utiliser ("monte_carlo", "analytic" ou "auto")
        """
        self._spot_price = spot_price
        self._domestic_rate = domestic_rate
        self._volatility = volatility
        self._dividend = dividend if dividend is not None else 0.0
        self._foreign_rate = foreign_rate
        self._num_paths = num_paths
        self._num_steps = num_steps
        self._pricing_method = pricing_method
        
        # Initialiser le générateur aléatoire si une graine est fournie
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_paths(self, maturity: Maturity, strike_price: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère des trajectoires simulées pour le sous-jacent avec un modèle log-normal.
        
        Args:
            maturity (Maturity): Maturité du produit
            strike_price (float): Prix d'exercice (utilisé pour la volatilité)
            
        Returns:
            Tuple[ndarray, ndarray]: (paths, time_grid) - les chemins simulés et la grille temporelle
        """
        # Le code existant reste inchangé
        time_grid = np.linspace(0, maturity.maturity_in_years, self._num_steps + 1)
        dt = maturity.maturity_in_years / self._num_steps
        
        # Initialiser le tableau des chemins
        paths = np.zeros((self._num_paths, self._num_steps + 1))
        paths[:, 0] = self._spot_price
        
        # Paramètres du modèle log-normal
        r = self._domestic_rate.get_rate(maturity)
        q = self._dividend
        if self._foreign_rate is not None:
            q = self._foreign_rate.get_rate(maturity)
        
        vol = self._volatility.get_volatility(
            strike_price / self._spot_price, 
            maturity.maturity_in_years
        )
        
        # Terme de dérive
        drift = (r - q - 0.5 * vol**2) * dt
        vol_sqrt_dt = vol * np.sqrt(dt)
        
        # Générer les chemins
        for i in range(1, self._num_steps + 1):
            z = np.random.standard_normal(self._num_paths)
            paths[:, i] = paths[:, i-1] * np.exp(drift + vol_sqrt_dt * z)
        
        return paths, time_grid
    
    def black_scholes_price(self, option_type: Literal["call", "put"], 
                           strike: float, maturity: Maturity) -> float:
        """
        Calcule le prix d'une option vanille avec la formule de Black-Scholes.
        
        Args:
            option_type (str): Type d'option ("call" ou "put")
            strike (float): Prix d'exercice
            maturity (Maturity): Maturité de l'option
            
        Returns:
            float: Prix Black-Scholes de l'option
        """
        S = self._spot_price
        K = strike
        T = maturity.maturity_in_years
        r = self._domestic_rate.get_rate(maturity)
        q = self._dividend
        if self._foreign_rate is not None:
            q = self._foreign_rate.get_rate(maturity)
        
        # On utilise la volatilité correspondant au strike et à la maturité
        sigma = self._volatility.get_volatility(K/S, T)
        
        # Formule de Black-Scholes
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            price = S * np.exp(-q * T) * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * np.exp(-q * T) * self._norm_cdf(-d1)
            
        return price
    
    def _norm_cdf(self, x: float) -> float:
        """
        Fonction de répartition de la loi normale (pour Black-Scholes).
        
        Args:
            x (float): Valeur à évaluer
            
        Returns:
            float: Valeur de la fonction de répartition
        """
        return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0
    
    def can_price_analytically(self, product: "Product") -> bool:
        """
        Détermine si un produit peut être évalué par une méthode analytique.
        
        Args:
            product (Product): Le produit à évaluer
            
        Returns:
            bool: True si une solution analytique est disponible
        """
        # Implémenter la logique de détection des produits qui peuvent être évalués analytiquement
        # Par exemple, vérifier si c'est une option européenne standard (call/put vanille)
        return hasattr(product, "has_analytic_solution") and product.has_analytic_solution
    
    def price_product(self, product: "Product", method: Optional[str] = None) -> float:
        """
        Calcule le prix d'un produit structuré.
        
        Args:
            product (Product): Le produit à évaluer 
            method (str, optional): Méthode de pricing à utiliser pour cette évaluation spécifique
                                   (si None, la méthode par défaut sera utilisée)
            
        Returns:
            float: Prix estimé du produit
        """
        # Déterminer la méthode de pricing à utiliser
        pricing_method = method if method is not None else self._pricing_method
        
        if pricing_method == "auto":
            # Choisir automatiquement la méthode appropriée
            if self.can_price_analytically(product):
                pricing_method = "analytic"
            else:
                pricing_method = "monte_carlo"
        
        # Pricing analytique si possible et demandé
        if pricing_method == "analytic" and self.can_price_analytically(product):
            return self._price_product_analytically(product)
        
        # Sinon, utiliser Monte Carlo
        return self._price_product_monte_carlo(product)
    
    def _price_product_analytically(self, product: "Product") -> float:
        """
        Calcule le prix d'un produit avec une méthode analytique.
        
        Args:
            product (Product): Le produit à évaluer
            
        Returns:
            float: Prix calculé analytiquement
        """
        # Vérifier si le produit possède une méthode pour se pricing analytiquement
        if hasattr(product, "price_analytically"):
            return product.price_analytically(
                spot=self._spot_price,
                rate=self._domestic_rate,
                foreign_rate=self._foreign_rate,
                volatility=self._volatility,
                dividend=self._dividend
            )
        
        # Sinon, implémenter ici la logique de pricing analytique pour les cas standards
        # Exemple pour une option vanille européenne
        if hasattr(product, "option_type") and hasattr(product, "strike"):
            return self.black_scholes_price(
                product.option_type,
                product.strike,
                product.maturity
            )
        
        raise ValueError(f"Pas de méthode analytique disponible pour {type(product).__name__}")
    
    def _price_product_monte_carlo(self, product: "Product") -> float:
        """
        Calcule le prix d'un produit par simulation Monte Carlo.
        
        Args:
            product (Product): Le produit à évaluer
            
        Returns:
            float: Prix estimé par Monte Carlo
        """
        # Code existant pour le pricing Monte Carlo
        # Simuler les trajectoires pour ce produit
        paths, time_grid = self.simulate_paths(product.maturity, getattr(product, "strike", self._spot_price))
        
        # Calculer les payoffs
        result = product.payoff(paths, time_grid)
        
        # Traiter les résultats selon leur format (payoffs à date unique ou multiple)
        if isinstance(result, tuple) and len(result) == 2:
            payoffs, payment_times = result
            
            # Vecteur de prix présents
            present_values = np.zeros_like(payoffs)
            
            # Actualiser chaque payoff selon sa date de paiement
            for i, t in enumerate(payment_times):
                discount_factor = np.exp(-self._domestic_rate.get_rate(Maturity(maturity_in_years=t)) * t)
                present_values[i] = payoffs[i] * discount_factor
                
            price = np.mean(present_values)
        else:
            # Tous les payoffs sont à maturité
            payoffs = result
            discount_factor = np.exp(-self._domestic_rate.get_rate(product.maturity) * 
                                  product.maturity.maturity_in_years)
            price = discount_factor * np.mean(payoffs)
        
        return price
    
    def compute_greeks(self, product: "Product", method: Optional[str] = None) -> Dict[str, float]:
        """
        Calcule les principales grecques du produit.
        
        Args:
            product (Product): Le produit à analyser
            method (str, optional): Méthode à utiliser ("monte_carlo", "analytic", "auto")
            
        Returns:
            Dict[str, float]: Dictionnaire des grecques (delta, gamma, vega, theta, rho)
        """
        # Initialiser les grecques
        greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
        
        # Déterminer la méthode de calcul
        pricing_method = method if method is not None else self._pricing_method
        
        if pricing_method == "auto":
            if self.can_price_analytically(product):
                pricing_method = "analytic"
            else:
                pricing_method = "monte_carlo"
        
        # Calcul des grecques selon la méthode choisie
        if pricing_method == "analytic" and self.can_price_analytically(product):
            return self._compute_greeks_analytically(product)
        else:
            return self._compute_greeks_by_bumping(product)
    
    def _compute_greeks_analytically(self, product: "Product") -> Dict[str, float]:
        """
        Calcule les grecques analytiquement pour les produits qui le supportent.
        
        Args:
            product (Product): Le produit à analyser
            
        Returns:
            Dict[str, float]: Dictionnaire des grecques
        """
        # Vérifier si le produit possède sa propre méthode de calcul des grecques
        if hasattr(product, "compute_greeks_analytically"):
            return product.compute_greeks_analytically(
                spot=self._spot_price,
                rate=self._domestic_rate,
                foreign_rate=self._foreign_rate,
                volatility=self._volatility,
                dividend=self._dividend
            )
        
        # Sinon, implémenter le calcul analytique des grecques pour les cas standards
        # Exemple pour une option vanille avec Black-Scholes
        greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
        
        # Formules analytiques pour les grecques...
        # (implémentation des formules de Black-Scholes pour les grecques)
        
        return greeks
    
    def _compute_greeks_by_bumping(self, product: "Product") -> Dict[str, float]:
        """
        Calcule les grecques par différences finies (bumping).
        
        Args:
            product (Product): Le produit à analyser
            
        Returns:
            Dict[str, float]: Dictionnaire des grecques
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
        base_price = self.price_product(product, method="monte_carlo")
        
        # Delta: dérivée par rapport au spot
        bump_spot = 0.01 * self._spot_price
        original_spot = self._spot_price
        
        # Calcul du spot up
        self._spot_price = original_spot + bump_spot
        price_up = self.price_product(product, method="monte_carlo")
        
        # Calcul du spot down (pour gamma)
        self._spot_price = original_spot - bump_spot
        price_down = self.price_product(product, method="monte_carlo")
        
        # Restaurer le spot
        self._spot_price = original_spot
        
        # Calcul delta et gamma
        greeks["delta"] = (price_up - price_down) / (2 * bump_spot)
        greeks["gamma"] = (price_up - 2 * base_price + price_down) / (bump_spot ** 2)
        
        # Calcul similaire pour les autres grecques...
        # (implémentation des calculs pour vega, theta, rho)
        
        return greeks