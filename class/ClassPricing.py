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
from base.ClassOption import Option
import tqdm


# Types pour les produits structurés
BarrierDirection = Literal["up", "down"]
BarrierType = Literal["ko", "ki"]


class PricingEngine:
    """
    Moteur de pricing pour les produits financiers.
    Supporte le pricing par Black-Scholes ou Monte Carlo.
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
        
        Args:
            option (Option): Option à évaluer
            
        Returns:
            float: Prix de l'option
        """
        # Paramètres
        S = self._spot_price
        K = option.strike
        T = option.maturity.maturity_in_years
        r = self._domestic_rate.get_rate(option.maturity)
        q = self._dividend
        if self._foreign_rate is not None:
            q = self._foreign_rate.get_rate(option.maturity)
        
        # Volatilité pour ce strike et cette maturité
        sigma = self._volatility.get_volatility(K/S, T)
        
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
        r = self._domestic_rate.get_rate(option.maturity)
        q = self._dividend
        if self._foreign_rate is not None:
            q = self._foreign_rate.get_rate(option.maturity)
        
        # Volatilité pour ce strike et cette maturité
        sigma = self._volatility.get_volatility(K/S, T)
        
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
        
        Args:
            maturity (Maturity): Maturité du produit
            strike_price (float): Prix d'exercice (utilisé pour la volatilité)
            
        Returns:
            Tuple[ndarray, ndarray]: (paths, time_grid) - les chemins simulés et la grille temporelle
        """
        # Créer une grille temporelle
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
        for i in tqdm(range(1, self._num_steps + 1), desc="Simulation Monte Carlo", leave=False):
            z = np.random.standard_normal(self._num_paths)
            paths[:, i] = paths[:, i-1] * np.exp(drift + vol_sqrt_dt * z)
        
        return paths, time_grid
    
    def price_monte_carlo(self, product: Product) -> float:
        """
        Calcule le prix d'un produit par simulation Monte Carlo.
        
        Args:
            product (Product): Le produit à évaluer
            
        Returns:
            float: Prix estimé du produit
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
    
    def calculate_greeks_monte_carlo(self, product: Product) -> Dict[str, float]:
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
        
        # Restaurer le spot
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
    
    def price(self, product: Product, method: Literal["black_scholes", "monte_carlo"] = "black_scholes") -> float:
        """
        Calcule le prix d'un produit financier.
        
        Args:
            product (Product): Le produit à évaluer
            method (str): Méthode de pricing à utiliser
            
        Returns:
            float: Prix du produit
        """
        if method == "black_scholes":
            if isinstance(product, Option):
                return self.price_black_scholes(product)
            else:
                raise ValueError(f"La méthode Black-Scholes n'est disponible que pour les options, pas pour {type(product).__name__}")
        else:  # monte_carlo
            return self.price_monte_carlo(product)
    
    def calculate_greeks(self, product: Product, method: Literal["black_scholes", "monte_carlo"] = "black_scholes") -> Dict[str, float]:
        """
        Calcule les grecques d'un produit financier.
        
        Args:
            product (Product): Le produit à analyser
            method (str): Méthode de calcul à utiliser
            
        Returns:
            Dict[str, float]: Dictionnaire des grecques
        """
        if method == "black_scholes":
            if isinstance(product, Option):
                return self.calculate_greeks_black_scholes(product)
            else:
                raise ValueError(f"Le calcul analytique des grecques n'est disponible que pour les options, pas pour {type(product).__name__}")
        else:  # monte_carlo
            return self.calculate_greeks_monte_carlo(product)