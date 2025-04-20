# Il existe plusieurs grands "types" de produits structurés : les produits avec protection du capital, 
# les produits de rendements, les produits de participation et enfin les produits plus complexes
# Nous avons essayé d'en coder plusieurs, de chaque catégorie, pour refléter leur diversité. 
# Nous avons mis en place des classes intermédiaires, propres aux types de produits, et nous avons créé des fichiers
# pour chaque type.  


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from datetime import datetime, date

from base.ClassMaturity import Maturity, DayCountConvention
from base.ClassOption import Option
from base.ClassRate import RateModel
from structuration.ClassDerives import BarrierOption, DigitalOption
from structuration.ClassFixedIncome import ABCBond, ZeroCouponBond
from structuration.ClassVolatility import VolatilityModel

# Types pour les produits structurés
BarrierDirection = Literal["up", "down"]
BarrierType = Literal["ko", "ki"]

class Product(ABC):
    """
    Classe abstraite de base pour les produits structurés.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float = 1000.0
    ):
        """
        Initialisation d'un produit structuré.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
        """
        self._underlying_id = underlying_id
        self._maturity = maturity
        self._nominal = nominal
    
    @property
    def underlying_id(self) -> str:
        """Identifiant du sous-jacent."""
        return self._underlying_id
    
    @property
    def maturity(self) -> Maturity:
        """Maturité du produit."""
        return self._maturity
    
    @property 
    def nominal(self) -> float:
        """Valeur nominale du produit."""
        return self._nominal
    
    @abstractmethod
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calcule le payoff du produit basé sur les chemins simulés.
        Gère 2 cas : lorsque des produits vanille, il y a en général un seul payoff. Mais dans le cas de 
        Produits structurés, il peut y en avoir plusieurs selon la maturité et le type de produit.
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray, optional): Grille temporelle utilisée pour la simulation
            
        Returns:
            Union[ndarray, Tuple[ndarray, ndarray]]: 
                - Soit les payoffs uniquement si tous à la même date
                - Soit un tuple (payoffs, payment_times) avec leurs dates de paiement respectives
        """
        pass
class DecomposableProduct(Product):
    """
    Interface pour les produits qui peuvent être décomposés en composantes élémentaires.
    """
    
    @abstractmethod
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose le produit en composantes élémentaires.
        
        Returns:
            List[Union[ABCBond, Option, Product]]: Liste des composantes
        """
        pass
