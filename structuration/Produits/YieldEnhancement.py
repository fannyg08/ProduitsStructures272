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
from structuration.Produits.ProductBase import DecomposableProduct, Product


class YieldEnhancementProduct(Product):
    """
    Classe abstraite intermédiaire pour les produits d'amélioration du rendement.
    Ces produits offrent généralement un rendement supérieur contre un prise de risque sur le capital.
    DOnc pas de garantie en capital sur ces produits
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        nominal: float = 1000.0,
        coupon: float = 0.0
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._coupon = coupon
    
    @property
    def coupon(self) -> float:
        return self._coupon



class ReverseConvertible(YieldEnhancementProduct, DecomposableProduct):
    """
    Reverse Convertible: Un produit structuré qui offre un coupon fixe élevé,
    mais expose l'investisseur au risque de baisse du sous-jacent.
    
    Caractéristiques:
    - Coupon fixe garanti
    - Si le sous-jacent termine au-dessus du strike, remboursement du nominal
    - Si le sous-jacent termine en-dessous du strike, l'investisseur reçoit
      des actions du sous-jacent ou l'équivalent en cash (performance)
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        nominal: float = 1000.0,
        coupon: float = 0.05,  # 5% par défaut
        strike_level: float = 1.0,  # 100% du niveau initial par défaut
    ):
        super().__init__(underlying_id, maturity, nominal, coupon)
        self._strike_level = strike_level
        
    @property
    def strike_level(self) -> float:
        """Niveau du strike en % du niveau initial."""
        return self._strike_level
    
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        n_paths = paths.shape[0]
        n_times = paths.shape[1]
        
        initial_level = paths[:, 0]
        
        # Niveau final du sous-jacent
        final_level = paths[:, -1]
        
        # Calcul du strike pour chaque chemin
        strike = initial_level * self._strike_level
        
        # Calcul du coupon (on suppose à maturité)
        coupon_payment = self._nominal * self._coupon * self._maturity.maturity_in_years
        
        # Calcul du payoff
        payoffs = np.zeros(n_paths)
        
        # Si le niveau final est au-dessus du strike, remboursement du nominal + coupon
        above_strike = final_level >= strike
        payoffs[above_strike] = self._nominal + coupon_payment
        
        # Si le niveau final est en-dessous du strike, l'investisseur reçoit la performance + coupon
        below_strike = ~above_strike
        payoffs[below_strike] = self._nominal * (final_level[below_strike] / initial_level[below_strike]) + coupon_payment
        
        return payoffs
    
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose le Reverse Convertible en composantes élémentaires.
        
        Un Reverse Convertible peut être décomposé en:
        1. Une obligation zéro-coupon (pour le paiement du coupon)
        2. Une position longue sur une obligation zéro-coupon
        3. Une position courte sur une option put (vente de protection)
        
        Returns:
            List[Union[ABCBond, Option, Product]]: Liste des composantes
        """
        # Cette méthode nécessiterait les vraies classes d'obligations et d'options
        # Voici une implémentation conceptuelle
        components = []
        
        
        # 1. Obligation pour le coupon
        # Obligation qui paie le coupon à maturité
        
        # 2. Obligation zéro-coupon pour le nominal
        # Obligation qui paie le nominal à maturité
        
        # 3. Option put vendue
        # L'investisseur vend implicitement une option put sur le sous-jacent
        # Strike = niveau initial * strike_level
        
        return components
    
    def __str__(self) -> str:
        return (f"Reverse Convertible on {self._underlying_id}, "
                f"Maturity: {self._maturity.maturity_date}, "
                f"Nominal: {self._nominal}, "
                f"Coupon: {self._coupon * 100:.2f}%, "
                f"Strike: {self._strike_level * 100:.2f}% of initial level")

class DiscountCertificate(YieldEnhancementProduct, DecomposableProduct):
    """
    Discount Certificate: Un produit structuré qui permet d'acheter un sous-jacent
    avec une décote en échange d'un plafonnement des gains.
    
    Caractéristiques:
    - Prix d'achat inférieur au prix du sous-jacent (discount/décote)
    - Participation à la hausse du sous-jacent jusqu'à un plafond (cap)
    - Exposition directe à la baisse du sous-jacent
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        nominal: float = 1000.0,
        discount: float = 0.1, 
        cap_level: float = 1.1,  
    ):

        # Le "coupon" pour un Discount Certificate est implicite via la décote (donc =0)
        # La valeur de ce coupon serait calculée en fonction de la décote et du cap
        coupon = 0.0  
        super().__init__(underlying_id, maturity, nominal, coupon)
        self._discount = discount
        self._cap_level = cap_level
        
    @property
    def discount(self) -> float:
        """Niveau de décote par rapport au prix du sous-jacent."""
        return self._discount
    
    @property
    def cap_level(self) -> float:
        """Niveau du plafond (cap) en % du niveau initial."""
        return self._cap_level
    
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        n_paths = paths.shape[0]
        
        initial_level = paths[:, 0]
        final_level = paths[:, -1]
        
        # Calcul du cap
        cap = initial_level * self._cap_level
        
        purchase_price = self._nominal * (1 - self._discount)
        
        payoffs = np.zeros(n_paths)
        
        # Si le niveau final est au-dessus du cap, remboursement au niveau du cap
        above_cap = final_level >= cap
        payoffs[above_cap] = self._nominal * self._cap_level
        
        # Si le niveau final est entre l'initial et le cap, remboursement indexé sur la performance
        between_initial_and_cap = (final_level < cap) & (final_level >= initial_level)
        payoffs[between_initial_and_cap] = self._nominal * (final_level[between_initial_and_cap] / initial_level[between_initial_and_cap])
        
        # Si le niveau final est en-dessous de l'initial, remboursement indexé sur la performance
        below_initial = final_level < initial_level
        payoffs[below_initial] = self._nominal * (final_level[below_initial] / initial_level[below_initial])
        
        return payoffs
    
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose le Discount Certificate en composantes élémentaires.
        
        Un Discount Certificate peut être décomposé en:
        1. Un investissement direct dans le sous-jacent
        2. Une position courte sur une option call (vente du potentiel de hausse au-delà du cap)
        """
        components = []
        
        # 1. Exposition directe au sous-jacent
        # Équivalent à une détention du sous-jacent
        
        # 2. Option call vendue
        # L'investisseur vend implicitement une option call sur le sous-jacent
        # avec un strike au niveau du cap
        
        return components
    
    def __str__(self) -> str:
        return (f"Discount Certificate on {self._underlying_id}, "
                f"Maturity: {self._maturity.maturity_date}, "
                f"Nominal: {self._nominal}, "
                f"Discount: {self._discount * 100:.2f}%, "
                f"Cap: {self._cap_level * 100:.2f}% of initial level")
    
