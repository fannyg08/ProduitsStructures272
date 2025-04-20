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
from structuration.Produits.ProductBase import BarrierDirection, BarrierType, DecomposableProduct, Product



##### Produit plus complexes #####
class AthenaProduct(DecomposableProduct):
    """
    Produit structuré Athéna. C'est l'un des produits les plus connus, donc nous 
    avons pensé qu'il était bon de le coder dans notre projet. 
    
    Fonctionnement : 
    - Remboursement anticipé automatique aux dates d'observation si le sous-jacent est au-dessus d'un seuil (en général, niveau d'un indice)
    - Coupon conditionnel (potentiellement avec effet mémoire)
    - Barrière de protection du capital à maturité
    - Exposition à la baisse si le sous-jacent franchit la barrière à maturité
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        observation_dates: List[date],
        autocall_barriers: List[float],
        coupon_barriers: List[float],
        coupons: List[float],
        capital_barrier: float,
        memory_effect: bool = True,
        nominal: float = 1000.0
    ):
        super().__init__(underlying_id, maturity, nominal)
        
        # Vérification de la cohérence des listes
        if len(observation_dates) != len(autocall_barriers) or len(observation_dates) != len(coupon_barriers) or len(observation_dates) != len(coupons):
            raise ValueError("Les listes observation_dates, autocall_barriers, coupon_barriers et coupons doivent avoir la même longueur")
        
        self._observation_dates = observation_dates
        self._autocall_barriers = autocall_barriers
        self._coupon_barriers = coupon_barriers
        self._coupons = coupons
        self._capital_barrier = capital_barrier
        self._memory_effect = memory_effect
    
    @property
    def observation_dates(self) -> List[date]:
        return self._observation_dates
    
    @property
    def autocall_barriers(self) -> List[float]:
        return self._autocall_barriers
    
    @property
    def coupon_barriers(self) -> List[float]:
        return self._coupon_barriers
    
    @property
    def coupons(self) -> List[float]:
        return self._coupons
    
    @property
    def capital_barrier(self) -> float:
        return self._capital_barrier
    
    @property
    def memory_effect(self) -> bool:
        return self._memory_effect
    
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        if time_grid is None:
            raise ValueError("Le paramètre time_grid est requis pour calculer le payoff d'un produit Athéna")
        
        n_paths = paths.shape[0]
        payoffs = np.zeros(n_paths)
        
        initial_level = paths[:, 0]
        
        # Calcul des indices temporels correspondant aux dates d'observation
        observation_indices = []
        for obs_date in self._observation_dates:
            idx = self._find_closest_time_index(obs_date, time_grid)
            observation_indices.append(idx)
        
        # Matrice pour suivre si un chemin a déjà été remboursé
        redeemed = np.zeros(n_paths, dtype=bool)
        
        memorized_coupons = np.zeros(n_paths)
        
        for i, obs_idx in enumerate(observation_indices):
            level = paths[:, obs_idx]
            
            # Vérification de la condition d'autocall (sauf pour la dernière date qui est la maturité)
            if i < len(observation_indices) - 1:
                autocall_condition = level >= initial_level * self._autocall_barriers[i]
                autocall_paths = autocall_condition & ~redeemed
                
                if np.any(autocall_paths):
                    coupon_condition = level >= initial_level * self._coupon_barriers[i]
                    current_coupon = np.zeros(n_paths)
                    current_coupon[coupon_condition & ~redeemed] = self._coupons[i] * self.nominal
                    
                    if self._memory_effect:
                        payoffs[autocall_paths] = self.nominal + current_coupon[autocall_paths] + memorized_coupons[autocall_paths]
                    else:
                        payoffs[autocall_paths] = self.nominal + current_coupon[autocall_paths]
                    
                    redeemed[autocall_paths] = True
            
            coupon_condition = level >= initial_level * self._coupon_barriers[i]
            coupon_paths = coupon_condition & ~redeemed
            
            if np.any(coupon_paths):
                if self._memory_effect:
                    memorized_coupons[~coupon_paths & ~redeemed] += self._coupons[i] * self.nominal
                    memorized_coupons[coupon_paths] = 0  
                else:
                    pass
        
        # Traitement à maturité pour les chemins qui n'ont pas été remboursés par anticipation
        maturity_idx = observation_indices[-1]
        maturity_level = paths[:, maturity_idx]
        not_redeemed = ~redeemed
        
        if np.any(not_redeemed):
            capital_protected = maturity_level >= initial_level * self._capital_barrier
            
            protected_paths = capital_protected & not_redeemed
            if np.any(protected_paths):
                coupon_condition = maturity_level >= initial_level * self._coupon_barriers[-1]
                current_coupon = np.zeros(n_paths)
                current_coupon[coupon_condition & not_redeemed] = self._coupons[-1] * self.nominal
                
                if self._memory_effect:
                    payoffs[protected_paths] = self.nominal + current_coupon[protected_paths] + memorized_coupons[protected_paths]
                else:
                    payoffs[protected_paths] = self.nominal + current_coupon[protected_paths]
            
            unprotected_paths = ~capital_protected & not_redeemed
            if np.any(unprotected_paths):
                performance = maturity_level[unprotected_paths] / initial_level[unprotected_paths]
                payoffs[unprotected_paths] = self.nominal * performance
        
        return payoffs
    
    def _find_closest_time_index(self, target_date: date, time_grid: np.ndarray) -> int:
        """
        Trouve l'indice le plus proche dans la grille temporelle pour une date donnée.
        """
        return len(time_grid) - 1  
  
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose le produit Athéna. 
        
        Un produit Athéna peut être décomposé en:
        1. Une obligation zéro-coupon (pour le remboursement du nominal)
        2. Une série d'options digitales (pour les coupons)
        3. Des options barrière down-and-in put (pour l'exposition à la baisse si la barrière est franchie)
        4. Des options digitales avec barrière (pour le mécanisme d'autocall)
        """
        components = []
        
        # Paramètres communs pour la création des options
        spot_price = None 
        rate_model = None  
        volatility = None  
        
        # 1. Obligation zéro-coupon pour le remboursement du nominal à maturité
        zc_bond = ZeroCouponBond(
            rate_model=rate_model,
            maturity=self._maturity,
            nominal=self._nominal
        )
        components.append(zc_bond)
        
        # 2. Options digitales pour les coupons à chaque date d'observation
        for i, obs_date in enumerate(self._observation_dates):
            obs_maturity = Maturity(obs_date)
            
            # Option digitale pour le coupon
            digital_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._coupon_barriers[i],  # Strike basé sur la barrière de coupon
                maturity=obs_maturity,
                domestic_rate=rate_model,
                volatility=volatility,
                option_type="call",  # Call car nous voulons que le sous-jacent soit au-dessus de la barrière
                payment=self._coupons[i] * self._nominal, 
                dividend=0.0 
            )
            components.append(digital_option)
            
            # Si effet mémoire, nous devons ajouter des options pour récupérer les coupons précédents
            if self._memory_effect and i > 0:
                # Pour chaque date d'observation précédente
                for j in range(i):
                    # Option digitale conditionnelle pour récupérer un coupon précédent
                    # Ce serait une option qui paie le coupon j si le sous-jacent est au-dessus de la barrière à la date i
                    # et qui n'a pas déjà été payé à une date antérieure
                    memory_digital_option = DigitalOption(
                        spot_price=spot_price,
                        strike_price=spot_price * self._coupon_barriers[i],
                        maturity=obs_maturity,
                        domestic_rate=rate_model,
                        volatility=volatility,
                        option_type="call",
                        payment=self._coupons[j] * self._nominal,
                        dividend=0.0
                    )
                    components.append(memory_digital_option)
        
        # 3. Mécanisme d'autocall (remboursement anticipé automatique)
        # Pour chaque date d'observation sauf la maturité
        for i, obs_date in enumerate(self._observation_dates[:-1]):
            # Création d'une maturité pour cette date d'observation
            obs_maturity = Maturity(obs_date)
            
            # Option digitale qui paie le nominal si le sous-jacent est au-dessus de la barrière d'autocall
            autocall_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._autocall_barriers[i],  # Strike basé sur la barrière d'autocall cette fois-ci
                maturity=obs_maturity,
                domestic_rate=rate_model,
                volatility=volatility,
                option_type="call",
                payment=self._nominal,  # Paie le nominal
                dividend=0.0
            )
            components.append(autocall_option)
            
        # 4. Protection conditionnelle du capital à maturité
        # Si le sous-jacent est en-dessous de la barrière à maturité, l'investisseur est exposé à la baisse
        # Cela peut être modélisé comme une option barrière down-and-in put
        barrier_option = BarrierOption(
            spot_price=spot_price,
            strike_price=spot_price,  # Strike égal au niveau initial
            maturity=self._maturity,
            domestic_rate=rate_model,
            volatility=volatility,
            option_type="put",  # Put car nous sommes exposés à la baisse
            barrier=spot_price * self._capital_barrier,  # Barrière de protection du capital
            barrier_type="down",  # Barrière basse
            knock_type="in",  # L'option est activée si la barrière est franchie
            dividend=0.0
        )
        components.append(barrier_option)
        
        return components
    

# autocall Phoenix -> variante de l'Athéna

# twin win certificate (option déjà codée)

# range accrual note 

# callable/putable structured products -> option pour l'émetteur de racheter le produit/remboursement anticipé
