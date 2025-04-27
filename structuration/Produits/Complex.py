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
        nominal: float = 1000.0,
        spot_price: float = 100.0,
        rate_model: RateModel = None,
        pricing_date: datetime = None,
        volatility_model: VolatilityModel = None
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._spot_price = spot_price
        if isinstance(pricing_date, str):
            self._pricing_date = datetime.strptime(pricing_date, "%Y-%m-%d").date()
        else:
            self._pricing_date = pricing_date
        self._rate_model = rate_model
        self._volatility_model = volatility_model
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
        spot_price = self._spot_price 
        rate_model = self._rate_model
        volatility = self._volatility_model  
        
        # 1. Obligation zéro-coupon pour le remboursement du nominal à maturité
        zc_bond = ZeroCouponBond(
            rate_model=rate_model,
            maturity=self._maturity,
            nominal=self._nominal
        )
        components.append(zc_bond)
        
        # 2. Options digitales pour les coupons à chaque date d'observation
        for i, obs_date in enumerate(self._observation_dates):
            obs_maturity = Maturity(
                start_date=self._pricing_date,
                end_date=obs_date)
            
            # Option digitale pour le coupon
            digital_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._coupon_barriers[i],  # Strike basé sur la barrière de coupon
                maturity=obs_maturity,
                domestic_rate=self._rate_model,
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
                        domestic_rate=self._rate_model,
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
            obs_maturity_autocall = Maturity(
                start_date=self._pricing_date,
                end_date=self._observation_dates[i],
                day_count_convention="ACT/365"
            )
            
            # Option digitale qui paie le nominal si le sous-jacent est au-dessus de la barrière d'autocall
            autocall_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._autocall_barriers[i],  # Strike basé sur la barrière d'autocall cette fois-ci
                maturity=obs_maturity_autocall,
                domestic_rate=self._rate_model,
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
class PhoenixProduct(DecomposableProduct):
    """
    Produit structuré Phoenix (variante de l'Athéna).
    
    Fonctionnement :
    - Remboursement anticipé automatique aux dates d'observation si le sous-jacent est au-dessus d'un seuil- donc comme un autocall
    - Coupons conditionnels payés périodiquement si le sous-jacent reste au-dessus d'une barrière de coupon
      (plus basse que la barrière d'autocall)
    - Effet mémoire pour les coupons manqués
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
        nominal: float = 1000.0,
        spot_price: float = 100.0,
        rate_model: RateModel = None,
        pricing_date: datetime = None,
        volatility_model: VolatilityModel = None
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._spot_price = spot_price
        if isinstance(pricing_date, str):
            self._pricing_date = datetime.strptime(pricing_date, "%Y-%m-%d").date()
        else:
            self._pricing_date = pricing_date
        self._rate_model = rate_model
        self._volatility_model = volatility_model
        
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
        """
        Calcule le payoff du produit Phoenix basé sur les chemins simulés.
        
        La principale différence avec l'Athéna est que les coupons sont payés à chaque date
        d'observation si le sous-jacent est au-dessus de la barrière de coupon, même si
        la condition d'autocall n'est pas déclenchée.
        """
        if time_grid is None:
            raise ValueError("Le paramètre time_grid est requis pour calculer le payoff d'un produit Phoenix")
        
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
        
        # Matrice pour suivre les coupons mémorisés si effet mémoire
        memorized_coupons = np.zeros(n_paths)
        
        # Pour chaque date d'observation
        for i, obs_idx in enumerate(observation_indices):
            level = paths[:, obs_idx]
            
            # Vérification de la condition de coupon pour tous les chemins non remboursés
            coupon_condition = level >= initial_level * self._coupon_barriers[i]
            coupon_paths = coupon_condition & ~redeemed
            
            # Paiement des coupons pour les chemins au-dessus de la barrière de coupon
            if np.any(coupon_paths):
                if self._memory_effect:
                    # Avec effet mémoire: on paie les coupons mémorisés plus le coupon actuel
                    payoffs[coupon_paths] += (self._coupons[i] * self.nominal + memorized_coupons[coupon_paths])
                    memorized_coupons[coupon_paths] = 0  # Réinitialisation après paiement
                else:
                    # Sans effet mémoire: on paie seulement le coupon actuel
                    payoffs[coupon_paths] += self._coupons[i] * self.nominal
            
            # Mémorisation des coupons manqués pour les chemins sous la barrière
            missed_coupon_paths = ~coupon_condition & ~redeemed
            if self._memory_effect and np.any(missed_coupon_paths):
                memorized_coupons[missed_coupon_paths] += self._coupons[i] * self.nominal
            
            # Vérification de la condition d'autocall (sauf pour la dernière date qui est la maturité)
            if i < len(observation_indices) - 1:
                autocall_condition = level >= initial_level * self._autocall_barriers[i]
                autocall_paths = autocall_condition & ~redeemed
                
                # Remboursement anticipé pour les chemins qui atteignent la condition d'autocall
                if np.any(autocall_paths):
                    payoffs[autocall_paths] += self.nominal  
                    redeemed[autocall_paths] = True
        
        # Traitement à maturité pour les chemins non remboursés par anticipation
        maturity_idx = observation_indices[-1]
        maturity_level = paths[:, maturity_idx]
        not_redeemed = ~redeemed
        
        if np.any(not_redeemed):
            # Protection du capital si au-dessus de la barrière
            capital_protected = maturity_level >= initial_level * self._capital_barrier
            
            # Remboursement du nominal pour les chemins protégés
            protected_paths = capital_protected & not_redeemed
            if np.any(protected_paths):
                payoffs[protected_paths] += self.nominal
            
            # Exposition à la baisse pour les chemins non protégés
            unprotected_paths = ~capital_protected & not_redeemed
            if np.any(unprotected_paths):
                performance = maturity_level[unprotected_paths] / initial_level[unprotected_paths]
                payoffs[unprotected_paths] += self.nominal * performance
        
        return payoffs
    
    def _find_closest_time_index(self, target_date: date, time_grid: np.ndarray) -> int:
        """
        Trouve l'indice le plus proche dans la grille temporelle pour une date donnée.
        Cette méthode devrait être implémentée correctement pour trouver l'indice
        le plus proche, plutôt que simplement retourner le dernier indice.
        """
        # Dans une implémentation réelle, vous devriez comparer target_date avec
        # les dates correspondantes dans time_grid pour trouver l'indice le plus proche.
        # Pour cela, il faudrait convertir target_date en nombre de jours ou d'années 
        # depuis une date de référence, puis comparer avec time_grid.
        
        # Simulons une implémentation plus correcte ici:
        if target_date >= self._maturity.end_date:
            return len(time_grid) - 1
        
        # En supposant que time_grid représente des fractions d'années depuis pricing_date
        days_from_start = (target_date - self._pricing_date).days
        years_from_start = days_from_start / 365.0
        
        # Trouver l'indice le plus proche
        closest_idx = np.abs(time_grid - years_from_start).argmin()
        return closest_idx
    
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose le produit Phoenix en ses composants financiers de base.
        """
        components = []
        spot_price = self._spot_price
        rate_model = self._rate_model
        volatility = self._volatility_model

        # 1. Obligation zéro-coupon pour le remboursement du nominal à maturité
        zc_bond = ZeroCouponBond(
            rate_model=rate_model,
            maturity=self._maturity,
            nominal=self._nominal
        )
        components.append(zc_bond)

        # 2. Options digitales pour les coupons à chaque date d'observation
        for i, obs_date in enumerate(self._observation_dates):
            obs_maturity = Maturity(
                start_date=self._pricing_date,
                end_date=obs_date
            )
            
            # Option digitale pour le coupon
            digital_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._coupon_barriers[i],
                maturity=obs_maturity,
                domestic_rate=rate_model,
                volatility=volatility,
                option_type="call",
                payment=self._coupons[i] * self._nominal,
                dividend=0.0
            )
            components.append(digital_option)
            
            # Si effet mémoire, nous devons ajouter des options pour récupérer les coupons précédents
            if self._memory_effect and i > 0:
                # Pour chaque date d'observation précédente
                for j in range(i):
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
            obs_maturity = Maturity(
                start_date=self._pricing_date,
                end_date=obs_date,
                day_count_convention="ACT/365"
            )
            
            # Option digitale qui paie le nominal si le sous-jacent est au-dessus de la barrière d'autocall
            autocall_option = DigitalOption(
                spot_price=spot_price,
                strike_price=spot_price * self._autocall_barriers[i],
                maturity=obs_maturity,
                domestic_rate=rate_model,
                volatility=volatility,
                option_type="call",
                payment=self._nominal,
                dividend=0.0
            )
            components.append(autocall_option)

        # 4. Protection conditionnelle du capital à maturité
        # Si le sous-jacent est en-dessous de la barrière à maturité, l'investisseur est exposé à la baisse
        barrier_option = BarrierOption(
            spot_price=spot_price,
            strike_price=spot_price,
            maturity=self._maturity,
            domestic_rate=rate_model,
            volatility=volatility,
            option_type="put",
            barrier=spot_price * self._capital_barrier,
            barrier_type="down",
            knock_type="in",
            dividend=0.0
        )
        components.append(barrier_option)

        return components


class RangeAccrualNote(DecomposableProduct):
    """
    Produit structuré Range Accrual Note.
    
    Fonctionnement :
    - Paie un taux d'intérêt prédéfini multiplié par le nombre de jours où le sous-jacent
      se trouve dans une plage spécifiée (range)
    - Remboursement du nominal à maturité, généralement avec une protection du capital
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        coupon_rate: float,
        lower_barrier: float,
        upper_barrier: float,
        observation_dates: List[date],
        payment_dates: List[date],
        capital_protection: float = 1.0,
        nominal: float = 1000.0,
        spot_price: float = 100.0,
        rate_model: RateModel = None,
        pricing_date: datetime = None,
        volatility_model: VolatilityModel = None
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._spot_price = spot_price
        if isinstance(pricing_date, str):
            self._pricing_date = datetime.strptime(pricing_date, "%Y-%m-%d").date()
        else:
            self._pricing_date = pricing_date
        self._rate_model = rate_model
        self._volatility_model = volatility_model
        
        # Vérifications
        if lower_barrier >= upper_barrier:
            raise ValueError("La barrière basse doit être inférieure à la barrière haute")
        if len(payment_dates) == 0:
            raise ValueError("La liste des dates de paiement ne peut pas être vide")
        if len(observation_dates) == 0:
            raise ValueError("La liste des dates d'observation ne peut pas être vide")
            
        self._coupon_rate = coupon_rate
        self._lower_barrier = lower_barrier
        self._upper_barrier = upper_barrier
        self._observation_dates = observation_dates
        self._payment_dates = payment_dates
        self._capital_protection = capital_protection
    
    @property
    def coupon_rate(self) -> float:
        return self._coupon_rate
    
    @property
    def lower_barrier(self) -> float:
        return self._lower_barrier
    
    @property
    def upper_barrier(self) -> float:
        return self._upper_barrier
    
    @property
    def observation_dates(self) -> List[date]:
        return self._observation_dates
    
    @property
    def payment_dates(self) -> List[date]:
        return self._payment_dates
    
    @property
    def capital_protection(self) -> float:
        return self._capital_protection
    
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcule le payoff de la Range Accrual Note basé sur les chemins simulés.
        Pour être en cohérence avec la classe AthenaProduct, on retourne uniquement
        les valeurs finales des payoffs (et non les dates de paiement).
        """
        if time_grid is None:
            raise ValueError("Le paramètre time_grid est requis pour calculer le payoff d'une Range Accrual Note")
        
        n_paths = paths.shape[0]
        initial_level = paths[:, 0]
        
        # Payoffs finaux (somme de tous les coupons + remboursement du capital)
        payoffs = np.zeros(n_paths)
        
        # Calculer les indices temporels pour les dates d'observation et paiement
        observation_indices = [self._find_closest_time_index(obs_date, time_grid) for obs_date in self._observation_dates]
        payment_indices = [self._find_closest_time_index(pay_date, time_grid) for pay_date in self._payment_dates]
        
        # Regrouper les dates d'observation par période de paiement
        observation_periods = []
        current_period = []
        current_payment_idx = 0
        
        # Attribution des observations aux périodes de paiement
        for obs_date, obs_idx in zip(self._observation_dates, observation_indices):
            # Trouver la période de paiement correspondante
            while current_payment_idx < len(self._payment_dates) and obs_date > self._payment_dates[current_payment_idx]:
                if current_period:
                    observation_periods.append(current_period)
                    current_period = []
                current_payment_idx += 1
            
            if current_payment_idx < len(self._payment_dates):
                current_period.append(obs_idx)
        
        # Ajouter la dernière période si elle existe
        if current_period and current_payment_idx < len(self._payment_dates):
            observation_periods.append(current_period)
        
        # Calculer les coupons pour chaque période de paiement
        for i, period_indices in enumerate(observation_periods):
            if i < len(payment_indices):
                # Compter les jours dans la plage pour chaque chemin
                in_range_count = np.zeros(n_paths)
                total_observations = len(period_indices)
                
                for obs_idx in period_indices:
                    level = paths[:, obs_idx]
                    in_range = (level >= initial_level * self._lower_barrier) & (level <= initial_level * self._upper_barrier)
                    in_range_count += in_range.astype(int)
                
                # Calculer le coupon proportionnel au nombre de jours dans la plage
                if total_observations > 0:
                    proportion_in_range = in_range_count / total_observations
                    payoffs += self._nominal * self._coupon_rate * proportion_in_range
        
        # Remboursement du capital à maturité (avec protection)
        payoffs += self._nominal * self._capital_protection
        
        return payoffs
    
    def _find_closest_time_index(self, target_date: date, time_grid: np.ndarray) -> int:
        """
        Trouve l'indice le plus proche dans la grille temporelle pour une date donnée.
        """
        if target_date >= self._maturity.end_date:
            return len(time_grid) - 1
        
        # En supposant que time_grid représente des fractions d'années depuis pricing_date
        days_from_start = (target_date - self._pricing_date).days
        years_from_start = days_from_start / 365.0
        
        # Trouver l'indice le plus proche
        closest_idx = np.abs(time_grid - years_from_start).argmin()
        return closest_idx
    
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose la Range Accrual Note en ses composants financiers de base.
        """
        components = []
        spot_price = self._spot_price
        rate_model = self._rate_model
        volatility = self._volatility_model

        # 1. Obligation zéro-coupon pour la protection du capital
        zc_bond = ZeroCouponBond(
            rate_model=rate_model,
            maturity=self._maturity,
            nominal=self._nominal * self._capital_protection
        )
        components.append(zc_bond)

        # 2. Pour chaque période de paiement
        for i, payment_date in enumerate(self._payment_dates):
            payment_maturity = Maturity(
                start_date=self._pricing_date,
                end_date=payment_date
            )
            
            # Identifier les dates d'observation pertinentes pour cette période
            relevant_obs_dates = []
            if i == 0:
                # Pour la première période, toutes les dates avant la première date de paiement
                relevant_obs_dates = [d for d in self._observation_dates if d <= payment_date]
            else:
                # Pour les autres périodes, toutes les dates entre les dates de paiement
                prev_payment_date = self._payment_dates[i-1]
                relevant_obs_dates = [d for d in self._observation_dates if prev_payment_date < d <= payment_date]
            
            n_obs_dates = len(relevant_obs_dates)
            
            if n_obs_dates > 0:
                # Coupon par jour d'observation dans la plage
                coupon_per_day = self._coupon_rate * self._nominal / n_obs_dates
                
                # Pour chaque date d'observation dans cette période
                for obs_date in relevant_obs_dates:
                    obs_maturity = Maturity(
                        start_date=self._pricing_date,
                        end_date=obs_date
                    )
                    
                    # Option digitale pour être au-dessus de la barrière basse
                    lower_digital = DigitalOption(
                        spot_price=spot_price,
                        strike_price=spot_price * self._lower_barrier,
                        maturity=obs_maturity,
                        domestic_rate=rate_model,
                        volatility=volatility,
                        option_type="call",  # Call car on veut être au-dessus
                        payment=coupon_per_day * 0.5,  # 50% de la valeur pour cette condition
                        dividend=0.0
                    )
                    
                    # Option digitale pour être en-dessous de la barrière haute
                    upper_digital = DigitalOption(
                        spot_price=spot_price,
                        strike_price=spot_price * self._upper_barrier,
                        maturity=obs_maturity,
                        domestic_rate=rate_model,
                        volatility=volatility,
                        option_type="put",  # Put car on veut être en-dessous
                        payment=coupon_per_day * 0.5,  # 50% de la valeur pour cette condition
                        dividend=0.0
                    )
                    
                    components.append(lower_digital)
                    components.append(upper_digital)

        return components
