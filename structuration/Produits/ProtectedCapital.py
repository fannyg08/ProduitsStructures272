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



##### Produits avec protection du capital #####
class CapitalProtectedNote(DecomposableProduct):
    """
    Note à capital protégé avec participation à la hausse.
    """
    
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        participation_rate: float,
        capital_protection: float,
        rate_model: RateModel
    ):
        """
        Initialisation d'une note à capital protégé.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Maturité du produit
            nominal (float): Valeur nominale
            strike (float): Prix d'exercice
            participation_rate (float): Taux de participation (>0)
            capital_protection (float): Niveau de protection (entre 0 et 1)
            rate_model (RateModel): Modèle de taux pour la composante obligataire
        """
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._participation_rate = participation_rate
        self._capital_protection = capital_protection
        self._rate_model = rate_model
    
    def decompose(self) -> List[Union[ABCBond, Option, Product]]:
        """
        Décompose la note à capital protégé en une obligation et une option.
        
        Returns:
            List[Union[ABCBond, Option, Product]]: Liste des composantes
        """
        # Composante obligataire (capital protégé)
        protected_amount = self._nominal * self._capital_protection
        bond = ZeroCouponBond(
            rate_model=self._rate_model,
            maturity=self._maturity,
            nominal=protected_amount
        )
        
        # Composante optionnelle (participation à la hausse)
        option_nominal = self._nominal * (1 - self._capital_protection) * self._participation_rate
        option = Option(
            spot_price=0, 
            strike_price=self._strike,
            maturity=self._maturity,
            domestic_rate=self._rate_model,
            volatility=0,  
            option_type="call",
            nominal=option_nominal
        )
        
        return [bond, option]
    
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcule le payoff de la note à capital protégé.
        
        Args:
            paths (ndarray): Trajectoires simulées
            time_grid (ndarray, optional): Grille temporelle
            
        Returns:
            ndarray: Payoffs du produit
        """
        # Retourne le capital protégé + participation à la hausse, s'il y en a
        final_prices = paths[:, -1]
        protected_amount = self._nominal * self._capital_protection
        participation = self._nominal * (1 - self._capital_protection) * self._participation_rate * np.maximum(0, final_prices / self._strike - 1)
        
        return protected_amount + participation

class AutocallNote(Product):

    """
    Note autocall: produit structuré qui peut être rappelé avant l'échéance
    si certaines conditions sont remplies aux dates d'observation.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        barriers: List[float],
        coupon_rates: List[float],
        observation_dates: List[float],
        capital_protection: Optional[float] = None
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._barriers = barriers
        self._coupon_rates = coupon_rates
        self._observation_dates = observation_dates
        self._capital_protection = capital_protection
        
    @property
    def strike(self) -> float:
        return self._strike
    
    @property
    def barriers(self) -> List[float]:
        return self._barriers
    
    @property
    def coupon_rates(self) -> List[float]:
        return self._coupon_rates
    
    @property
    def observation_dates(self) -> List[float]:
        return self._observation_dates
    
    @property
    def capital_protection(self) -> Optional[float]:
        return self._capital_protection
        
    def payoff(self, paths: np.ndarray, time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le payoff de la note autocall avec dates de paiement.
        """
        if time_grid is None:
            raise ValueError("time_grid est requis pour le calcul du payoff de l'AutocallNote")
            
        nb_simulations = paths.shape[0]
        payoffs = np.zeros(nb_simulations)
        redemption_times = np.ones(nb_simulations) * self._maturity.maturity_in_years

        observation_indices = [np.abs(time_grid - date).argmin() for date in self._observation_dates]
        
        # Vérifier les conditions d'autocall à chaque date d'observation
        for i, (obs_idx, barrier, coupon) in enumerate(zip(observation_indices, self._barriers, self._coupon_rates)):
            obs_date = self._observation_dates[i]
            
            # Condition d'autocall: sous-jacent >= barrière * strike
            autocall_condition = paths[:, obs_idx] >= barrier * self._strike
            
            # Pour les simulations qui n'ont pas encore été remboursées et remplissent la condition
            not_redeemed = redemption_times == self._maturity.maturity_in_years
            to_redeem = not_redeemed & autocall_condition
            
            # Mise à jour des payoffs et des dates de remboursement
            payoffs[to_redeem] = self._nominal * (1 + coupon)
            redemption_times[to_redeem] = obs_date
        
        # Pour les simulations qui atteignent la maturité
        at_maturity = redemption_times == self._maturity.maturity_in_years
        final_prices = paths[at_maturity, -1]
        
        if self._capital_protection:
            # Avec protection du capital
            final_performance = (final_prices - self._strike) / self._strike
            payoffs[at_maturity] = self._nominal * np.maximum(
                self._capital_protection,
                1 + np.maximum(0, final_performance)  # Participation à la hausse
            )
        else:
            # Sans protection du capital
            payoffs[at_maturity] = self._nominal * final_prices / self._strike
            
        return payoffs, redemption_times

class CapitalProtectedNoteWithBarrier(Product):
    """
    Note à capital protégé avec barrière: combine protection du capital avec
    participation conditionnelle à la hausse selon qu'une barrière est franchie ou non.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        barrier: float,
        participation_rate: float,
        capital_protection: float,
        rebate: float = 0.0,
        barrier_direction: BarrierDirection = "up",
        barrier_type: BarrierType = "ko"
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._barrier = barrier
        self._participation_rate = participation_rate
        self._capital_protection = capital_protection
        self._rebate = rebate
        self._barrier_direction = barrier_direction
        self._barrier_type = barrier_type
        
    @property
    def strike(self) -> float:
        return self._strike
    
    @property
    def barrier(self) -> float:
        return self._barrier
    
    @property
    def participation_rate(self) -> float:
        return self._participation_rate
    
    @property
    def capital_protection(self) -> float:
        return self._capital_protection
    
    @property
    def rebate(self) -> float:
        return self._rebate
    
    @property
    def barrier_direction(self) -> BarrierDirection:
        return self._barrier_direction
    
    @property
    def barrier_type(self) -> BarrierType:
        return self._barrier_type
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        # Déterminer si la barrière a été franchie
        if self._barrier_direction == "up":
            barrier_hit = np.any(paths > self._barrier, axis=1)
        else:  
            barrier_hit = np.any(paths < self._barrier, axis=1)
            
        # Inverser la condition pour knock-in
        if self._barrier_type == "ki":
            barrier_hit = ~barrier_hit


        final_prices = paths[:, -1]
        performance = (final_prices - self._strike) / self._strike
        
        # Payoff si la condition de barrière est satisfaite
        participation_payoff = self._nominal * (
            self._capital_protection + self._participation_rate * np.maximum(0, performance)
        )
        
        # Payoff si la condition de barrière n'est pas satisfaite
        protection_payoff = self._nominal * self._capital_protection
        
        rebate_amount = self._nominal * self._rebate
        
        # Combinaison des payoffs selon la condition de barrière
        final_payoff = np.where(barrier_hit, 
                             participation_payoff + rebate_amount, 
                             protection_payoff)
        
        return final_payoff

class CapitalProtectedNoteTwinWin(Product):
    """
    Note à capital protégé avec Twin Win: participation à la hausse et à la baisse,
    tant que certaines barrières ne sont pas franchies.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        upper_barrier: float,
        lower_barrier: float,
        participation_rate_up: float,
        participation_rate_down: float,
        capital_protection: float,
        rebate: float = 0.0
    ):
        """
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
            strike (float): Prix d'exercice du produit
            upper_barrier (float): Barrière supérieure
            lower_barrier (float): Barrière inférieure
            participation_rate_up (float): Taux de participation à la hausse
            participation_rate_down (float): Taux de participation à la baisse (absolu)
            capital_protection (float): Niveau de protection du capital (en % du nominal)
            rebate (float): Remise si une barrière est franchie (en % du nominal)
        """
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._upper_barrier = upper_barrier
        self._lower_barrier = lower_barrier
        self._participation_rate_up = participation_rate_up
        self._participation_rate_down = participation_rate_down
        self._capital_protection = capital_protection
        self._rebate = rebate
        
    @property
    def strike(self) -> float:
        return self._strike
    
    @property
    def upper_barrier(self) -> float:
        return self._upper_barrier
    
    @property
    def lower_barrier(self) -> float:
        return self._lower_barrier
    
    @property
    def participation_rate_up(self) -> float:
        return self._participation_rate_up
    
    @property
    def participation_rate_down(self) -> float:
        return self._participation_rate_down
    
    @property
    def capital_protection(self) -> float:
        return self._capital_protection
    
    @property
    def rebate(self) -> float:
        return self._rebate
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        # Vérifier si les barrières ont été franchies
        upper_hit = np.any(paths > self._upper_barrier, axis=1)
        lower_hit = np.any(paths < self._lower_barrier, axis=1)
        any_barrier_hit = upper_hit | lower_hit
        
        final_prices = paths[:, -1]
        performance = (final_prices - self._strike) / self._strike
        
        # Payoff pour la participation à la hausse (si performance > 0)
        up_payoff = self._nominal * (self._capital_protection + 
                                   self._participation_rate_up * np.maximum(0, performance))
        
        # Payoff pour la participation à la baisse (si performance < 0)
        down_payoff = self._nominal * (self._capital_protection + 
                                     self._participation_rate_down * np.maximum(0, -performance))
        
        # Combiner les payoffs selon la performance
        performance_payoff = np.where(performance >= 0, up_payoff, down_payoff)
        
        # Payoff si une barrière est franchie
        barrier_payoff = self._nominal * self._capital_protection + self._nominal * self._rebate
        

        final_payoff = np.where(any_barrier_hit, barrier_payoff, performance_payoff)
        
        return final_payoff

class CapitalProtectedNoteWithCoupon(Product):
    """
    Note à capital protégé avec coupon: offre la protection du capital et
    des coupons conditionnels basés sur la performance du sous-jacent.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        coupon_rate: float,
        coupon_cap: float,
        payment_dates: List[float],
        capital_protection: float
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._coupon_rate = coupon_rate
        self._coupon_cap = coupon_cap
        self._payment_dates = payment_dates
        self._capital_protection = capital_protection
        
    @property
    def strike(self) -> float:
        return self._strike
    
    @property
    def coupon_rate(self) -> float:
        return self._coupon_rate
    
    @property
    def coupon_cap(self) -> float:
        return self._coupon_cap
    
    @property
    def payment_dates(self) -> List[float]:
        return self._payment_dates
    
    @property
    def capital_protection(self) -> float:
        return self._capital_protection
        
    def payoff(self, paths: np.ndarray, time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if time_grid is None:
            raise ValueError("time_grid est requis pour le calcul du payoff de CapitalProtectedNoteWithCoupon")
        
        nb_simulations = paths.shape[0]
        
        all_payoffs = []
        all_payment_times = []
        
        # Pour chaque date de paiement, calculer le coupon
        for payment_date in self._payment_dates:
            time_idx = np.abs(time_grid - payment_date).argmin()
            
            prices_at_date = paths[:, time_idx]
            performance_at_date = (prices_at_date - self._strike) / self._strike
            
            # Coupon conditionnel (limité par le cap)
            coupon = self._nominal * np.minimum(
                self._coupon_rate * np.maximum(0, performance_at_date), 
                self._coupon_cap
            )
            
            all_payoffs.append(coupon)
            all_payment_times.append(np.full(nb_simulations, payment_date))
        
        # Ajouter le remboursement du capital à l'échéance
        capital_payoff = self._nominal * self._capital_protection
        all_payoffs.append(np.full(nb_simulations, capital_payoff))
        all_payment_times.append(np.full(nb_simulations, self._maturity.maturity_in_years))
        
        # Concaténer tous les payoffs et leurs dates
        payoffs = np.concatenate(all_payoffs)
        payment_times = np.concatenate(all_payment_times)
        
        return payoffs, payment_times

