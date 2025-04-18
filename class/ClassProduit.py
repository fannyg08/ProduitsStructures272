from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from datetime import datetime

from base.ClassMaturity import Maturity, DayCountConvention
from base.ClassRate import RateModel
from base.ClassVolatility import VolatilityModel

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


class CapitalProtectedNote(Product):
    """
    Note à capital protégé: combine protection du capital avec participation à la hausse.
    """
    def __init__(
        self, 
        underlying_id: str,
        maturity: Maturity,
        nominal: float,
        strike: float,
        participation_rate: float,
        capital_protection: float
    ):
        """
        Initialisation d'une note à capital protégé.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
            strike (float): Prix d'exercice du produit (niveau du strike)
            participation_rate (float): Taux de participation à la hausse
            capital_protection (float): Niveau de protection (en % du nominal)
        """
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._participation_rate = participation_rate
        self._capital_protection = capital_protection
        
    @property
    def strike(self) -> float:
        """Prix d'exercice."""
        return self._strike
    
    @property
    def participation_rate(self) -> float:
        """Taux de participation."""
        return self._participation_rate
    
    @property
    def capital_protection(self) -> float:
        """Niveau de protection du capital."""
        return self._capital_protection
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcule le payoff de la note à capital protégé.
        
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray, optional): Non utilisé dans ce cas
            
        Returns:
            ndarray: Vecteur des payoffs pour chaque simulation
        """
        final_prices = paths[:, -1]
        strike = self._strike
        performance = (final_prices - strike) / strike
        
        # Protection du capital + participation à la hausse
        payoffs = self._nominal * np.maximum(
            self._capital_protection, 
            1 + self._participation_rate * np.maximum(0, performance)
        )
        
        return payoffs


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
        """
        Initialisation d'une note autocall.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
            strike (float): Prix d'exercice du produit
            barriers (List[float]): Liste des barrières pour chaque date d'observation (en % du strike)
            coupon_rates (List[float]): Liste des taux de coupon pour chaque date d'observation
            observation_dates (List[float]): Liste des dates d'observation (en années)
            capital_protection (float, optional): Niveau de protection à l'échéance (en % du nominal)
        """
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._barriers = barriers
        self._coupon_rates = coupon_rates
        self._observation_dates = observation_dates
        self._capital_protection = capital_protection
        
    @property
    def strike(self) -> float:
        """Prix d'exercice."""
        return self._strike
    
    @property
    def barriers(self) -> List[float]:
        """Barrières d'autocall."""
        return self._barriers
    
    @property
    def coupon_rates(self) -> List[float]:
        """Taux de coupon."""
        return self._coupon_rates
    
    @property
    def observation_dates(self) -> List[float]:
        """Dates d'observation."""
        return self._observation_dates
    
    @property
    def capital_protection(self) -> Optional[float]:
        """Niveau de protection du capital."""
        return self._capital_protection
        
    def payoff(self, paths: np.ndarray, time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le payoff de la note autocall avec dates de paiement.
        
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray): Grille temporelle utilisée pour la simulation
            
        Returns:
            Tuple[ndarray, ndarray]: (payoffs, payment_times) avec leurs dates de paiement respectives
        """
        if time_grid is None:
            raise ValueError("time_grid est requis pour le calcul du payoff de l'AutocallNote")
            
        nb_simulations = paths.shape[0]
        payoffs = np.zeros(nb_simulations)
        redemption_times = np.ones(nb_simulations) * self._maturity.maturity_in_years
        
        # Convertir les dates d'observation en indices dans la grille temporelle
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
        """
        Initialisation d'une note à capital protégé avec barrière.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
            strike (float): Prix d'exercice du produit
            barrier (float): Niveau de la barrière
            participation_rate (float): Taux de participation
            capital_protection (float): Niveau de protection du capital (en % du nominal)
            rebate (float): Remise en cas de franchissement de la barrière (en % du nominal)
            barrier_direction (BarrierDirection): Direction de la barrière ("up" ou "down")
            barrier_type (BarrierType): Type de barrière ("ko" pour knock-out, "ki" pour knock-in)
        """
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
        """Prix d'exercice."""
        return self._strike
    
    @property
    def barrier(self) -> float:
        """Niveau de barrière."""
        return self._barrier
    
    @property
    def participation_rate(self) -> float:
        """Taux de participation."""
        return self._participation_rate
    
    @property
    def capital_protection(self) -> float:
        """Niveau de protection du capital."""
        return self._capital_protection
    
    @property
    def rebate(self) -> float:
        """Remise en cas de franchissement de barrière."""
        return self._rebate
    
    @property
    def barrier_direction(self) -> BarrierDirection:
        """Direction de la barrière."""
        return self._barrier_direction
    
    @property
    def barrier_type(self) -> BarrierType:
        """Type de barrière."""
        return self._barrier_type
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcule le payoff de la note à capital protégé avec barrière.
        
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray, optional): Non utilisé dans ce cas
            
        Returns:
            ndarray: Vecteur des payoffs pour chaque simulation
        """
        # Déterminer si la barrière a été franchie
        if self._barrier_direction == "up":
            barrier_hit = np.any(paths > self._barrier, axis=1)
        else:  # "down"
            barrier_hit = np.any(paths < self._barrier, axis=1)
            
        # Inverser la condition pour knock-in
        if self._barrier_type == "ki":
            barrier_hit = ~barrier_hit
            
        # Performances du sous-jacent
        final_prices = paths[:, -1]
        performance = (final_prices - self._strike) / self._strike
        
        # Payoff si la condition de barrière est satisfaite
        participation_payoff = self._nominal * (
            self._capital_protection + self._participation_rate * np.maximum(0, performance)
        )
        
        # Payoff si la condition de barrière n'est pas satisfaite
        protection_payoff = self._nominal * self._capital_protection
        
        # Ajout du rebate si applicable
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
        Initialisation d'une note Twin Win.
        
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
        """Prix d'exercice."""
        return self._strike
    
    @property
    def upper_barrier(self) -> float:
        """Barrière supérieure."""
        return self._upper_barrier
    
    @property
    def lower_barrier(self) -> float:
        """Barrière inférieure."""
        return self._lower_barrier
    
    @property
    def participation_rate_up(self) -> float:
        """Taux de participation à la hausse."""
        return self._participation_rate_up
    
    @property
    def participation_rate_down(self) -> float:
        """Taux de participation à la baisse."""
        return self._participation_rate_down
    
    @property
    def capital_protection(self) -> float:
        """Niveau de protection du capital."""
        return self._capital_protection
    
    @property
    def rebate(self) -> float:
        """Remise si une barrière est franchie."""
        return self._rebate
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcule le payoff de la note Twin Win.
        
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray, optional): Non utilisé dans ce cas
            
        Returns:
            ndarray: Vecteur des payoffs pour chaque simulation
        """
        # Vérifier si les barrières ont été franchies
        upper_hit = np.any(paths > self._upper_barrier, axis=1)
        lower_hit = np.any(paths < self._lower_barrier, axis=1)
        any_barrier_hit = upper_hit | lower_hit
        
        # Performance finale
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
        
        # Payoff final selon les conditions de barrière
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
        """
        Initialisation d'une note à capital protégé avec coupon.
        
        Args:
            underlying_id (str): Identifiant du sous-jacent
            maturity (Maturity): Objet représentant la maturité du produit
            nominal (float): Valeur nominale du produit
            strike (float): Prix d'exercice du produit
            coupon_rate (float): Taux de coupon (en % du nominal)
            coupon_cap (float): Limite maximale pour le coupon (en % du nominal)
            payment_dates (List[float]): Liste des dates de paiement des coupons (en années)
            capital_protection (float): Niveau de protection du capital (en % du nominal)
        """
        super().__init__(underlying_id, maturity, nominal)
        self._strike = strike
        self._coupon_rate = coupon_rate
        self._coupon_cap = coupon_cap
        self._payment_dates = payment_dates
        self._capital_protection = capital_protection
        
    @property
    def strike(self) -> float:
        """Prix d'exercice."""
        return self._strike
    
    @property
    def coupon_rate(self) -> float:
        """Taux de coupon."""
        return self._coupon_rate
    
    @property
    def coupon_cap(self) -> float:
        """Plafond du coupon."""
        return self._coupon_cap
    
    @property
    def payment_dates(self) -> List[float]:
        """Dates de paiement des coupons."""
        return self._payment_dates
    
    @property
    def capital_protection(self) -> float:
        """Niveau de protection du capital."""
        return self._capital_protection
        
    def payoff(self, paths: np.ndarray, time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le payoff de la note avec coupon.
        
        Args:
            paths (ndarray): Matrice de simulations de trajectoires (n_simulations x n_steps)
            time_grid (ndarray): Grille temporelle utilisée pour la simulation
            
        Returns:
            Tuple[ndarray, ndarray]: (payoffs, payment_times) - payoffs et leurs dates de paiement
        """
        if time_grid is None:
            raise ValueError("time_grid est requis pour le calcul du payoff de CapitalProtectedNoteWithCoupon")
        
        nb_simulations = paths.shape[0]
        
        # Tableau pour stocker tous les payoffs et leurs dates de paiement
        all_payoffs = []
        all_payment_times = []
        
        # Pour chaque date de paiement, calculer le coupon
        for payment_date in self._payment_dates:
            # Trouver l'indice correspondant à la date dans la grille temporelle
            time_idx = np.abs(time_grid - payment_date).argmin()
            
            # Performance à cette date
            prices_at_date = paths[:, time_idx]
            performance_at_date = (prices_at_date - self._strike) / self._strike
            
            # Coupon conditionnel (limité par le cap)
            coupon = self._nominal * np.minimum(
                self._coupon_rate * np.maximum(0, performance_at_date), 
                self._coupon_cap
            )
            
            # Ajouter le coupon et sa date à nos tableaux
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
