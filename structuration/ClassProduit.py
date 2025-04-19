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

# Il existe plusieurs grands "types" de produits structurés : les produits avec protection du capital, 
# les produits de rendements, les produits de participation et enfin les produits plus complexes
# Nous avons essayé d'en coder plusieurs, de chaque catégorie, pour refléter leur diversité. 
# Nous avons mis en place des classes intermédiaires, propres aux types de produits.
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

#### Produits de rendements ####


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