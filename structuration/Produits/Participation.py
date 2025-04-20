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



class ParticipationProduct(Product):
    """
    Classe intermédiaire pour les produits de participation.
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        participation_rate: float = 1.0,
        nominal: float = 1000.0
    ):
        super().__init__(underlying_id, maturity, nominal)
        self._participation_rate = participation_rate
        
    @property
    def participation_rate(self) -> float:
        """Taux de participation au rendement du sous-jacent."""
        return self._participation_rate

#### Produits de participation ##### 
# Tracker certificate 
# Bonus certificate 
# outperformance certificate 
class TrackerCertificate(ParticipationProduct):
    """
    Certificat Tracker qui suit la performance du sous-jacent avec un ratio de participation.
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        participation_rate: float = 1.0,
        nominal: float = 1000.0,
        management_fee: float = 0.0 
    ):
        super().__init__(underlying_id, maturity, participation_rate, nominal)
        self._management_fee = management_fee
        
    @property
    def management_fee(self) -> float:
        """Frais de gestion annuels en pourcentage."""
        return self._management_fee
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        # Récupération des valeurs initiales et finales
        initial_values = paths[:, 0]
        final_values = paths[:, -1]
        
        # Calcul de la performance relative
        performance = final_values / initial_values - 1.0
        
        # Application du taux de participation
        participation_performance = performance * self.participation_rate
        
        # Application des frais de gestion (approximation simple)
        if time_grid is not None:
            years_to_maturity = time_grid[-1]
            fee_factor = (1 - self._management_fee) ** years_to_maturity
        else:
            fee_factor = 1 - self._management_fee
            
        # Calcul du payoff final
        payoffs = self.nominal * (1 + participation_performance) * fee_factor
        
        return payoffs
    

class BonusCertificate(ParticipationProduct):
    """
    Certificat Bonus qui offre une protection conditionnelle et un bonus si une barrière n'est pas franchie.
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        barrier_level: float,  # Niveau de barrière en pourcentage du prix initial
        bonus_level: float,    # Niveau de bonus en pourcentage du prix initial
        participation_rate: float = 1.0,
        nominal: float = 1000.0
    ):
        super().__init__(underlying_id, maturity, participation_rate, nominal)
        self._barrier_level = barrier_level
        self._bonus_level = bonus_level
        
    @property
    def barrier_level(self) -> float:
        """Niveau de barrière en pourcentage du prix initial."""
        return self._barrier_level
        
    @property
    def bonus_level(self) -> float:
        """Niveau de bonus en pourcentage du prix initial."""
        return self._bonus_level
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        initial_values = paths[:, 0]
        final_values = paths[:, -1]
        
        # Vérification si la barrière a été touchée (minimum sur toute la trajectoire)
        min_values = np.min(paths, axis=1)
        barrier_touched = min_values <= (initial_values * self._barrier_level)
        
        performance = final_values / initial_values - 1.0
        
        # Payoffs selon si la barrière a été touchée ou non
        payoffs = np.zeros_like(final_values)
        
        # Si la barrière a été touchée: participation directe à la performance
        payoffs[barrier_touched] = self.nominal * (1 + performance[barrier_touched] * self.participation_rate)
        
        # Si la barrière n'a pas été touchée: max(performance, bonus_level)
        bonus_return = self._bonus_level - 1.0  # Conversion du niveau de bonus en rendement
        barrier_not_touched = ~barrier_touched
        payoffs[barrier_not_touched] = self.nominal * (1 + np.maximum(
            performance[barrier_not_touched] * self.participation_rate,
            bonus_return
        ))
        
        return payoffs

class OutperformanceCertificate(ParticipationProduct):
    """
    Certificat Outperformance qui offre une participation accrue à la hausse du sous-jacent.
    """
    def __init__(
        self,
        underlying_id: str,
        maturity: Maturity,
        upside_participation_rate: float, 
        downside_participation_rate: float = 1.0, 
        strike_level: float = 1.0,  # Niveau de strike en % du prix initial
        nominal: float = 1000.0
    ):
        super().__init__(underlying_id, maturity, 1.0, nominal)  # Le taux de participation de base est redéfini
        self._upside_participation_rate = upside_participation_rate
        self._downside_participation_rate = downside_participation_rate
        self._strike_level = strike_level
        
    @property
    def upside_participation_rate(self) -> float:
        """Taux de participation à la hausse."""
        return self._upside_participation_rate
    
    @property
    def downside_participation_rate(self) -> float:
        """Taux de participation à la baisse."""
        return self._downside_participation_rate
    
    @property
    def strike_level(self) -> float:
        """Niveau de strike en pourcentage du prix initial."""
        return self._strike_level
        
    def payoff(self, paths: np.ndarray, time_grid: Optional[np.ndarray] = None) -> np.ndarray:
        initial_values = paths[:, 0]
        final_values = paths[:, -1]
        
        performance = final_values / initial_values - 1.0
        
        strike_return = self._strike_level - 1.0
        
        # Payoffs selon performance par rapport au strike
        payoffs = np.zeros_like(final_values)
        
        # Performance supérieure au strike
        above_strike = performance > strike_return
        payoffs[above_strike] = self.nominal * (1 + strike_return + 
                                               (performance[above_strike] - strike_return) * self._upside_participation_rate)
        
        # Performance inférieure ou égale au strike
        below_strike = ~above_strike
        payoffs[below_strike] = self.nominal * (1 + performance[below_strike] * self._downside_participation_rate)
        
        return payoffs
    
