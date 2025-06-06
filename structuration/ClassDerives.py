from abc import ABC, abstractmethod
import numpy as np
from base.ClassMaturity import Maturity, OptionType
from base.ClassRate import Rate
from base.ClassOption import Option, Optional
from structuration.ClassVolatility import VolatilityModel

class EuropeanOption(Option):
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        maturity: Maturity,
        domestic_rate: Rate,
        volatility: VolatilityModel,
        option_type: str, 
        dividend: Optional[float] = None,
        foreign_rate: Optional[Rate] = None,
    ) -> None:
        """
        Option européenne (Call ou Put) sur un actif sous-jacent.
        """
        super().__init__(
            spot_price=spot_price,
            strike_price=strike_price,
            maturity=maturity,
            domestic_rate=domestic_rate,
            volatility=volatility,
            option_type=option_type,
            dividend=dividend,
            foreign_rate=foreign_rate,
        )

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff de l'option à maturité selon le type.
        """
        final_prices = paths[:, -1]
        if self._option_type == "call":
            return np.maximum(0, final_prices - self._strike_price)
        elif self._option_type == "put":
            return np.maximum(0, self._strike_price - final_prices)
        else:
            raise ValueError(f"Type d'option non reconnu : {self._option_type}")

    def compute_price(self):
        raise NotImplementedError("Utilisez un pricer pour calculer le prix.")

    def compute_greeks(self):
        raise NotImplementedError("Utilisez un pricer pour les grecques.")

    def compute_delta(self):
        raise NotImplementedError()

    def compute_gamma(self):
        raise NotImplementedError()

    def compute_vega(self):
        raise NotImplementedError()

    def compute_theta(self):
        raise NotImplementedError()

    def compute_rho(self):
        raise NotImplementedError()


class BarrierOption(Option):
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        maturity: Maturity,
        domestic_rate: Rate,
        volatility: VolatilityModel,
        option_type: OptionType,
        barrier: float,
        barrier_type: str,
        knock_type: str,
        dividend: Optional[float] = None,
        foreign_rate: Optional[Rate] = None,
    ) -> None:

        super().__init__(
            spot_price=spot_price,
            strike_price=strike_price,
            maturity=maturity,
            domestic_rate=domestic_rate,
            volatility=volatility,
            option_type=option_type,
            dividend=dividend,
            foreign_rate=foreign_rate,
        )
        self._barrier = barrier
        
        # Gestion des erreurs / Vérification des valeurs acceptées pour les types de barrière
        if barrier_type not in ['up', 'down']:
            raise ValueError("barrier_type doit être 'up' ou 'down'")
        self._barrier_type = barrier_type
        
        if knock_type not in ['in', 'out']:
            raise ValueError("knock_type doit être 'in' ou 'out'")
        self._knock_type = knock_type
    
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff de l'option à barrière.
        """
        final_prices = paths[:, -1]
        
        # Détermination si la barrière a été franchie pour chaque trajectoire
        if self._barrier_type == 'up':
            # Barrière haute: check si le prix a dépassé la barrière
            barrier_hit = np.any(paths >= self._barrier, axis=1)
        else:  # 'down'
            # Barrière basse: check si le prix est descendu sous la barrière
            barrier_hit = np.any(paths <= self._barrier, axis=1)
        
        # Calcul du payoff de base selon le type d'option
        if self._option_type == "call":
            base_payoff = np.maximum(0, final_prices - self._strike_price)
        else:  # PUT
            base_payoff = np.maximum(0, self._strike_price - final_prices)
        
        # Application de la condition de barrière
        if self._knock_type == 'in':
            # Knock-in: l'option est active uniquement si la barrière a été franchie
            payoff = np.where(barrier_hit, base_payoff, 0)
        else:  
            # Knock-out: l'option est active uniquement si la barrière n'a pas été franchie
            payoff = np.where(barrier_hit, 0, base_payoff)
        
        return payoff

class DigitalOption(Option):
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        maturity: Maturity,
        domestic_rate: Rate,
        volatility: VolatilityModel,
        option_type: OptionType,
        payment: float = 1.0,
        barrier: Optional[float] = None,
        barrier_type: Optional[str] = 'up',
        dividend: Optional[float] = None,
        foreign_rate: Optional[Rate] = None,
    ) -> None:
        """
        Option digitale (binaire) qui paie un montant fixe si une condition est remplie à l'échéance.
        Peut également inclure une condition de barrière.
        """
        super().__init__(
            spot_price=spot_price,
            strike_price=strike_price,
            maturity=maturity,
            domestic_rate=domestic_rate,
            volatility=volatility,
            option_type=option_type,
            dividend=dividend,
            foreign_rate=foreign_rate,
        )
        self._payment = payment
        self._barrier = barrier
        
        if barrier is not None and barrier_type not in ['up', 'down']:
            raise ValueError("barrier_type doit être 'up' ou 'down'")
        self._barrier_type = barrier_type
    
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff de l'option digitale.
        """
        final_prices = paths[:, -1]
        
        # Condition de payoff selon le type d'option
        if self._option_type == "call":
            condition = final_prices > self._strike_price
        else:  
            condition = final_prices < self._strike_price
        
        if self._barrier is None:
            # Option digitale simple
            return self._payment * condition
        else:
            # Option digitale avec barrière
            if self._barrier_type == 'up':
                # Barrière haute: check si le prix a dépassé la barrière
                barrier_hit = np.any(paths >= self._barrier, axis=1)
            else:  # 'down'
                # Barrière basse: check si le prix est descendu sous la barrière
                barrier_hit = np.any(paths <= self._barrier, axis=1)
            
            return self._payment * condition * barrier_hit