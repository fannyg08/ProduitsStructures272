from abc import ABC, abstractmethod
import numpy as np


class Option(ABC):
    def __init__(self, underlying, strike, maturity):
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
    
    @abstractmethod
    def payoff(self, paths):
        pass


class CallOption(Option):
    def payoff(self, paths):
        final_prices = paths[:, -1]
        return np.maximum(0, final_prices - self.strike)


class PutOption(Option):
    def payoff(self, paths):
        final_prices = paths[:, -1]
        return np.maximum(0, self.strike - final_prices)


class BarrierOption(Option):
    def __init__(self, underlying, strike, maturity, barrier, barrier_type, knock_type):
        """
        Parameters:
        -----------
        barrier: float
            Niveau de barrière
        barrier_type: str
            'up' ou 'down'
        knock_type: str
            'in' ou 'out'
        """
        super().__init__(underlying, strike, maturity)
        self.barrier = barrier
        self.barrier_type = barrier_type  # 'up' ou 'down'
        self.knock_type = knock_type  # 'in' ou 'out'
    
    def payoff(self, paths):
        final_prices = paths[:, -1]
        
        # Vérifier si la barrière a été franchie
        if self.barrier_type == 'up':
            barrier_hit = np.any(paths > self.barrier, axis=1)
        else:  # 'down'
            barrier_hit = np.any(paths < self.barrier, axis=1)
        
        # Payoff conditionnel selon le type de knock
        if self.knock_type == 'in':
            # Le payoff existe uniquement si la barrière a été franchie
            payoffs = np.maximum(0, final_prices - self.strike) * barrier_hit
        else:  # 'out'
            # Le payoff existe uniquement si la barrière n'a pas été franchie
            payoffs = np.maximum(0, final_prices - self.strike) * (~barrier_hit)
            
        return payoffs


class DigitalOption(Option):
    def __init__(self, underlying, strike, maturity, barrier=None, payment=1.0):
        super().__init__(underlying, strike, maturity)
        self.barrier = barrier  # Optionnel pour les options digitales avec barrière
        self.payment = payment
    
    def payoff(self, paths):
        final_prices = paths[:, -1]
        
        if self.barrier is None:
            # Option digitale simple
            return self.payment * (final_prices > self.strike)
        else:
            # Option digitale avec barrière
            barrier_hit = np.any(paths > self.barrier, axis=1)
            return self.payment * (final_prices > self.strike) * barrier_hit
