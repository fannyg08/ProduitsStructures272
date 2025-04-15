from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class MarketData:
    spot_price: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float
    
class Underlying:
    def __init__(self, name, market_data):
        self.name = name
        self.market_data = market_data
        
    def simulate_paths(self, time_grid, nb_simulations, seed=None):
        """Génère des simulations de trajectoires du sous-jacent"""
        if seed:
            np.random.seed(seed)
            
        dt = time_grid[1] - time_grid[0]
        num_steps = len(time_grid) - 1
        
        paths = np.zeros((nb_simulations, len(time_grid)))
        paths[:, 0] = self.market_data.spot_price
        
        for t in range(1, len(time_grid)):
            z = np.random.normal(0, 1, nb_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.market_data.risk_free_rate - self.market_data.dividend_yield - 0.5 * self.market_data.volatility**2) * dt 
                + self.market_data.volatility * np.sqrt(dt) * z
            )
            
        return paths

