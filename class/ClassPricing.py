from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass
from ClassProduit import Product,CapitalProtectedNoteTwinWin,CapitalProtectedNote, CapitalProtectedNoteWithBarrier, CapitalProtectedNoteWithCoupon


class PricingEngine:
    def __init__(self, product, nb_simulations=10000, time_steps=252, seed=None):
        self.product = product
        self.nb_simulations = nb_simulations
        self.time_steps = time_steps
        self.seed = seed
        
    def calculate_price(self):
        """Calcule le prix du produit par simulation Monte Carlo"""
        # Créer la grille temporelle
        time_grid = np.linspace(0, self.product.maturity, self.time_steps + 1)
        
        # Simuler les trajectoires du sous-jacent
        paths = self.product.underlying.simulate_paths(time_grid, self.nb_simulations, self.seed)
        
        # Calculer les payoffs avec l'interface standardisée
        payoffs, payment_times = self.product.payoff(paths, time_grid)
        
        # Si payment_times est None, utiliser la maturité du produit
        if payment_times is None:
            payment_times = np.full(payoffs.shape, self.product.maturity)
        
        # Actualiser les payoffs à leurs dates de paiement respectives
        risk_free_rate = self.product.underlying.market_data.risk_free_rate
        discount_factors = np.exp(-risk_free_rate * payment_times)
        discounted_payoffs = payoffs * discount_factors
        
        # Prix = moyenne des payoffs actualisés
        price = np.mean(discounted_payoffs)
        
        return price
    
    def calculate_greeks(self, epsilon=0.01):
        """Calcule les principaux Greeks par différences finies"""
        market_data = self.product.underlying.market_data
        original_price = self.calculate_price()
        
        # Delta
        original_spot = market_data.spot_price
        market_data.spot_price = original_spot * (1 + epsilon)
        up_price = self.calculate_price()
        market_data.spot_price = original_spot * (1 - epsilon)
        down_price = self.calculate_price()
        market_data.spot_price = original_spot
        delta = (up_price - down_price) / (2 * epsilon * original_spot)
        
        # Vega
        original_vol = market_data.volatility
        market_data.volatility = original_vol + epsilon
        up_price = self.calculate_price()
        market_data.volatility = original_vol - epsilon
        down_price = self.calculate_price()
        market_data.volatility = original_vol
        vega = (up_price - down_price) / (2 * epsilon)
        
        # Theta
        # Réduire la maturité d'une petite quantité
        original_maturity = self.product.maturity
        self.product.maturity -= epsilon
        time_grid = np.linspace(0, self.product.maturity, self.time_steps + 1)
        paths = self.product.underlying.simulate_paths(time_grid, self.nb_simulations, self.seed)
        if isinstance(self.product, AutocallNote):
            payoffs, redemption_times = self.product.payoff(paths)
            discount_factors = np.exp(-market_data.risk_free_rate * redemption_times)
            discounted_payoffs = payoffs * discount_factors
        else:
            payoffs = self.product.payoff(paths)
            discount_factor = np.exp(-market_data.risk_free_rate * self.product.maturity)
            discounted_payoffs = payoffs * discount_factor
        reduced_price = np.mean(discounted_payoffs)
        self.product.maturity = original_maturity
        theta = (reduced_price - original_price) / epsilon
        
        return {
            'delta': delta,
            'vega': vega,
            'theta': theta
        }