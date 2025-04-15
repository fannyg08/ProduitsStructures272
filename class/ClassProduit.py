
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass
from ClassDerives import PutOption, CallOption, BarrierOption,DigitalOption


class Product(ABC):
    def __init__(self, underlying, maturity):
        self.underlying = underlying
        self.maturity = maturity
    
    @abstractmethod
    def payoff(self, paths, time_grid=None):
        """
        Calcule le payoff du produit
        
        Parameters:
        -----------
        paths: ndarray
            Les trajectoires simulées du sous-jacent
        time_grid: ndarray, optional
            La grille temporelle utilisée pour la simulation
            
        Returns:
        --------
        tuple: (payoffs, payment_times)
            - payoffs: ndarray des payoffs pour chaque simulation
            - payment_times: ndarray des temps de paiement (maturity par défaut)
        """
        pass

    
class CapitalProtectedNote(Product):
    def __init__(self, underlying, maturity, nominal, strike, participation_rate, capital_protection):
        """
        Parameters:
        -----------
        underlying: Underlying
            Le sous-jacent du produit
        maturity: float
            Maturité en années
        nominal: float
            Valeur nominale du produit
        strike: float
            Prix d'exercice
        participation_rate: float
            Taux de participation (en pourcentage)
        capital_protection: float
            Niveau de protection (en pourcentage du nominal)
        """
        super().__init__(underlying, maturity)
        self.nominal = nominal
        self.strike = strike
        self.participation_rate = participation_rate
        self.capital_protection = capital_protection
        
    def payoff(self, paths):
        """
        Calcule le payoff de la note à capital protégé
        
        Parameters:
        -----------
        paths: ndarray
            Array de simulations de trajectoires de sous-jacent
            
        Returns:
        --------
        ndarray: Payoff pour chaque simulation
        """
        final_prices = paths[:, -1]
        performance = (final_prices - self.strike) / self.strike
        
        # Protection du capital + participation à la hausse
        payoffs = self.nominal * np.maximum(
            self.capital_protection, 
            1 + self.participation_rate * np.maximum(0, performance)
        )
        
        return payoffs

class AutocallNote(Product):
    def __init__(self, underlying, maturity, nominal, strike, barriers, coupon_rates, observation_dates, capital_protection=None):
        """
        Parameters:
        -----------
        underlying: Underlying
            Le sous-jacent du produit
        maturity: float
            Maturité en années
        nominal: float
            Valeur nominale du produit
        strike: float
            Prix d'exercice
        barriers: list
            Liste des barrières pour chaque date d'observation (en % du strike)
        coupon_rates: list
            Liste des taux de coupon pour chaque date d'observation
        observation_dates: list
            Liste des dates d'observation (en années)
        capital_protection: float, optional
            Niveau de protection à l'échéance (en pourcentage du nominal)
        """
        super().__init__(underlying, maturity)
        self.nominal = nominal
        self.strike = strike
        self.barriers = barriers
        self.coupon_rates = coupon_rates
        self.observation_dates = observation_dates
        self.capital_protection = capital_protection
        
    def payoff(self, paths):
        """
        Calcule le payoff de la note autocall
        
        Parameters:
        -----------
        paths: ndarray
            Array de simulations de trajectoires de sous-jacent
            
        Returns:
        --------
        ndarray: Payoff pour chaque simulation et date de remboursement
        """
        nb_simulations = paths.shape[0]
        payoffs = np.zeros(nb_simulations)
        redemption_times = np.ones(nb_simulations) * self.maturity
        
        # Convertir les dates d'observation en indices
        observation_indices = [np.abs(time_grid - date).argmin() for date in self.observation_dates]
        
        # Vérifier les conditions d'autocall à chaque date d'observation
        for i, (obs_idx, barrier, coupon) in enumerate(zip(observation_indices, self.barriers, self.coupon_rates)):
            obs_date = self.observation_dates[i]
            
            # Condition d'autocall: sous-jacent >= barrière * strike
            autocall_condition = paths[:, obs_idx] >= barrier * self.strike
            
            # Pour les simulations qui n'ont pas encore été remboursées et remplissent la condition
            not_redeemed = redemption_times == self.maturity
            to_redeem = not_redeemed & autocall_condition
            
            # Mise à jour des payoffs et des dates de remboursement
            payoffs[to_redeem] = self.nominal * (1 + coupon)
            redemption_times[to_redeem] = obs_date
        
        # Pour les simulations qui atteignent la maturité
        at_maturity = redemption_times == self.maturity
        final_prices = paths[at_maturity, -1]
        
        if self.capital_protection:
            # Avec protection du capital
            final_performance = (final_prices - self.strike) / self.strike
            payoffs[at_maturity] = self.nominal * np.maximum(
                self.capital_protection,
                1 + np.maximum(0, final_performance)  # Participation à la hausse
            )
        else:
            # Sans protection du capital
            payoffs[at_maturity] = self.nominal * final_prices / self.strike
            
        return payoffs, redemption_times
        


class CapitalProtectedNoteWithBarrier(Product):
    def __init__(self, underlying, maturity, nominal, strike, barrier, participation_rate, capital_protection, rebate=0):
        """
        Note à capital protégé avec barrière (1130)
        
        Parameters:
        -----------
        underlying: Underlying
            Le sous-jacent du produit
        maturity: float
            Maturité en années
        nominal: float
            Valeur nominale du produit
        strike: float
            Prix d'exercice
        barrier: float
            Niveau de barrière
        participation_rate: float
            Taux de participation (en pourcentage)
        capital_protection: float
            Niveau de protection (en pourcentage du nominal)
        rebate: float
            Remise en cas de franchissement de la barrière (en pourcentage du nominal)
        """
        super().__init__(underlying, maturity)
        self.nominal = nominal
        self.strike = strike
        self.barrier = barrier
        self.participation_rate = participation_rate
        self.capital_protection = capital_protection
        self.rebate = rebate
        
        # Créer des options pour modéliser les composantes du produit
        self.call_option = CallOption(underlying, strike, maturity)
        self.barrier_option = BarrierOption(underlying, strike, maturity, barrier, 'up', 'out')
        
    def payoff(self, paths, time_grid=None):
        """Calcule le payoff de la note à capital protégé avec barrière"""
        # Vérifier si la barrière a été franchie durant la vie du produit
        barrier_hit = np.any(paths > self.barrier, axis=1)
        
        # Performances du sous-jacent
        final_prices = paths[:, -1]
        performance = (final_prices - self.strike) / self.strike
        
        # Payoffs conditionnels
        # 1. Si barrière non franchie: protection du capital
        payoff_if_no_barrier = self.nominal * self.capital_protection
        
        # 2. Si barrière franchie: protection + participation à la hausse + rebate
        participation_payoff = self.nominal * (self.capital_protection + 
                                              self.participation_rate * np.maximum(0, performance))
        rebate_payoff = self.nominal * self.rebate
        payoff_if_barrier = np.where(barrier_hit, 
                                     participation_payoff + rebate_payoff, 
                                     payoff_if_no_barrier)
        
        # Retourne les payoffs et None pour indiquer que tous les paiements sont à maturité
        return payoff_if_barrier, None


class CapitalProtectedNoteTwinWin(Product):
    def __init__(self, underlying, maturity, nominal, strike, upper_barrier, lower_barrier, 
                 participation_rate_up, participation_rate_down, capital_protection, rebate=0):
        """
        Note à capital protégé avec Twin Win (1135)
        
        Parameters:
        -----------
        underlying: Underlying
            Le sous-jacent du produit
        maturity: float
            Maturité en années
        nominal: float
            Valeur nominale du produit
        strike: float
            Prix d'exercice
        upper_barrier: float
            Barrière supérieure
        lower_barrier: float
            Barrière inférieure
        participation_rate_up: float
            Taux de participation à la hausse
        participation_rate_down: float
            Taux de participation à la baisse (absolu)
        capital_protection: float
            Niveau de protection (en pourcentage du nominal)
        rebate: float
            Remise si une barrière est franchie
        """
        super().__init__(underlying, maturity)
        self.nominal = nominal
        self.strike = strike
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.participation_rate_up = participation_rate_up
        self.participation_rate_down = participation_rate_down
        self.capital_protection = capital_protection
        self.rebate = rebate
        
        # Créer des options pour modéliser les composantes
        self.call_option = CallOption(underlying, strike, maturity)
        self.put_option = PutOption(underlying, strike, maturity)
        self.up_barrier_option = BarrierOption(underlying, strike, maturity, upper_barrier, 'up', 'out')
        self.down_barrier_option = BarrierOption(underlying, strike, maturity, lower_barrier, 'down', 'out')
        
    def payoff(self, paths, time_grid=None):
        """Calcule le payoff de la note Twin Win"""
        # Vérifier si les barrières ont été franchies
        upper_hit = np.any(paths > self.upper_barrier, axis=1)
        lower_hit = np.any(paths < self.lower_barrier, axis=1)
        any_barrier_hit = upper_hit | lower_hit
        
        # Performance finale
        final_prices = paths[:, -1]
        performance = (final_prices - self.strike) / self.strike
        
        # Payoff pour la participation à la hausse (si performance > 0)
        up_payoff = self.nominal * (self.capital_protection + 
                                   self.participation_rate_up * np.maximum(0, performance))
        
        # Payoff pour la participation à la baisse (si performance < 0)
        down_payoff = self.nominal * (self.capital_protection + 
                                     self.participation_rate_down * np.maximum(0, -performance))
        
        # Combiner les payoffs selon la performance
        performance_payoff = np.where(performance >= 0, up_payoff, down_payoff)
        
        # Payoff si une barrière est franchie
        barrier_payoff = self.nominal * self.capital_protection + self.nominal * self.rebate
        
        # Payoff final selon les conditions de barrière
        final_payoff = np.where(any_barrier_hit, barrier_payoff, performance_payoff)
        
        # Retourne les payoffs et None pour indiquer que tous les paiements sont à maturité
        return final_payoff, None



class CapitalProtectedNoteWithCoupon(Product):
    def __init__(self, underlying, maturity, nominal, strike, coupon_rate, 
                 coupon_cap, payment_dates, capital_protection):
        """
        Note à capital protégé avec coupon (1140)
        
        Parameters:
        -----------
        underlying: Underlying
            Le sous-jacent du produit
        maturity: float
            Maturité en années
        nominal: float
            Valeur nominale du produit
        strike: float
            Prix d'exercice
        coupon_rate: float
            Taux de coupon (en pourcentage du nominal)
        coupon_cap: float
            Limite maximale pour le coupon (en pourcentage du nominal)
        payment_dates: list
            Liste des dates de paiement des coupons (en années)
        capital_protection: float
            Niveau de protection (en pourcentage du nominal)
        """
        super().__init__(underlying, maturity)
        self.nominal = nominal
        self.strike = strike
        self.coupon_rate = coupon_rate
        self.coupon_cap = coupon_cap
        self.payment_dates = payment_dates
        self.capital_protection = capital_protection
        
        # Créer des options pour modéliser les composantes
        self.call_option = CallOption(underlying, strike, maturity)
        self.digital_options = []
        
        # Créer une option digitale pour chaque date de paiement
        for date in payment_dates:
            self.digital_options.append(DigitalOption(underlying, strike, date, payment=coupon_rate))
            
    def payoff(self, paths, time_grid=None):
        """Calcule le payoff de la note avec coupon"""
        if time_grid is None:
            raise ValueError("time_grid est requis pour CapitalProtectedNoteWithCoupon")
        
        nb_simulations = paths.shape[0]
        
        # Tableau pour stocker tous les payoffs et leurs dates de paiement
        all_payoffs = []
        all_payment_times = []
        
        # Pour chaque date de paiement, calculer le coupon
        for payment_date in self.payment_dates:
            # Trouver l'indice correspondant à la date dans la grille temporelle
            time_idx = np.abs(time_grid - payment_date).argmin()
            
            # Performance à cette date
            prices_at_date = paths[:, time_idx]
            performance_at_date = (prices_at_date - self.strike) / self.strike
            
            # Coupon conditionnel (limité par le cap)
            coupon = self.nominal * np.minimum(self.coupon_rate * np.maximum(0, performance_at_date), 
                                              self.coupon_cap)
            
            # Ajouter le coupon et sa date à nos tableaux
            all_payoffs.append(coupon)
            all_payment_times.append(np.full(nb_simulations, payment_date))
        
        # Ajouter le remboursement du capital à l'échéance
        capital_payoff = self.nominal * self.capital_protection
        all_payoffs.append(capital_payoff * np.ones(nb_simulations))
        all_payment_times.append(np.full(nb_simulations, self.maturity))
        
        # Concaténer tous les payoffs et leurs dates
        payoffs = np.concatenate(all_payoffs)
        payment_times = np.concatenate(all_payment_times)
        
        return payoffs, payment_times