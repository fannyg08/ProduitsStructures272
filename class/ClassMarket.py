from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from dataclasses import dataclass
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from scipy.stats import norm
from scipy.optimize import brentq
import datetime as dt

@dataclass
class MarketData:
    """
    Classe pour stocker et gérer les données de marché nécessaires à la calibration 
    des modèles de volatilité. Ses attributs sont : 

        - spot : prix du sous-jacent
        - risk-free rate : valeur du taux sans risque
        - strikes : plage de strikes considérée
        - maturities : plage de maturités considérée
        - market_prices : matrice des prix de marché pour les strikes et maturités considérés
        - dividend_yield : taux de dividende (continu)
        - option_type : précise le type de l'option considérée (call ou put)
        - market_ivs : surface de volatilité implicite (du modèle de BS), calculée en interne
        - 
    """
    # Données de base
    spot: float  
    risk_free_rate: float  
    strikes: np.ndarray  # Ensemble des prix d'exercice
    maturities: np.ndarray  # Ensemble des maturités
    market_prices: np.ndarray  # Matrice des prix de marché [strikes, maturities]
    dividend_yield: float = 0.0 
    option_type: str = 'call'  # Matrice des types d'options ('call' ou 'put')
    
    # Volatilités implicites (matrices [strikes, maturities])
    market_ivs: np.ndarray = None  

    def __post_init__(self):
        """
        Initialisation après la création de l'objet.
        """
        # Dimensions
        if self.market_prices.shape != (len(self.strikes), len(self.maturities)):
            raise ValueError("La matrice market_prices doit avoir la forme (len(strikes), len(maturities))")
        
        # On initialise option_types
        self.option_types = np.full_like(self.market_prices, self.option_type, dtype=object)
        
        # Calcul des volatilités implicites
        self.compute_implied_volatilities()
    
    def compute_implied_volatilities(self) -> None:
        """
        Calcule les volatilités implicites (Black-Scholes) à partir des prix de marché.
        La matrice est directement un attribut de l'objet.
        """
        self.market_ivs = np.zeros_like(self.market_prices)
        # Calcul de la matrice des volatilités implicites
        for i, strike in enumerate(self.strikes):
            for j, maturity in enumerate(self.maturities):
                self.market_ivs[i, j] = self._implied_volatility_bs(
                    self.market_prices[i, j],
                    strike,
                    maturity,
                    self.option_types[i, j]
                )
    
    def _implied_volatility_bs(self, price: float, strike: float, maturity: float, option_type: str) -> float:
        """
        Calcule la volatilité implicite à partir du prix d'une option.
        
        Inputs : 
        ---------
            price : Prix de l'option
            strike : Prix d'exercice
            maturity : Maturité en années
            option_type : 'call' ou 'put'
            
        Output :
        ---------
            Volatilité implicite
        """
        def bs_price(sigma: float) -> float:
            """
            Renvoie le prix de Black & Scholes de l'option, en fonction de la volatilité
            """
            S: float = self.spot
            K: float = strike
            r: float = self.risk_free_rate
            q: float = self.dividend_yield
            T: float = maturity
            
            # Cas limite
            if T < 1e-8 or sigma < 1e-8:
                if option_type.lower() == 'call':
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            # Si cas non dégénéré
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type.lower() == 'call':
                return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        """
        On cherche la volatilité implicite qui permet d'égaliser
        le prix de marché avec le prix de BS. Cela revient à chercher
        les racines de la fonction PrixBS - prix de marché. 
        """
        def objective(sigma: float) -> float:
            return bs_price(sigma) - price
        
        # Estimation initiale
        if strike <= self.spot:
            iv_guess = 0.2  # ATM ou ITM
        else:
            iv_guess = 0.3  # OTM
        
        # Recherche de la volatilité implicite en calculant les racines 
        try:     
            iv = brentq(objective, 0.001, 2.0, rtol=1e-6)
        except ValueError:
            iv = iv_guess

        return iv
    
    def to_dict(self) -> Dict:
        """
        Convertit l'objet MarketData en dictionnaire.
        """
        return {
            'spot': self.spot,
            'risk_free_rate': self.risk_free_rate,
            'dividend_yield': self.dividend_yield,
            'strikes': self.strikes,
            'maturities': self.maturities,
            'market_prices': self.market_prices,
            'option_types': self.option_types,
            'market_ivs': self.market_ivs
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketData':
        """
        Crée un objet MarketData à partir d'un dictionnaire.
        
        Input :
        ---------
            data : Dictionnaire contenant les données
            
        Output :
        ---------
            Objet MarketData
        """
        return cls(
            spot=data['spot'],
            risk_free_rate=data['risk_free_rate'],
            strikes=data['strikes'],
            maturities=data['maturities'],
            market_prices=data['market_prices'],
            dividend_yield=data.get('dividend_yield', 0.0),
            option_types=data.get('option_types', None)
        )
    
    @classmethod
    def from_file(cls, filepath: str, spot: float, risk_free_rate: float, dividend_yield: float = 0.0, 
                  sheet_name: str = 0, option_type: str = 'call', 
                  from_what: str = 'excel', delimiter = ',',
                   pricing_date: Optional[str] = None) -> 'MarketData':
        """
        Crée un objet MarketData à partir d'un fichier CSV. Concrètement,
        on s'attend à ce que le fichier CSV ou Excel soit une matrice de prix d'options pour un spot donnée 
        avec les maturités en ligne et les strikes en colonne, ceci afin d'ensuite calculer la surface de volatilité
        ou de calibrer les modèles de volatilité. 
        
        Inputs :
        ---------
            filepath : Chemin vers le fichier (CSV ou Excel)
            spot : Prix du sous-jacent
            risk_free_rate : Taux sans risque
            dividend_yield : Rendement du dividende
            sheet_name : Nom ou index de la feuille pour les fichiers Excel
            option_type : Type d'option ('call' par défaut, ou 'put')
            from_what : Type de fichier ('excel' ou 'csv')
            delimiter : Pour les fichiers CSV, indique le type de délimitation des colonnes (',', ';',...)
            
        Output :
        ---------
            Objet MarketData
        """
        df = pd.read_csv(filepath, delimiter=delimiter) \
            if from_what.lower() == 'csv' else  pd.read_excel(filepath, sheet_name=sheet_name)
        
        # Extraction des données de base : 
        # strikes en colonnes (en ne considérant pas la première colonne de maturités)
        strikes = np.array(df.columns[1:].astype(float))
        raw_maturities = pd.to_datetime(df.iloc[:, 0]) 
        market_prices = df.iloc[:, 1:].values
        if pricing_date is None : 
            pricing_date = dt.datetime.today()
        else:
            pricing_date = pd.to_datetime(pricing_date)

        maturities = np.array((raw_maturities - pricing_date).dt.days / 365)
        # On renvoie l'objet MarketData
        return cls(
            spot=spot,
            risk_free_rate=risk_free_rate,
            strikes=strikes,
            maturities=maturities,
            market_prices=market_prices,
            dividend_yield=dividend_yield,
            option_type=option_type
        )


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