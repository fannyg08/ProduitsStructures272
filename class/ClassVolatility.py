from abc import ABC, abstractmethod 
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Union, Callable
from ClassMarket import MarketData


class VolatilityModel(ABC):
    PARAMETERS: list = []
    """
    Classe abstraite de base pour tous les modèles de volatilité
    """
    @abstractmethod
    def __init__(self, market_data:Optional[MarketData] = None, 
                 fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Initialise le modèle de volatilité.
        Inputs : 
        ---------
            market_data : objet contenant la surface des prix de l'option, permettant la calibration
            fixed_parameters : dictionnaire contenant les valeurs des paramètres (si l'on n'a pas de surface de prix)
        """
        self.market_data: MarketData = market_data
        if fixed_parameters is not None :
            for param in fixed_parameters.keys():
                if param not in self.__class__.PARAMETERS:
                    raise ValueError(f"Le paramètre '{param}' n’est pas reconnu pour {self.__class__.__name__}")
        self.parameters: Dict[str, float] = self.calibrate() \
            if self.market_data is not None else fixed_parameters
    
    @abstractmethod
    def calibrate(self) -> Dict[str, float]:
        """
        Procède à la calibration du modèle aux données de marché
        Input : 
        ---------
            Instance qui contient les données de marché
        
        Output : 
        ---------
            Dictionnaire contenant les paramètres calibrés
        """
        pass

    def compute_price_error(self, 
                           market_data: Dict, 
                           model_params: Dict,
                           weights: Optional[np.ndarray] = None) -> float:
        """
        Calcule l'erreur au carré entre les prix du modèle et les prix de marché.
        Inputs : 
        ---------
            market_data : Données de marché (spot, strikes, maturities, market_prices, etc.)
            model_params: Paramètres actuels du modèle à évaluer
            weights: Poids à appliquer aux différents points
            
        Returns:
            Erreur quadratique pondérée
        """
        # Sauvegarder les paramètres actuels
        old_params = self.parameters.copy()
        
        # Mettre à jour temporairement les paramètres
        self.parameters = model_params
        
        # Extraire les données
        spot = market_data['spot']
        r = market_data['risk_free_rate']
        q = market_data.get('dividend_yield', 0.0)
        strikes = market_data['strikes']
        maturities = market_data['maturities']
        market_prices = market_data['market_prices']
        option_types = market_data.get('option_types', 
                                     np.full((len(strikes), len(maturities)), 'call'))
        
        # Calcul de l'erreur
        total_error = 0.0
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                option_type = option_types[i, j]
                market_price = market_prices[i, j]
                # Calculer le prix selon le modèle
                vol = self.get_implied_volatility(strike, maturity)
                model_price = self._black_scholes_price(
                    spot, strike, maturity, r, q, vol, option_type
                )
                
                # Calculer l'erreur pondérée
                if market_price > 1.0:
                    # Erreur relative pour les prix significatifs
                    error = ((model_price - market_price) / market_price) ** 2
                else:
                    # Erreur absolue pour les petits prix
                    error = (model_price - market_price) ** 2
                
                total_error += error
        
        # Restaurer les paramètres d'origine
        self.parameters = old_params
        
        return total_error 

    @abstractmethod
    def get_implied_volatility(self, strike: float, maturity: float) -> float:
        """
        Calcule la volatilité implicite pour un strike et une maturité donnés.
        """
        pass
    
    @abstractmethod
    def get_local_volatility(self, spot: float, time: float) -> float:
        """
        Calcule la volatilité locale pour un spot et un temps donnés.
        """
        pass

    def _black_scholes_price(self, 
                            spot: float, 
                            strike: float, 
                            maturity: float, 
                            r: float, 
                            q: float, 
                            sigma: float, 
                            option_type: str) -> float:
        """
        Calcule le prix d'une option selon la formule de Black-Scholes (mêmes notations).
        """
        from scipy.stats import norm
        
        # Cas particulier pour les maturités très courtes
        if maturity < 1e-8:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        # Calcul 
        d1 = (np.log(spot/strike) + (r - q + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)
        
        # Prix selon Black-Scholes
        if option_type.lower() == 'call':
            price = spot * np.exp(-q * maturity) * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
        else:  # 'put'
            price = strike * np.exp(-r * maturity) * norm.cdf(-d2) - spot * np.exp(-q * maturity) * norm.cdf(-d1)
        return price

class ThetaSSVI:
    """
    Modélise la fonction theta(t) qui représente la variance totale ATM en fonction du temps.
    La forme correspond à la moyenne d'un processus CIR.
    """
    def __init__(self, kappa: float, v0: float, v_inf: float):
        self.kappa = kappa
        self.v0 = v0
        self.v_inf = v_inf

    def __call__(self, t: float) -> float:
        if t <= 0:
            return 0.0
        # Cas limite
        factor = (1 - np.exp(-self.kappa * t)) / (self.kappa * t) if self.kappa > 1e-8 else 1.0
        theta_t = (factor * (self.v0 - self.v_inf) + self.v_inf) * t
        return theta_t

    def get_parameters(self) -> Dict[str, float]:
        return {'kappa': self.kappa, 'v0': self.v0, 'v_inf': self.v_inf}


class SSVIModel(VolatilityModel):
    PARAMETERS = ['kappa', 'v0', 'v_inf', 'rho', 'eta', 'lambda']

    def __init__(self, market_data: Optional[MarketData] = None,
                 fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        super().__init__(market_data, fixed_parameters)

    def calibrate(self) -> Dict[str, float]:
        """
        Calibre les paramètres en deux étapes :
        1. Ajuste kappa, v0, v_inf sur les points ATM uniquement
        2. Ajuste rho, eta, lambda avec theta_t fixé
        """
        if self.market_data is None :
            raise ValueError("Aucune donnée de marché fournie")

        spot = self.market_data.spot
        maturities = self.market_data.maturities
        strikes = self.market_data.strikes
        prices = self.market_data.market_prices

        # Étape 1 : estimation de kappa, v0 et v_inf par minimisation de l'erreur ATM
        # Récupération du strike ATM (possible qu'on soit proche mais pas exactement, on peut donc en avoir plusieurs)
        atm_indices = [i for i, K in enumerate(strikes) if np.abs(K - spot) / spot < 0.01]
        maturities_atm = maturities
        prices_atm = prices[atm_indices, :]

        def atm_objective(params):
            kappa, v0, v_inf = params
            theta_model = ThetaSSVI(kappa, v0, v_inf)
            error = 0.0
            # Boucle sur les maturités
            for j, t in enumerate(maturities_atm):
                # Calcul de la valeur de la fonction theta
                theta_t = theta_model(t)
                model_vol = np.sqrt(theta_t / t)
                # Si plusieurs strikes ATM à cause des effets de seuil, boucle sur ceux-ci
                for i in atm_indices:
                    # Calcul du prix théorique
                    price_model = self._black_scholes_price(
                        spot, strikes[i], t,
                        self.market_data.risk_free_rate,
                        self.market_data.dividend_yield,
                        model_vol,
                        self.market_data.option_types[i, j]
                    )
                    error += (price_model - prices[i, j])**2
            return error
        # On met des bornes aux paramètres pour assurer la cohérence et stabilité
        # kappa -> vitesse de retour à la moyenne, positif et < 5 (arbitraire)
        # v0 -> variance donc > 0 et < 2 (2 arbitraire mais c'est une variance donc est normalement faible)
        # Même chose pour v_inf
        bounds1 = [(1e-6, 5.0), (1e-6, 2.0), (1e-6, 2.0)]
        # Minimisation
        res1 = minimize(atm_objective, [0.5, 0.2, 0.1], bounds=bounds1, method='L-BFGS-B')

        kappa, v0, v_inf = res1.x

        print(f"Première optimisation - Statut: {res1.success}, Fonction objectif: {res1.fun}")
        print(f"Paramètres: kappa={kappa}, v0={v0}, v_inf={v_inf}")
        print(f"Bornes atteintes? kappa: {kappa <= bounds1[0][0]+1e-5 or kappa >= bounds1[0][1]-1e-5}, " + 
        f"v0: {v0 <= bounds1[1][0]+1e-5 or v0 >= bounds1[1][1]-1e-5}, " + 
        f"v_inf: {v_inf <= bounds1[2][0]+1e-5 or v_inf >= bounds1[2][1]-1e-5}")
        self.theta_model = ThetaSSVI(kappa, v0, v_inf)

        # Étape 2 : calibration du smile avec rho, lambda et eta
        def smile_objective(params):
            rho, eta, lam = params
            total_error = 0.0
            # Boucle sur les strikes (plus uniquement ATM !)
            for i, K in enumerate(strikes):
                # Boucle sur les maturités
                for j, t in enumerate(maturities):
                    #k = log-moneyness forward
                    k = np.log(K / (spot * np.exp((self.market_data.risk_free_rate - self.market_data.dividend_yield) * t)))
                    theta_t = self.theta_model(t)
                    phi = eta * theta_t**lam
                    w = (theta_t / 2) * (
                        1 + rho * phi * k +
                        np.sqrt((phi * k + rho)**2 + (1 - rho**2))
                    )
                    sigma = np.sqrt(w / t)
                    price_model = self._black_scholes_price(
                        spot, K, t,
                        self.market_data.risk_free_rate,
                        self.market_data.dividend_yield,
                        sigma,
                        self.market_data.option_types[i, j]
                    )
                    total_error += (price_model - prices[i, j])**2
            return total_error
        # Ici aussi, on impose des bornes
        # rho = corrélation donc entre -1 et 1 (sans atteindre ces bornes sinon problème de singularité),
        # eta > 0 et limité arbitrairement à 5 : si eta = 0, on n'a pas de smile, si eta < 0, problèmes de signe dans la racine carrée
        # lambda entre 0 et 1 (sans atteindre sa borne inférieure) : exposant de courbure
        bounds2 = [(-0.999, 0.999), (1e-6, 7.0), (0.01, 1.0)]
        res2 = minimize(fun = smile_objective, 
                        x0 = [0.0, 0.5, 0.5], 
                        bounds=bounds2, 
                        method='L-BFGS-B')
        
        rho, eta, lam = res2.x
        print(f"Seconde optimisation - Statut: {res2.success}, Fonction objectif: {res2.fun}")
        print(f"Paramètres: rho={rho}, eta={eta}, lambda={lam}")
        print(f"Bornes atteintes? rho: {rho <= bounds2[0][0]+1e-5 or rho >= bounds2[0][1]-1e-5}, " + 
        f"eta: {eta <= bounds2[1][0]+1e-5 or eta >= bounds2[1][1]-1e-5}, " + 
        f"lambda: {lam <= bounds2[2][0]+1e-5 or lam >= bounds2[2][1]-1e-5}")
        self.parameters = {
            'kappa': kappa,
            'v0': v0,
            'v_inf': v_inf,
            'rho': rho,
            'eta': eta,
            'lambda': lam
        }

        return self.parameters
    
    def get_implied_volatility(self, strike: float, maturity: float) -> float:
        spot = self.market_data.spot
        k = np.log(strike / (spot * np.exp((self.market_data.risk_free_rate - self.market_data.dividend_yield) * maturity)))
        theta_t = self.theta_model(maturity)
        phi = self.parameters["eta"] * theta_t**self.parameters["lambda"]
        w = (theta_t / 2) * (
            1 + self.parameters["rho"] * phi * k +
            np.sqrt((phi * k + self.parameters["rho"])**2 + 1 - self.parameters["rho"]**2)
        )
        return np.sqrt(w / maturity)
    
    def get_local_volatility(self, spot, time):
        raise NotImplementedError("La volatilité locale n'est pas définie dans le cadre du modèle SSVI. \n" \
        "Si vous souhaitez une volatilité locale, choisissez le modèle de Dupire. Sinon, vous pouvez récupérer la " \
        "volatilité implicite de ce modèle avec la fonction `get_implied_volatility`")

"""
filepath = "Surface_Prix_SPX_Calls_Fictive.xlsx"
test = MarketData.from_file(filepath, 5300, 0.02)
test_dict = test.to_dict()
SSVI_test = SSVIModel(market_data=test)
print(SSVI_test.parameters)
"""

