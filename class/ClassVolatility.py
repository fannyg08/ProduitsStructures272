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
        self.is_calibrated = False if fixed_parameters is None else True
        self.parameters: Dict[str, float] = self.calibrate() \
            if fixed_parameters is None else fixed_parameters
        
    
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
                        model_params: Dict) -> float:
        """
        Calcule l'erreur au carré entre les prix du modèle et les prix de marché.
        Inputs : 
        ---------
            market_data : Données de marché (spot, strikes, maturities, market_prices, etc.)
            model_params : Paramètres actuels du modèle à évaluer
        
        Output :
        ---------
        Erreur quadratique pondérée
        """
    
        old_params = self.parameters.copy()
        self.parameters = model_params
    
        # Extraction des données
        spot = market_data.spot
        r = market_data.risk_free_rate
        q  = market_data.dividend_yield
        strikes = market_data.strikes
        maturities = market_data.maturities
        market_prices = market_data.market_prices
        option_types = market_data.option_types
    
        # Calcul de l'erreur
        total_error = 0.0
    
        # Boucle sur les maturités (en ligne)
        for j, maturity in enumerate(maturities):
            # Boucle sur les strikes (en colonne)
            for i, strike in enumerate(strikes):
                option_type = option_types[j, i]
                market_price = market_prices[j, i]
                vol = self.get_implied_volatility(strike, maturity)
                model_price = self._black_scholes_price(
                spot, strike, maturity, r, q, vol, option_type
            )
            
            
            if market_price > 1.0:
                # Erreur relative pour les prix significatifs
                error = ((model_price - market_price) / market_price) ** 2
            else:
                # Erreur absolue pour les petits prix
                error = (model_price - market_price) ** 2
            
            total_error += error

        # On restaure les paramètres d'origine
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
        # Impossible de calibrer sans données
        if self.market_data is None:
            raise ValueError("Aucune donnée de marché fournie")

        spot = self.market_data.spot
        maturities = self.market_data.maturities
        strikes = self.market_data.strikes
        prices = self.market_data.market_prices

        # Informations pour l'user
        print(f"Calibration avec spot={spot}, {len(maturities)} maturités et {len(strikes)} strikes")
        print(f"Premier strike: {strikes[0]}, dernier strike: {strikes[-1]}")
        
        # Étape 1 : estimation de kappa, v0 et v_inf par minimisation de l'erreur ATM
        # Récupération du strike ATM (possible qu'on soit proche mais pas exactement, on peut donc en avoir plusieurs)
        # On s'autorise un delta de 0.05% (dans notre exemple, c'est assez restreint pour n'avoir qu'un seul strike ATM)
        atm_indices = [i for i, K in enumerate(strikes) if np.abs(K - spot) / spot < 0.0005]
        
        # Si avec les données fournies, aucun strike n'est assez proche, on prend celui le plus près.
        if not atm_indices:
            print(f"Aucun strike ATM trouvé avec une tolérance de 0.05%. Utilisation du strike le plus proche.")
            atm_indices = [np.argmin(np.abs(strikes - spot))]
        
        print(f"Indices ATM trouvés: {atm_indices}")
        print(f"Strikes ATM correspondants: {[strikes[i] for i in atm_indices]}")
        
        def atm_objective(params):
            """
            Fonction objectif pour la minimisation ATM.
            """
            kappa, v0, v_inf = params
            theta_model = ThetaSSVI(kappa, v0, v_inf)
            error = 0.0
            
            # Boucle sur les maturités
            for j, t in enumerate(maturities):
                # Calcul de la valeur de la fonction theta
                theta_t = theta_model(t)
                model_vol = np.sqrt(theta_t / t)
                
                # Si plusieurs strikes ATM à cause des effets de seuil, boucle sur ceux-ci
                for i in atm_indices:
                    # Vérifier si l'indice est valide
                    if j < prices.shape[0] and i < prices.shape[1]:
                        # Calcul du prix théorique
                        price_model = self._black_scholes_price(
                            spot, strikes[i], t,
                            self.market_data.risk_free_rate,
                            self.market_data.dividend_yield,
                            model_vol,
                            self.market_data.option_types[j, i]
                        )
                        market_price = prices[j, i]
                        
                        # Poids inversement proportionnel à la maturité
                        
                        # Erreur relative normalisée
                        if market_price > 1.0:
                            rel_error = ((price_model - market_price) / market_price) ** 2
                            error += rel_error
                        else:
                            abs_error = (price_model - market_price) ** 2
                            error += abs_error
                
            return error
            
        # Bornes pour les paramètres
        bounds1 = [(0.0001, np.inf), (0.0001, np.inf), (0.0001, np.inf)]
        
        # On essaye plusieurs points de départ pour ne pas prendre le risque d'être "bloqué" au départ
        best_res = None
        best_fun = float('inf')
        start_points = [
            [0.5, 0.2, 0.1],    
            [2.0, 0.5, 0.3],    # Point alternatif 1
            [0.1, 0.05, 0.02]   # Point alternatif 2 
        ]
        
        # Lancement de la minimisation
        for start in start_points:
            res_attempt = minimize(atm_objective, start, bounds=bounds1, method='L-BFGS-B')
            print(f"Point de départ {start}: résultat={res_attempt.fun}")
            # On conserve le meilleur résultat ("meilleur" point de départ)
            if res_attempt.fun < best_fun:
                best_fun = res_attempt.fun
                best_res = res_attempt
        
        res1 = best_res
        kappa, v0, v_inf = res1.x

        print(f"Optimisation ATM - Statut : {res1.success}, Fonction objectif : {res1.fun}")
        print(f"Paramètres : kappa={kappa}, v0={v0}, v_inf={v_inf}")
        # Calcul de la fonction theta avec paramètres calibrés
        self.theta_model = ThetaSSVI(kappa, v0, v_inf)

        # Étape 2 : calibration du smile avec rho, lambda et eta
        def smile_objective(params):
            """
            Fonction objectif pour la calibration du smile (après calibration ATM)
            """
            rho, eta, lam = params
            total_error = 0.0
            
            # Boucle sur les maturités 
            for j, t in enumerate(maturities):
                # Boucle sur les strikes
                for i, K in enumerate(strikes):
                    #k = log-moneyness forward
                    k = np.log(K / (spot * np.exp((self.market_data.risk_free_rate - self.market_data.dividend_yield) * t)))
                    
                    # Calcul de la variance implicite totale
                    theta_t = self.theta_model(t)
                    phi = eta * theta_t**lam
                    w = (theta_t / 2) * (
                        1 + rho * phi * k +
                        np.sqrt((phi * k + rho)**2 + (1 - rho**2))
                    )
                    sigma = np.sqrt(w / t)
                    # Prix avec modèle SSVI
                    price_model = self._black_scholes_price(
                        spot, K, t,
                        self.market_data.risk_free_rate,
                        self.market_data.dividend_yield,
                        sigma,
                        self.market_data.option_types[j, i]
                    )
                    
                    market_price = prices[j, i]
                    
                    # Erreur (avec gestion de cas limites)
                    if market_price > 1.0:
                        rel_error = ((price_model - market_price) / market_price) ** 2
                        total_error += rel_error
                    else:
                        abs_error = (price_model - market_price) ** 2
                        total_error += abs_error
                
            return total_error
            
        # La suite de la procédure est identique à l'optimisation ATM
        bounds2 = [(-0.99, 0.99), (0.0001, np.inf), (0.001, np.inf)]
        best_res2 = None
        best_fun2 = float('inf')
        start_points2 = [
            [0.0, 0.5, 0.5],    
            [-0.5, 1.0, 0.3],   
            [0.5, 2.0, 0.7]     
        ]
        
        for start in start_points2:
            res_attempt = minimize(fun=smile_objective, x0=start, bounds=bounds2, method='L-BFGS-B')
            print(f"Point de départ {start}: résultat={res_attempt.fun}")
            
            if res_attempt.fun < best_fun2:
                best_fun2 = res_attempt.fun
                best_res2 = res_attempt
        
        res2 = best_res2
        rho, eta, lam = res2.x
        
        print(f"Optimisation smile - Statut : {res2.success}, Fonction objectif : {res2.fun}")
        print(f"Paramètres : rho={rho}, eta={eta}, lambda={lam}")
        self.parameters = {
            'kappa': kappa,
            'v0': v0,
            'v_inf': v_inf,
            'rho': rho,
            'eta': eta,
            'lambda': lam
        }
        self.is_calibrated = True
        return self.parameters
    
    def get_implied_volatility(self, strike: float, maturity: float) -> float:
        """
        Fonction qui permet de calculer la volatilité implicite à partir de la formule du SSVI. 
        """
        if not self.is_calibrated:
            raise ValueError("Le modèle doit être calibré avant de calculer les volatilités implicites !")
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
    
    def plot_vol_surface(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        vol_matrix = np.zeros((len(maturities), len(strikes)))

        for i, t in enumerate(maturities):
            for j, k in enumerate(strikes):
                vol_matrix[i, j] = self.get_implied_volatility(k, t)

        X, Y = np.meshgrid(strikes, maturities)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, vol_matrix, cmap='viridis')

        # Ajout des titres et noms d'axes
        ax.set_title("Surface de volatilité implicite - Modèle SSVI")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturité (années)")
        ax.set_zlabel("Volatilité implicite")

        plt.tight_layout()
        plt.show()


from abc import ABC, abstractmethod 
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import cmath

# Supposons que vous ayez déjà la classe MarketData définie
# from ClassMarket import MarketData


class VolatilityModel(ABC):
    PARAMETERS: list = []
    """
    Classe abstraite de base pour tous les modèles de volatilité
    """
    @abstractmethod
    def __init__(self, market_data=None, 
                 fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Initialise le modèle de volatilité.
        Inputs : 
        ---------
            market_data : objet contenant la surface des prix de l'option, permettant la calibration
            fixed_parameters : dictionnaire contenant les valeurs des paramètres (si l'on n'a pas de surface de prix)
        """
        self.market_data = market_data
        if fixed_parameters is not None:
            for param in fixed_parameters.keys():
                if param not in self.__class__.PARAMETERS:
                    raise ValueError(f"Le paramètre '{param}' n'est pas reconnu pour {self.__class__.__name__}")
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

    def compute_price_error(self, model_params: Dict) -> float:
        """
        Calcule l'erreur au carré entre les prix du modèle et les prix de marché.
        Inputs : 
        ---------
            model_params : Paramètres actuels du modèle à évaluer
        
        Output :
        ---------
        Erreur quadratique pondérée
        """
    
        old_params = self.parameters.copy() if hasattr(self, 'parameters') and self.parameters is not None else {}
        self.parameters = model_params
    
        # Extraction des données
        spot = self.market_data.spot
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        market_prices = self.market_data.market_prices
        option_types = self.market_data.option_types
    
        # Calcul de l'erreur
        total_error = 0.0
    
        # Boucle sur les maturités (en ligne)
        for j, maturity in enumerate(maturities):
            # Boucle sur les strikes (en colonne)
            for i, strike in enumerate(strikes):
                option_type = option_types[j, i]
                market_price = market_prices[j, i]
                
                # Pour Heston, on calcule directement le prix du modèle (pas besoin de passer par la volatilité implicite)
                model_price = self.price_option(spot, strike, maturity, r, q, option_type)
                
                # Erreur relative pour les prix significatifs, absolue pour les petits prix
                if market_price > 1.0:
                    error = ((model_price - market_price) / market_price) ** 2
                else:
                    error = (model_price - market_price) ** 2
                
                total_error += error

        # On restaure les paramètres d'origine
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
        
    def implied_volatility_newton(self, price: float, S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
        """
        Calcule la volatilité implicite par la méthode de Newton-Raphson.
        """
        from scipy.stats import norm
        
        # Fonction à résoudre: BS(sigma) - prix = 0
        def f(sigma):
            return self._black_scholes_price(S, K, T, r, q, sigma, option_type) - price
        
        # Dérivée de la fonction BS par rapport à sigma (vega)
        def vega(sigma):
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Initialisation de sigma avec une approximation simple
        sigma = 0.2
        
        # Itérations de Newton-Raphson
        for i in range(50):
            price_diff = f(sigma)
            vega_val = vega(sigma)
            
            # Si vega proche de zéro, éviter la division par zéro
            if abs(vega_val) < 1e-10:
                break
                
            # Mise à jour de sigma
            sigma = sigma - price_diff / vega_val
            
            # Limites physiques pour la volatilité
            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 5:
                sigma = 5
                break
                
            # Critère de convergence
            if abs(price_diff) < 1e-8:
                break
                
        return sigma


from abc import ABC, abstractmethod 
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import cmath

# Supposons que vous ayez déjà la classe MarketData définie
# from ClassMarket import MarketData


class VolatilityModel(ABC):
    PARAMETERS: list = []
    """
    Classe abstraite de base pour tous les modèles de volatilité
    """
    @abstractmethod
    def __init__(self, market_data=None, 
                 fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Initialise le modèle de volatilité.
        Inputs : 
        ---------
            market_data : objet contenant la surface des prix de l'option, permettant la calibration
            fixed_parameters : dictionnaire contenant les valeurs des paramètres (si l'on n'a pas de surface de prix)
        """
        self.market_data = market_data
        if fixed_parameters is not None:
            for param in fixed_parameters.keys():
                if param not in self.__class__.PARAMETERS:
                    raise ValueError(f"Le paramètre '{param}' n'est pas reconnu pour {self.__class__.__name__}")
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

    def compute_price_error(self, model_params: Dict) -> float:
        """
        Calcule l'erreur au carré entre les prix du modèle et les prix de marché.
        Inputs : 
        ---------
            model_params : Paramètres actuels du modèle à évaluer
        
        Output :
        ---------
        Erreur quadratique pondérée
        """
    
        old_params = self.parameters.copy() if hasattr(self, 'parameters') and self.parameters is not None else {}
        self.parameters = model_params
    
        # Extraction des données
        spot = self.market_data.spot
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        market_prices = self.market_data.market_prices
        option_types = self.market_data.option_types
    
        # Calcul de l'erreur
        total_error = 0.0
    
        # Boucle sur les maturités (en ligne)
        for j, maturity in enumerate(maturities):
            # Boucle sur les strikes (en colonne)
            for i, strike in enumerate(strikes):
                option_type = option_types[j, i]
                market_price = market_prices[j, i]
                
                # Pour Heston, on calcule directement le prix du modèle
                model_price = self.price_option(spot, strike, maturity, r, q, option_type)
                
                # Erreur relative pour les prix significatifs, absolue pour les petits prix
                if market_price > 1.0:
                    error = ((model_price - market_price) / market_price) ** 2
                else:
                    error = (model_price - market_price) ** 2
                
                total_error += error

        # On restaure les paramètres d'origine
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
        
    def implied_volatility_newton(self, price: float, S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
        """
        Calcule la volatilité implicite par la méthode de Newton-Raphson.
        """
        from scipy.stats import norm
        
        # Fonction à résoudre: BS(sigma) - prix = 0
        def f(sigma):
            return self._black_scholes_price(S, K, T, r, q, sigma, option_type) - price
        
        # Dérivée de la fonction BS par rapport à sigma (vega)
        def vega(sigma):
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Initialisation de sigma avec une approximation simple
        sigma = 0.2
        
        # Itérations de Newton-Raphson
        for i in range(50):
            price_diff = f(sigma)
            vega_val = vega(sigma)
            
            # Si vega proche de zéro, éviter la division par zéro
            if abs(vega_val) < 1e-10:
                break
                
            # Mise à jour de sigma
            sigma = sigma - price_diff / vega_val
            
            # Limites physiques pour la volatilité
            if sigma <= 0.001:
                sigma = 0.001
            elif sigma > 5:
                sigma = 5
                break
                
            # Critère de convergence
            if abs(price_diff) < 1e-8:
                break
                
        return sigma




filepath = "prix call.xlsx"
test = MarketData.from_file(filepath, 5275.7, 0.02)
test_dict = test.to_dict()
SSVI_test = SSVIModel(market_data=test)
print(test.strikes)
print(SSVI_test.parameters)
SSVI_test.plot_vol_surface()


