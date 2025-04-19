from abc import ABC, abstractmethod 
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional
from structuration.ClassMarket import MarketData
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize_scalar, brentq
import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

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

class DupireModel(VolatilityModel):
    """
    Implémentation du modèle de Dupire à volatilité locale, avec 
    des méthodes numériques robustes pour le calcul des volatilités implicites et locales.
    """
    PARAMETERS = []  # Le modèle de Dupire n'a pas de paramètres à calibrer
    
    def __init__(self, market_data: Optional[MarketData] = None) -> None:
        """
        Initialise le modèle de Dupire.
        
        Inputs :
        -----------
        market_data : Objet MarketData avec les données de marché pour la calibration
        """
        self.is_calibrated = False
        self.market_data = market_data
        self.parameters = {}
        
        if market_data is None:
            raise ValueError('Pour le modèle de Dupire, la présence de prix de marché est obligatoire !')
        self.calibrate()
    
    def calibrate(self) -> Dict[str, float]:
        """
        "Calibration" du modèle de Dupire, qui consiste à:
        1. Extraire la surface de volatilité implicite
        2. Calculer la surface de volatilité locale
        
        Ouput :
        --------
            Dictionnaire vide (Dupire n'a pas de paramètres à calibrer)
        """
        print("Calibration du modèle de Dupire")
        
        # Calcul de la surface de volatilité implicite
        self._create_implied_vol_grid()
        
        # Calcul de la surface de volatilité locale
        self._compute_local_vol_surface()
        
        self.is_calibrated = True
        print("Calibration du modèle de Dupire terminée")
        return {}
        
    def _create_implied_vol_grid(self) -> None:
        """
        Calcule la surface de volatilité implicite à partir des prix d'options observés
        en inversant la formule de Black-Scholes. 
        """

        
        # Création de variables locales (pour éviter des lignes trop lourdes)
        spot = self.market_data.spot
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        market_prices = self.market_data.market_prices
        option_types = self.market_data.option_types
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
        
        # Création d'une grille pour stocker les volatilités implicites
        self.implied_vol_grid = np.zeros((len(maturities), len(strikes)))
        
        num_fails = 0
        # Calcul des volatilités implicites pour chaque point (maturité, strike)
        # Boucle sur les lignes (maturités)
        for i, T in enumerate(maturities):
            # Boucle sur les colonnes (strikes)
            for j, K in enumerate(strikes):
                option_type = option_types[i, j]
                market_price = market_prices[i, j]
                
                """
                Les lignes suivantes servent à identifier des cas limites / incohérents qui viennent 
                biaiser les calculs. Cela comprend : 
                1/ Les cas où les prix sont trop faibles (proches de 0) ou trop élevés (trop proche du 
                spot pour un call par exemple qui pourrait mener à un arbitrage)
                2/ Les cas où les prix sont proches de voir inférieurs à la valeur intrinsèque (non sens écocnomique)
                3/ Les cas où le prix est trop proche de sa borne supérieure
                """
                intrinsic = max(0, spot - K) if option_type.lower() == 'call' else max(0, K - spot)
                upper_bound = spot if option_type.lower() == 'call' else K
                # Cas 1
                if market_price < 0.01:
                    moneyness = K / spot
                    self.implied_vol_grid[i, j] = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                    continue
                # Cas 2
                if abs(market_price - intrinsic) < 1e-4:
                    # Prix proche de la valeur intrinsèque => volatilité très faible
                    self.implied_vol_grid[i, j] = 0.05
                    continue
                # Cas 3
                elif abs(market_price - upper_bound) < 1e-4:
                    # Prix proche de la borne supérieure => volatilité très élevée
                    self.implied_vol_grid[i, j] = 1.5
                    continue
                
                """
                Désormais, on peut procéder à l'optimisation
                """
                # Fonction à minimiser : somme des carrés des erreurs 
                def objective(sigma):
                    if sigma <= 0:
                        return 1e10  # Pénalité pour les valeurs irréalisables
                    try:
                        model_price = self._black_scholes_price(spot, K, T, r, q, sigma, option_type)
                        return (model_price - market_price)**2
                    except:
                        return 1e10
                
                try:
                    # D'abord on essaye brentq 
                    def root_objective(sigma):
                        return self._black_scholes_price(spot, K, T, r, q, sigma, option_type) - market_price
                    sigma_low, sigma_high = 0.001, 5.0
                    
                    # On vérifie si les valeurs aux limites ont des signes opposés
                    y_low = root_objective(sigma_low)
                    y_high = root_objective(sigma_high)
                    
                    if y_low * y_high < 0:  # Signes opposés => on utilise brentq
                        if abs(y_low) < 1e-5:
                            implied_vol = sigma_low
                        elif abs(y_high) < 1e-5:
                            implied_vol = sigma_high
                        elif y_low * y_high < 0:
                            implied_vol = brentq(root_objective, sigma_low, sigma_high, rtol=1e-5, maxiter=100)
                        else:
                            result = minimize_scalar(objective, bounds=(0.001, 3.0), method='bounded')
                            implied_vol = result.x
                    
                    # Sinon on utilise une minimisation directe
                    else:  
                        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
                        implied_vol = result.x
                    
                    # On vérifie que la valeur est raisonnable
                    if implied_vol < 0.001 or implied_vol > 5.0 or np.isnan(implied_vol):
                        # On estime la volatilité à partir de la moneyness (cas non idéal)
                        moneyness = K / spot
                        implied_vol = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                        num_fails += 1
                        
                except Exception as e:
                    # Même approximation
                    moneyness = K / spot
                    implied_vol = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                    num_fails += 1
                    if num_fails <= 5: 
                        print(f"Échec de convergence pour K={K}, T={T}. Utilisation de sigma={implied_vol:.4f}")

                # Remplissage de la grille de volatilité implicite
                self.implied_vol_grid[i, j] = implied_vol
        
        if num_fails > 5:
            print(f"... et {num_fails - 5} autres échecs de convergence.")
        
        """
        On veut maintenant lisser la surface, la méthode de Dupire pouvant donner des surfaces irrégulières.
        Cela permet par ailleurs de réduire le bruit capté. Pour cela, on applique un filtre gaussien sur la 
        surface : on remplace chaque point par une moyenne pondérée de ses voisins, la pondération suivant une 
        loi gaussienne (les points proches ont plus de poids). Plus sigma est grand, plus le lissage est fort.
        Ce traitement est utile après la calibration point par point, car il atténue les effets numériques instables 
        et assure une première régularisation de la grille.

        Une fois la grille lissée, on construit une fonction d’interpolation cubique à l’aide de splines cubiques. 
        Cette interpolation introduit un second niveau de lissage global contrôlé par le paramètre `s`. 
        Pour `s > 0`, l'interpolateur ne passe pas exactement par les points de la grille mais ajuste une surface plus régulière, 
        ce qui améliore la stabilité et la continuité de la volatilité implicite interpolée. Chaque segment est déterminé de manière 
        à assurer une continuité des dérivées premières et secondes entre les segments, créant ainsi une courbe lisse. 
        """
        self.implied_vol_interpolator = RectBivariateSpline(
            maturities, strikes, self.implied_vol_grid, kx=min(3, len(maturities)-1), 
            ky=min(3, len(strikes)-1), s=1  
        )
        print(f"Surface de volatilité implicite calculée !")

    def _compute_local_vol_surface(self) -> None:
        """
        Calcule la surface de volatilité locale à partir de la surface de volatilité implicite
        en utilisant la formule de Dupire avec régularisation. Cela implique en outre le calcul 
        des dérivées de la volatilité implicite par rapport au strike (première et seconde) et au 
        temps (première uniquement).
        """
        spot = self.market_data.spot
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        r = self.market_data.risk_free_rate
        q = self.market_data.dividend_yield
    
    # Création d'une grille pour la volatilité locale
        self.local_vol_grid = np.zeros((len(maturities), len(strikes)))
    
    # Calcul de la volatilité locale pour chaque point (K, T)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Extraction de la volatilité implicite lissée
                sigma_imp = float(self.implied_vol_interpolator(T, K)[0][0])
            
            # Vérification de la valeur de volatilité implicite
                if np.isnan(sigma_imp) or sigma_imp <= 0.001:
                    # Si la volatilité implicite est NaN ou trop petite, on utilise une approximation
                    moneyness = K / spot
                    sigma_imp = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                """
                On calcule les dérivées par différence finie (formules usuelles).
                Les cas aux bornes sont traités. 
                """
                # Dérivée première par rapport au strike - différences finies centrées
                if j > 0 and j < len(strikes) - 1:
                    # Différence centrée standard
                    K_plus = strikes[j + 1]
                    K_minus = strikes[j - 1]
                    sigma_plus = float(self.implied_vol_interpolator(T, K_plus)[0][0])
                    sigma_minus = float(self.implied_vol_interpolator(T, K_minus)[0][0])
                
                    # Vérification des valeurs
                    if np.isnan(sigma_plus):
                        moneyness_plus = K_plus / spot
                        sigma_plus = 0.2 * (1 + 0.5 * abs(moneyness_plus - 1))
                    if np.isnan(sigma_minus):
                        moneyness_minus = K_minus / spot
                        sigma_minus = 0.2 * (1 + 0.5 * abs(moneyness_minus - 1))
                
                    dSigma_dK = (sigma_plus - sigma_minus) / (K_plus - K_minus)
            
                # Cas à la borne inférieure
                elif j == 0:
                    K_plus = strikes[j+1]
                    sigma_plus = float(self.implied_vol_interpolator(T, K_plus)[0][0])
                    if np.isnan(sigma_plus):
                        moneyness_plus = K_plus / spot
                        sigma_plus = 0.2 * (1 + 0.5 * abs(moneyness_plus - 1))
                
                    dSigma_dK = (sigma_plus - sigma_imp) / (K_plus - K)
            
                # Cas à la borne supérieure
                else:
                    K_minus = strikes[j-1]
                    sigma_minus = float(self.implied_vol_interpolator(T, K_minus)[0][0])
                    if np.isnan(sigma_minus):
                        moneyness_minus = K_minus / spot
                        sigma_minus = 0.2 * (1 + 0.5 * abs(moneyness_minus - 1))
                
                    dSigma_dK = (sigma_imp - sigma_minus) / (K - K_minus)
            
                # Régularisation de la dérivée première
                dSigma_dK = np.clip(dSigma_dK, -10.0, 10.0)
            
                # Dérivée seconde par rapport au strike avec régularisation
                # Cas standard
                if j > 0 and j < len(strikes) - 1:
                    h = (strikes[j+1] - strikes[j-1]) / 2
                    d2Sigma_dK2 = (sigma_plus - 2*sigma_imp + sigma_minus) / (h**2)
                    # Régularisation pour éviter les oscillations
                    d2Sigma_dK2 = np.sign(d2Sigma_dK2) * min(abs(d2Sigma_dK2), 5.0)
                # Cas aux bornes
                else:
                    # On fixe à 0 pour éviter les instabilités
                    d2Sigma_dK2 = 0
            
                # Dérivée par rapport au temps avec régularisation
                # Cas standard
                if i > 0 and i < len(maturities) - 1:
                    T_plus = maturities[i+1]
                    T_minus = maturities[i-1]
                    sigma_T_plus = float(self.implied_vol_interpolator(T_plus, K)[0][0])
                    sigma_T_minus = float(self.implied_vol_interpolator(T_minus, K)[0][0])
                
                    # Vérification des valeurs
                    if np.isnan(sigma_T_plus):
                        moneyness = K / spot
                        sigma_T_plus = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                    if np.isnan(sigma_T_minus):
                        moneyness = K / spot
                        sigma_T_minus = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                
                    dSigma_dT = (sigma_T_plus - sigma_T_minus) / (T_plus - T_minus)
            
                # Cas à la borne inférieure
                elif i == 0 and i + 1 < len(maturities):
                    T_plus = maturities[i+1]
                    sigma_T_plus = float(self.implied_vol_interpolator(T_plus, K)[0][0])
                    if np.isnan(sigma_T_plus):
                        moneyness = K / spot
                        sigma_T_plus = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                
                    dSigma_dT = (sigma_T_plus - sigma_imp) / (T_plus - T)
            
                # Cas à la borne supérieure
                elif i > 0:
                    T_minus = maturities[i-1]
                    sigma_T_minus = float(self.implied_vol_interpolator(T_minus, K)[0][0])
                    if np.isnan(sigma_T_minus):
                        moneyness = K / spot
                        sigma_T_minus = 0.2 * (1 + 0.5 * abs(moneyness - 1))
                
                    dSigma_dT = (sigma_imp - sigma_T_minus) / (T - T_minus)
                else:
                    # Si un seul point de maturité
                    dSigma_dT = 0
            
                # Régularisation
                dSigma_dT = np.clip(dSigma_dT, -5.0, 5.0)
            
                """
                Désormais, on peut passer au calcul de la volatilité locale
                en appliquant la formule de Dupire.
                """
                try:
                    # Protection contre T trop petit
                    if T < 1e-6:
                        local_vol = sigma_imp
                    else:
                        # Calcul de la volatilité locale
                        numerator = (sigma_imp**2 + 2 * sigma_imp * T * 
                               (dSigma_dT + (r - q) * K * dSigma_dK))
                    
                        denominator = ((1 + K * q * dSigma_dK * np.sqrt(T))**2 + 
                                      sigma_imp * K**2 * T * (d2Sigma_dK2 - q * (dSigma_dK**2) * np.sqrt(T)))
                    
                        # On traite les cas extrêmes en utilisant la volatilité implicite comme proxy
                        if denominator <= 0.001 or numerator <= 0.001 or numerator / denominator <= 0:
                            local_vol = sigma_imp
                        else:
                            local_vol = np.sqrt(numerator / denominator)
                        
                            # Limites physiques raisonnables
                            local_vol = min(max(local_vol, 0.01), 2.0)
                except Exception as e:
                    # En cas d'erreur dans le calcul, on reprend la volatilité implicite
                    local_vol = sigma_imp
            
                # Dernière vérification pour s'assurer qu'il n'y a pas de NaN
                if np.isnan(local_vol) or local_vol <= 0:
                    local_vol = sigma_imp
                    if np.isnan(local_vol) or local_vol <= 0:
                        # Dernière tentative: utiliser une valeur par défaut
                        moneyness = K / spot
                        local_vol = 0.2 * (1 + 0.5 * abs(moneyness - 1))
            
                self.local_vol_grid[i, j] = local_vol
    
        # Vérification finale: remplacer les NaN restants par des valeurs raisonnables
        mask = np.isnan(self.local_vol_grid)
        if np.any(mask):
            print(f"Attention: {np.sum(mask)} valeurs NaN détectées dans la grille de volatilité locale")
            # On remplace les NaN par la moyenne des voisins non-NaN, ou par une valeur par défaut
            for i in range(len(maturities)):
                for j in range(len(strikes)):
                    if np.isnan(self.local_vol_grid[i, j]):
                        # Trouver les voisins non-NaN
                        neighbors = []
                        if i > 0 and not np.isnan(self.local_vol_grid[i-1, j]):
                            neighbors.append(self.local_vol_grid[i-1, j])
                        if i < len(maturities)-1 and not np.isnan(self.local_vol_grid[i+1, j]):
                            neighbors.append(self.local_vol_grid[i+1, j])
                        if j > 0 and not np.isnan(self.local_vol_grid[i, j-1]):
                            neighbors.append(self.local_vol_grid[i, j-1])
                        if j < len(strikes)-1 and not np.isnan(self.local_vol_grid[i, j+1]):
                            neighbors.append(self.local_vol_grid[i, j+1])
                    
                        if neighbors:
                            self.local_vol_grid[i, j] = np.mean(neighbors)
                        else:
                            # Pas de voisins non-NaN, on utilise une valeur par défaut
                            moneyness = strikes[j] / spot
                            self.local_vol_grid[i, j] = 0.2 * (1 + 0.5 * abs(moneyness - 1))
    
        """
        On applique un lissage gaussien puis on crée l'interpolateur
        """
        # Appliquer un lissage gaussien à la grille
        self.local_vol_grid = gaussian_filter(self.local_vol_grid, sigma=1.0)
    
        # Créer l'interpolateur
        self.local_vol_interpolator = RectBivariateSpline(
            maturities, strikes, self.local_vol_grid, 
            kx=min(3, len(maturities)-1), ky=min(3, len(strikes)-1), s=1 
        )
    
        print(f"Surface de volatilité locale calculée avec succès!")

    def get_implied_volatility(self, strike: float, maturity: float) -> float:
        """
        Calcule la volatilité implicite pour un strike et une maturité donnés.

        Inputs :
        -----------
            strike : Prix d'exercice
            maturity : Temps jusqu'à l'échéance en années

        Output :
        --------
            Volatilité implicite
        """
        if not self.is_calibrated:
            raise ValueError("Le modèle doit être calibré avant de calculer les volatilités implicites")
    
        # Vérification des limites et ajustement si nécessaire
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
    
    # S'assurer que les valeurs sont dans les bornes de l'interpolateur
        bounded_strike = max(min(strike, max(strikes)), min(strikes))
        bounded_maturity = max(min(maturity, max(maturities)), min(maturities))
    
    # Utilisation de l'interpolateur pour obtenir la volatilité implicite
        try:
            implied_vol = float(self.implied_vol_interpolator(bounded_maturity, bounded_strike)[0][0])
            return implied_vol
        except Exception as e:
            # En cas d'erreur, on utilise une estimation basée sur la moneyness
            spot = self.market_data.spot
            moneyness = bounded_strike / spot
            return 0.2 * (1 + 0.5 * abs(moneyness - 1))

    def get_local_volatility(self, spot: float, maturity: float) -> float:
        """
        Calcule la volatilité locale pour un spot et une maturité donnés.

        Inputs :
        -----------
            spot : Prix du sous-jacent (utilisé comme strike dans le modèle de Dupire)
            maturity : Temps jusqu'à l'échéance en années

        Output :
        ----------
            Volatilité locale
        """
        if not self.is_calibrated:
            raise ValueError("Le modèle doit être calibré avant de calculer les volatilités locales")
    
    # Vérification des limites et ajustement si nécessaire
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
    
    # S'assurer que les valeurs sont dans les bornes de l'interpolateur
        bounded_strike = max(min(spot, max(strikes)), min(strikes))
        bounded_maturity = max(min(maturity, max(maturities)), min(maturities))
    
    # Utilisation de l'interpolateur pour obtenir la volatilité locale
        try:
            local_vol = float(self.local_vol_interpolator(bounded_maturity, bounded_strike)[0][0])
        
        # Vérifier que la valeur est raisonnable, sinon utiliser la volatilité implicite comme proxy
            if np.isnan(local_vol) or local_vol < 0.01 or local_vol > 2.0:
                return self.get_implied_volatility(bounded_strike, bounded_maturity)
            
            return local_vol
        except Exception as e:
            # En cas d'erreur, on utilise la volatilité implicite comme proxy
            return self.get_implied_volatility(bounded_strike, bounded_maturity)
    
    def plot_vol_surface(self, vol_type: str = 'local'):
        """
        Affiche la surface de volatilité (locale ou implicite).
        
        Inputs :
        -----------
            vol_type : Type de volatilité à afficher ('local' ou 'implied')
        """        
        if not self.is_calibrated:
            raise ValueError("Le modèle doit être calibré avant d'afficher les surfaces de volatilité !")
        
        strikes = self.market_data.strikes
        maturities = self.market_data.maturities
        
        # Créer une grille plus fine pour une meilleure visualisation
        grid_strikes = np.linspace(min(strikes), max(strikes), 100)
        grid_maturities = np.linspace(min(maturities), max(maturities), 50)
        X, Y = np.meshgrid(grid_strikes, grid_maturities)
        
        # Calculer les valeurs de volatilité sur la grille fine
        Z = np.zeros_like(X)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if vol_type.lower() == 'local':
                    Z[i, j] = self.get_local_volatility(X[i, j], Y[i, j])
                else:  # 'implied'
                    Z[i, j] = self.get_implied_volatility(X[i, j], Y[i, j])
        # Configurer le titre et les labels selon le type de volatilité
        if vol_type.lower() == 'local':
            title = "Surface de volatilité locale - Modèle de Dupire"
            zlabel = "Volatilité locale"
        else:  # 'implied'
            title = "Surface de volatilité implicite - Modèle de Dupire"
            zlabel = "Volatilité implicite"
        
        # Création de la figure 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        ax.set_title(title)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturité (années)")
        ax.set_zlabel(zlabel)
        
        # Ajout d'une barre de couleur
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Ajustement de l'angle de vue
        ax.view_init(elev=30, azim=135)
        
        plt.tight_layout()
        plt.show()
        
"""
import datetime as dt
filepath = "OptionData (1).xlsx"
test = MarketData.from_file(filepath, spot = 198,
                            risk_free_rate=0.02,
                            sheet_name= "Matrice finale", pricing_date=dt.datetime(2025,4,17))
test_dict = test.to_dict()
SSVI_test = SSVIModel(market_data=test)
#print(test.strikes)
print(SSVI_test.parameters)
SSVI_test.plot_vol_surface()
dupire_test = DupireModel(market_data=test)
print(dupire_test.implied_vol_grid)
#print(dupire_test.local_vol_grid)
dupire_test.plot_vol_surface('local')
dupire_test.plot_vol_surface('implied')
"""