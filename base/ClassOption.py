from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from tqdm import tqdm

from structuration.ClassVolatility import VolatilityModel
from base.ClassRate import Rate
from base.ClassMaturity import OptionType, Maturity


class Option(ABC):
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        maturity: Maturity,
        domestic_rate: Rate,
        volatility: VolatilityModel,
        option_type: OptionType,
        dividend: Optional[float] = None,
        foreign_rate: Optional[Rate] = None,
    ) -> None:
        """
        Classe de base abstraite pour les options. Cette classe peut être utilisée pour modéliser
        des options actions ou de change, en prenant en compte les dividendes et taux étrangers.

        Args:
            spot_price (float) : Prix du sous-jacent.
            strike_price (float) : Prix d'exercice de l'option.
            maturity (Maturity) : Objet représentant la maturité en années.
            domestic_rate (Rate) : Taux sans risque domestique.
            volatility (Volatility) : Surface ou modèle de volatilité.
            option_type (OptionType) : Type d'option ('call' ou 'put').
            dividend (float, optionnel) : Taux de dividende. Par défaut 0.
            foreign_rate (Rate, optionnel) : Taux étranger, pour les options de change.
        """
        self._spot_price = spot_price
        self._strike_price = strike_price
        self._maturity = maturity
        self._domestic_rate = domestic_rate
        self._volatility = volatility
        self._option_type = option_type
        self._dividend = dividend if dividend is not None else 0.0
        self._foreign_rate = foreign_rate
        self._d1 = self.__d1_func()
        self._d2 = self.__d2_func()

    def __d1_func(self) -> float:
        """
        Calcule le paramètre d1 utilisé dans le modèle de Black-Scholes.

        Returns:
            float : Valeur de d1.
        """
        maturity = self._maturity.maturity_in_years
        if maturity <= 0:
            raise ValueError("La maturité doit être supérieure à zéro.")
        rate_difference = (
            (self._domestic_rate.discount_factor(maturity) - self._dividend)
            if self._foreign_rate is None
            else (
                self._domestic_rate.discount_factor(maturity)
                - self._foreign_rate.discount_factor(maturity)
            )
        )
        return (
            np.log(self._spot_price / self._strike_price)
            + (
                (self._domestic_rate.discount_factor(maturity) - self._dividend)
                + 0.5
                * self._volatility.get_implied_volatility(
                    self._strike_price / self._spot_price,
                    self._maturity.maturity_in_years,
                )
                ** 2
            )
            * self._maturity.maturity_in_years
        ) / (
            self._volatility.get_implied_volatility(
                self._strike_price / self._spot_price, self._maturity.maturity_in_years
            )
            * np.sqrt(self._maturity.maturity_in_years)
        )

    def __d2_func(self) -> float:
        """
        Calcule le paramètre d2 basé sur d1 et la volatilité.

        Returns:
            float : Valeur de d2.
        """
        maturity = self._maturity.maturity_in_years
        if maturity <= 0:
            raise ValueError("La maturité doit être supérieure à zéro.")
        return (
            np.log(self._spot_price / self._strike_price)
            + (
                (self._domestic_rate.discount_factor(maturity) - self._dividend)
                + 0.5
                * self._volatility.get_implied_volatility(
                    self._strike_price / self._spot_price,
                    self._maturity.maturity_in_years,
                )
                ** 2
            )
            * self._maturity.maturity_in_years
        ) / (
            self._volatility.get_implied_volatility(
                self._strike_price / self._spot_price, self._maturity.maturity_in_years
            )
            * np.sqrt(self._maturity.maturity_in_years)
        ) - self._volatility.get_implied_volatility(
            self._strike_price / self._spot_price, self._maturity.maturity_in_years
        ) * np.sqrt(
            self._maturity.maturity_in_years
        )

    @property
    def d1(self) -> float:
        """Renvoie la valeur de d1."""
        return self._d1

    @property
    def d2(self) -> float:
        """Renvoie la valeur de d2 (recalculé à partir de d1 pour éviter les erreurs cumulées)."""
        return self._d1 - self._volatility.get_implied_volatility(
            self._strike_price / self._spot_price, self._maturity.maturity_in_years
        ) * np.sqrt(self._maturity.maturity_in_years)

    @abstractmethod
    def compute_price(self):
        """Méthode abstraite à implémenter : retourne le prix de l’option."""
        pass

    @abstractmethod
    def compute_greeks(self):
        """Méthode abstraite à implémenter : retourne l’ensemble des grecques de l’option."""
        pass

    @abstractmethod
    def compute_vega(self):
        """Méthode abstraite à implémenter : retourne le vega de l’option."""
        pass

    @abstractmethod
    def compute_delta(self):
        """Méthode abstraite à implémenter : retourne le delta de l’option."""
        pass

    @abstractmethod
    def compute_rho(self):
        """Méthode abstraite à implémenter : retourne le rho de l’option."""
        pass

    @abstractmethod
    def compute_theta(self):
        """Méthode abstraite à implémenter : retourne le theta de l’option."""
        pass

    @abstractmethod
    def compute_gamma(self):
        """Méthode abstraite à implémenter : retourne le gamma de l’option."""
        pass

    def __str__(self) -> str:
        """
        Fournit une représentation lisible de l’objet OptionBase.

        Returns:
            str : Chaîne résumant les paramètres principaux de l’option.
        """
        return (
            f"Option<Spot Price={self._spot_price:.2f}, Strike Price={self._strike_price:.2f}, "
            f"Maturity={self._maturity}, Option Type={self._option_type}, Volatility={self._volatility}>"
        )

    def monte_carlo_simulation(self, num_paths, num_steps):
        """
        Génère des trajectoires simulées pour le sous-jacent selon un schéma log-normal
        sous mesure risque-neutre (Black-Scholes Monte Carlo).

        Args:
            num_paths (int) : Nombre de trajectoires simulées.
            num_steps (int) : Nombre de pas de temps par trajectoire.

        Returns:
            np.ndarray : Tableau (num_paths, num_steps + 1) contenant les trajectoires.
        """
        dt = self._maturity.maturity_in_years / num_steps
        nudt = (
            (self._domestic_rate.discount_factor(self._maturity) - self._dividend)
            - 0.5
            * self._volatility.get_implied_volatility(
                self._strike_price / self._spot_price, self._maturity.maturity_in_years
            )**2
        ) * dt

        volsdt = self._volatility.get_implied_volatility(
            self._strike_price / self._spot_price, self._maturity.maturity_in_years
        ) * np.sqrt(dt)

        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self._spot_price

        for step in tqdm(
            range(1, num_steps + 1), desc="Calcul des trajectoires Monte Carlo...", leave=False
        ):
            random_shocks = np.random.normal(0, 1, num_paths)
            paths[:, step] = paths[:, step - 1] * np.exp(nudt + volsdt * random_shocks)

        return paths
