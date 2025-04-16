from abc import ABC, abstractmethod
import numpy as np
import math
from scipy import interpolate
from typing import Optional, Dict, Literal
from src.utility.types import Maturity


class RateModel(ABC):
    """
    Classe abstraite représentant un modèle de taux d'intérêt.
    Toute sous-classe doit implémenter une méthode 'discount_factor'.
    """

    @abstractmethod
    def discount_factor(self, t: float) -> float:
        """
        Calcule le facteur d'actualisation pour une échéance donnée.

        Args:
            t (float) : Échéance (en années).

        Returns:
            float : Facteur d'actualisation.
        """
        pass


class VasicekModel(RateModel):
    """
    Modèle de taux court de Vasicek.
    Ce modèle permet d'obtenir des facteurs d'actualisation fermés
    dans un cadre stochastique.

    Paramètres :
        - a : vitesse de retour à la moyenne
        - b : taux de long terme
        - sigma : volatilité
        - r0 : taux instantané initial
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def discount_factor(self, t: float) -> float:
        """
        Calcule le facteur d'actualisation à l'horizon t selon le modèle de Vasicek.

        Args:
            t (float) : Maturité en années.

        Returns:
            float : Valeur du facteur d'actualisation.
        """
        a, b, sigma, r0 = self.a, self.b, self.sigma, self.r0
        B = (1 - np.exp(-a * t)) / a
        A = (b - sigma**2 / (2 * a**2)) * (B - t) - (sigma**2 / (4 * a)) * B**2
        return np.exp(-A - B * r0)


class Rate(RateModel):
    """
    Modèle de taux déterministe, constant ou basé sur une courbe interpolée.

    Ce modèle peut gérer :
        - Un taux constant
        - Une courbe de taux donnée à des maturités discrètes, interpolée
        - Deux types de taux : "continuous" ou "compounded"
    """

    def __init__(
        self,
        rate: Optional[float] = None,
        rate_type: Literal["continuous", "compounded"] = "continuous",
        rate_curve: Optional[Dict[Maturity, float]] = None,
        interpolation_type: Literal["linear", "quadratic", "cubic"] = "linear",
    ) -> None:
        """
        Initialise un modèle de taux déterministe.

        Args:
            rate (float, optionnel) : Taux constant.
            rate_type (str) : Type de capitalisation ("continuous" ou "compounded").
            rate_curve (dict, optionnel) : Courbe de taux par maturité (objets Maturity).
            interpolation_type (str) : Type d'interpolation à appliquer sur la courbe.
        """
        self.__rate = rate
        self.__rate_type = rate_type
        self.__interpol = None

        if rate_curve is not None:
            self.__interpol = interpolate.interp1d(
                [m.maturity_in_years for m in rate_curve.keys()],
                list(rate_curve.values()),
                fill_value="extrapolate",
                kind=interpolation_type,
            )

    def get_rate(self, t: float) -> float:
        """
        Récupère le taux applicable à une échéance donnée.

        Args:
            t (float) : Maturité en années.

        Returns:
            float : Taux interpolé ou constant.

        Raises:
            ValueError : Si aucun taux ou courbe n'a été fourni.
        """
        if self.__rate is not None:
            return self.__rate
        if self.__interpol is not None:
            return float(self.__interpol(t))
        raise ValueError("Aucun taux ni courbe de taux n'a été défini.")

    def discount_factor(self, t: float) -> float:
        """
        Calcule le facteur d'actualisation selon le type de taux.

        Args:
            t (float) : Échéance en années.

        Returns:
            float : Facteur d'actualisation.

        Raises:
            ValueError : Si le type de taux est invalide.
        """
        rate = self.get_rate(t)
        if self.__rate_type == "continuous":
            return math.exp(-rate * t)
        elif self.__rate_type == "compounded":
            return 1.0 / ((1 + rate) ** t)
        else:
            raise ValueError("Le type de taux spécifié est invalide.")
