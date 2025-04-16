from abc import ABC, abstractmethod
import numpy as np
import math
from scipy import interpolate
from typing import Optional, Dict, Literal
from src.utility.types import Maturity


class RateModel(ABC):
    """
    Classe abstraite représentant un modèle de taux d'intérêt.
    """

    @abstractmethod
    def discount_factor(self, t: float) -> float:
        """
        Calcule le facteur d'actualisation pour une échéance donnée.

        Args:
            t (float) : Échéance en années.

        Returns:
            float : Facteur d'actualisation.
        """
        pass


class VasicekModel(RateModel):
    """
    Modèle de taux court de Vasicek.
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def discount_factor(self, t: float) -> float:
        a, b, sigma, r0 = self.a, self.b, self.sigma, self.r0
        B = (1 - np.exp(-a * t)) / a
        A = (b - sigma**2 / (2 * a**2)) * (B - t) - (sigma**2 / (4 * a)) * B**2
        return np.exp(-A - B * r0)


class CIRModel(RateModel):
    """
    Modèle de Cox-Ingersoll-Ross (CIR), utilisé pour modéliser l'évolution des taux courts
    avec une variance strictement positive.

    L’équation différentielle stochastique est :
        dr_t = a(b - r_t)dt + σ√r_t dW_t
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def discount_factor(self, t: float) -> float:
        a, b, sigma, r0 = self.a, self.b, self.sigma, self.r0
        gamma = np.sqrt(a**2 + 2 * sigma**2)
        denom = 2 * gamma + (a + gamma) * (np.exp(gamma * t) - 1)
        B = 2 * (np.exp(gamma * t) - 1) / denom
        A = (
            (2 * gamma * np.exp((a + gamma) * t / 2))
            / denom
        ) ** (2 * a * b / sigma**2)
        return A * np.exp(-B * r0)


class HullWhiteModel(RateModel):
    """
    Modèle de Hull-White (version affine à un facteur), extension du modèle de Vasicek
    avec ajustement à la courbe de taux initiale.

    Équation :
        dr_t = [θ(t) - a r_t] dt + σ dW_t
    Ici, θ(t) est implicite car on suppose une courbe plate.
    """

    def __init__(self, a: float, sigma: float, r0: float):
        self.a = a
        self.sigma = sigma
        self.r0 = r0

    def discount_factor(self, t: float) -> float:
        a, sigma, r0 = self.a, self.sigma, self.r0
        B = (1 - np.exp(-a * t)) / a
        A = (
            np.exp(
                -0.25 * sigma**2 / a**3 * (1 - np.exp(-a * t))**2
                * (1 - np.exp(-2 * a * t))
            )
        )
        return A * np.exp(-B * r0)


class Rate(RateModel):
    """
    Modèle de taux déterministe, constant ou basé sur une courbe interpolée.
    """

    def __init__(
        self,
        rate: Optional[float] = None,
        rate_type: Literal["continuous", "compounded"] = "continuous",
        rate_curve: Optional[Dict[Maturity, float]] = None,
        interpolation_type: Literal["linear", "quadratic", "cubic"] = "linear",
    ) -> None:
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
        """
        rate = self.get_rate(t)
        if self.__rate_type == "continuous":
            return math.exp(-rate * t)
        elif self.__rate_type == "compounded":
            return 1.0 / ((1 + rate) ** t)
        else:
            raise ValueError("Le type de taux spécifié est invalide.")
