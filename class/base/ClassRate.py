from abc import ABC, abstractmethod
import numpy as np
import math
from scipy import interpolate
from typing import Optional, Dict, Literal
from src.utility.types import Maturity


class RateModel(ABC):
    @abstractmethod
    def discount_factor(self, t: float) -> float:
        pass


class VasicekModel(RateModel):
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


class Rate(RateModel):
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
        if self.__rate is not None:
            return self.__rate
        if self.__interpol is not None:
            return float(self.__interpol(t))
        raise ValueError("Error: no rate or rate curve defined.")

    def discount_factor(self, t: float) -> float:
        rate = self.get_rate(t)
        if self.__rate_type == "continuous":
            return math.exp(-rate * t)
        elif self.__rate_type == "compounded":
            return 1.0 / ((1 + rate) ** t)
        else:
            raise ValueError("Invalid rate_type")
