from abc import ABC, abstractmethod
from logging import warn
from typing import Dict, List, Optional, Union, Callable

from scipy.optimize import minimize

from base.ClassRate import RateModel  # importe la classe abstraite ou ses dérivés
from base.ClassMaturity import Maturity


# === 1. Classe abstraite d'obligation ===
class ABCBond(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute_price(self):
        pass


# === 2. Optimisation pour le calcul du YTM ===
class Optimization:
    def __init__(
        self,
        pricing_function: Callable[[float], float],
        target_value: float,
        epsilon: float = 0.001,
        initial_value: float = 0.01,
    ) -> None:
        self.__pricing_function = pricing_function
        self.__target_value = target_value
        self.__epsilon = epsilon
        self.__initial_value = initial_value

    def run(self):
        return minimize(
            lambda x: (self.__target_value - self.__pricing_function(x)) ** 2,
            x0=(self.__initial_value),
            method="SLSQP",
            tol=self.__epsilon,
        )


# === 3. Obligation Zéro-Coupon ===
class ZeroCouponBond(ABCBond):
    _price: Optional[float] = None

    def __init__(
        self,
        rate_model: RateModel,
        maturity: Maturity,
        nominal: float,
    ) -> None:
        self.__rate_model = rate_model
        self.__maturity = maturity
        self.__nominal = nominal

    def compute_price(self) -> float:
        if self._price is None:
            self._price = self.__nominal * self.__rate_model.discount_factor(
                self.__maturity.maturity_in_years
            )
        return self._price
    
    @property
    def rate_model(self) -> RateModel:
        return self.__rate_model
    
    @property
    def maturity(self) -> Maturity:
        return self.__maturity
    
    @property
    def nominal(self) -> float:
        return self.__nominal


# === 4. Obligation à coupon ===
class Bond(ABCBond):
    _price: Optional[float] = None
    _ytm: Optional[float] = None

    def __init__(
        self,
        rate_model: RateModel,
        maturity: Maturity,
        nominal: float,
        coupon_rate: float,
        nb_coupon: int,
    ) -> None:
        self.__rate_model = rate_model
        self.__maturity = maturity
        self.__nominal = nominal
        self.__coupon_rate = coupon_rate
        self.__nb_coupon = nb_coupon
        self.__components = self.__run_components()

    def compute_price(self, force_rate: Optional[float] = None) -> float:
        if self._price is None or force_rate is not None:
            price = sum(
                [
                    zc_bond.get("zc_bond").compute_price()
                    for zc_bond in self.__components
                ]
            )
            if force_rate is not None:
                return price
            else:
                self._price = price
        return self._price

    def ytm(self):
        optimizer = Optimization(
            pricing_function=lambda rate: self.compute_price(force_rate=rate),
            target_value=self._price,
            initial_value=0.01,
        )
        optim_res = optimizer.run()
        if optim_res.status == 0:
            self._ytm = optim_res["x"][0]
        else:
            warn("Error while optimizing", optim_res)
        return self._ytm

    def __run_components(self):
        t = self.__maturity.maturity_in_years
        terms = []
        step = 1.0 / self.__nb_coupon
        while t > 0:
            terms.append(t)
            t -= step
        terms.sort()
        freq_coupon = float(self.__coupon_rate) / self.__nb_coupon * self.__nominal
        coupons: List[Dict[str, Union[str, float, ZeroCouponBond]]] = [
            {
                "maturity": t,
                "cf_type": "coupon",
                "zc_bond": ZeroCouponBond(
                    self.__rate_model,
                    Maturity(maturity_in_years=t),
                    freq_coupon
                    + (
                        0.0 if t < self.__maturity.maturity_in_years else self.__nominal
                    ),
                ),
            }
            for t in terms
        ]
        return coupons
    @property
    def rate_model(self) -> RateModel:
        return self.__rate_model
    
    @property
    def maturity(self) -> Maturity:
        return self.__maturity
    
    @property
    def nominal(self) -> float:
        return self.__nominal
    
    @property
    def coupon_rate(self) -> float:
        return self.__coupon_rate
    
    @property
    def nb_coupon(self) -> int:
        return self.__nb_coupon
