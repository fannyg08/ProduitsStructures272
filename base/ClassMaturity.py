from datetime import datetime
from typing import Dict, Optional, Literal


# Types littéraux pour différents éléments de produit financier
DayCountConvention = Literal["ACT/360", "ACT/365"]
OptionType = Literal["call", "put"]
ProductKindType = Literal["reverse-convertible", "outperformer-certificate"]
OptionKindType = Literal["vanilla", "binary", "barrier"]
BondType = Literal["vanilla", "zero-coupon"]
OptionStrategyType = Literal[
    "straddle", "strangle", "butterfly", "call-spread", "put-spread", "strip", "strap"
]
BarrierDirection = Literal["up", "down"]
BarrierType = Literal["ko", "ki"]


class Maturity:
    # Dictionnaire de correspondance des conventions de base de calcul d'année
    DAY_COUNT_MAPPING: Dict[DayCountConvention, float] = {
        "ACT/360": 360.0,
        "ACT/365": 365.0,
    }

    def __init__(
        self,
        maturity_in_years: Optional[float] = None,
        maturity_in_days: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        day_count_convention: DayCountConvention = "ACT/360",
    ):
        """
        Initialise un objet Maturity à partir d'une durée en années, en jours,
        ou entre deux dates, selon une convention de calcul de jours.

        Args:
            maturity_in_years (float, optionnel) : Durée exprimée en années.
            maturity_in_days (float, optionnel) : Durée exprimée en jours.
            start_date (datetime, optionnel) : Date de début.
            end_date (datetime, optionnel) : Date de fin.
            day_count_convention (str) : Convention jour/année ("ACT/360" ou "ACT/365").
        """
        self._day_count_convention = day_count_convention

        if maturity_in_years is not None:
            self._maturity_in_years = maturity_in_years

        elif maturity_in_days is not None:
            self._maturity_in_years = (
                maturity_in_days / self.DAY_COUNT_MAPPING[day_count_convention]
            )

        elif start_date is not None and end_date is not None:
            self._maturity_in_years = (
                (end_date - start_date).days
                / self.DAY_COUNT_MAPPING[day_count_convention]
            )

        else:
            raise ValueError(
                "Veuillez fournir une durée en années, en jours, ou bien une paire de dates."
            )

    @property
    def maturity_in_years(self) -> float:
        """
        Renvoie la maturité exprimée en années.

        Returns:
            float : Maturité en années.
        """
        return self._maturity_in_years

    def __str__(self) -> str:
        """
        Renvoie une représentation lisible de l'objet Maturity.

        Returns:
            str : Description textuelle de la maturité et de la convention utilisée.
        """
        return (
            f"Maturité<durée={self.maturity_in_years:.4f} ans, "
            f"convention={self._day_count_convention}>"
        )
