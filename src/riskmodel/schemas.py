from dataclasses import dataclass


@dataclass(frozen=True)
class StatioResult:
    """Stores pvalues of the stationatity tests"""

    kpss: float
    adf: float
    pp: float
