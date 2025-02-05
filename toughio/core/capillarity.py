from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ._curve import BaseCurve


class CapillarityModel(BaseCurve):
    """Base class for capillarity model."""
    _short = "CP"

    def _eval(self, sl: ArrayLike, *args) -> None: ...

    def plot(
        self,
        n: int = 100,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> None:
        """
        Plot capillary pressure curve.

        Parameters
        ----------
        n : int, default 100
            Number of saturation points.
        ax : matplotlib.axes.Axes, optional
            Plot axes.
        **kwargs : dict, optional
            Additional keyword arguments. See ``matplotlib.pyplot.plot`` for more details.

        """
        # Calculate capillary pressure
        sl = np.linspace(0.0, 1.0, n)
        pcap = self(sl)

        # Plot
        ax = ax if ax is not None else plt.gca()

        ax.semilogy(sl, np.abs(pcap), **kwargs)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Saturation (liquid)")
        ax.set_ylabel("Capillary pressure (Pa)")


class Linear(CapillarityModel):
    """
    Linear function.

    Parameters
    ----------
    pmax : scalar
        Maximum pressure (CP(1)).
    smin : scalar
        Lower liquid saturation threshold (CP(2)).
    smax : scalar
        Upper liquid saturation threshold (CP(3)).

    """
    def __init__(self, pmax: float, smin: float, smax: float) -> None:
        """Initialize linear capillarity model."""
        if smax <= smin:
            raise ValueError("smax must be greater than smin")

        super().__init__(pmax, smin, smax)
        self._id = 1
        self._name = "Linear"

    def _eval(self, sl: ArrayLike, *args) -> ArrayLike:
        """Linear function."""
        pmax, smin, smax = args

        pcap = np.where(sl < smax, -pmax * (smax - sl) / (smax - smin), 0.0)
        pcap[sl <= smin] = -pmax

        return pcap


class Pickens(CapillarityModel):
    """
    Pickens et al. function.

    After Pickens et al. (1979).

    Parameters
    ----------
    p0 : scalar
        Capillary pressure strength (CP(1)).
    slr : scalar
        Irreducible liquid saturation (CP(2)).
    sl0 : scalar
        Initial liquid saturation (CP(3)).
    x : scalar
        CP(4).

    """
    def __init__(self, p0: float, slr: float, sl0: float, x: float) -> None:
        """Initialize Pickens' capillarity model."""
        if not (0.0 < slr < 1.0):
            raise ValueError("slr must be between 0.0 and 1.0")

        if sl0 < 1.0:
            raise ValueError("sl0 must be greater than 1.0")

        if x == 0.0:
            raise ValueError("x cannot be 0.0")

        super().__init__(p0, slr, sl0, x)
        self._id = 2
        self._name = "Pickens"

    def _eval(self, sl: ArrayLike, *args) -> ArrayLike:
        """Pickens et al function."""
        p0, slr, sl0, x = args
        
        sl = sl.clip(1.001 * slr, 0.999 * sl0)
        A = (1.0 + sl / sl0) * (sl0 - slr) / (sl0 + slr)
        B = 1.0 - sl / sl0

        return -p0 * (np.log(A / B * (1.0 + (1.0 - B**2 / A**2) ** 0.5))) ** (1.0 / x)


class TRUST(CapillarityModel):
    """
    TRUST capillary pressure.

    After Narasimhan et al. (1978).

    Parameters
    ----------
    p0 : scalar
        Capillary pressure strength (CP(1)).
    slr : scalar
        Irreducible liquid saturation (CP(2)).
    eta : scalar
        CP(3).
    pe : scalar
        Capillary entry pressure (CP(4)).
    pmax : scalar
        Maximum pressure (CP(5)).

    """
    def __init__(self, p0: float, slr: float, eta: float, pe: float, pmax: float) -> None:
        """Initialize TRUST capillarity model."""
        if slr < 0.0:
            raise ValueError("slr must be greater than 0.0")

        if eta == 0.0:
            raise ValueError("eta cannot be 0.0")

        super().__init__(p0, slr, eta, pe, pmax)
        self._id = 3
        self._name = "TRUST"

    def _eval(self, sl: ArrayLike, *args) -> ArrayLike:
        """TRUST capillary pressure."""
        p0, slr, eta, pe, pmax = args

        mask = sl > slr
        pcap = np.zeros_like(sl)
        pcap[mask] = -pe - p0 * ((1.0 - sl[mask]) / (sl[mask] - slr)) ** (1.0 / eta)
        pcap[~mask] = -abs(pmax)
        pcap = pcap.clip(-abs(pmax), None)
        pcap *= np.where(sl > 0.999, (1.0 - sl) / 0.001, 1.0)

        return pcap


class Milly(CapillarityModel):
    """
    Milly's function.

    After Milly (1982).

    Parameters
    ----------
    slr : scalar
        Irreducible liquid saturation (CP(1)).

    """
    def __init__(self, slr: float) -> None:
        """Initialize Milly's capillarity model."""
        if slr < 0.0:
            raise ValueError("slr must be greater than 0.0")

        super().__init__(slr)
        self._id = 4
        self._name = "Milly"

    def _eval(self, sl: ArrayLike, *args) -> ArrayLike:
        """Milly's function."""
        (slr,) = args

        sl = sl.clip(1.001 * slr, 1.0)
        mask = sl - slr < 0.371
        fac = np.ones_like(sl)
        fac[mask] = 10.0 ** (2.26 * (0.371 / (sl[mask] - slr) - 1.0) ** 0.25 - 2.0)

        return -97.783 * fac


class vanGenuchten(CapillarityModel):
    """
    van Genuchten's function.

    After van Genuchten (1980).

    Parameters
    ----------
    m : scalar
        Related to pore size distribution (CP(1)).
    slr : scalar
        Irreducible liquid saturation (CP(2)).
    alpha : scalar
        Inverse of capillary pressure strength 1/P0 (CP(3)).
    pmax : scalar
        Maximum pressure (CP(4)).
    sls : scalar
        Maximum liquid saturation (CP(5)).

    """
    def __init__(self, m: float, slr: float, alpha: float, pmax: float, sls: float) -> None:
        """Initialize van Genuchten's capillarity model."""
        if pmax < 0.0:
            raise ValueError("pmax must be greater than 0.0")

        super().__init__(m, slr, alpha, pmax, sls)
        self._id = 7
        self._name = "van Genuchten"

    def _eval(self, sl: ArrayLike, *args) -> ArrayLike:
        """van Genuchten's function."""
        m, slr, alpha, pmax, sls = args

        Seff = (sl - slr) / (sls - slr)
        mask = sl > slr
        pcap = np.full_like(sl, -abs(pmax))
        pcap[mask] = (-1.0 / abs(alpha) * (Seff[mask] ** (-1.0 / m) - 1.0) ** (1.0 - m)).clip(-abs(pmax), None)
        pcap *= np.where(sl > 0.999, (1.0 - sl) / 0.001, 1.0)

        return pcap
