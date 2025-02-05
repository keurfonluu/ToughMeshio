from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ._curve import BaseCurve


class RelativePermeabilityModel(BaseCurve):
    """Base class for relative permeability model."""
    _short = "RP"

    def _eval(self, sl: ArrayLike, *args) -> None: ...

    def plot(
        self,
        n: int = 100,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> None:
        """
        Plot relative permeability curve.

        Parameters
        ----------
        n : int, default 100
            Number of saturation points.
        ax : matplotlib.axes.Axes, optional
            Plot axes.
        **kwargs : dict, optional
            Additional keyword arguments. See ``matplotlib.pyplot.plot`` for more details.

        """
        # Calculate liquid and gas relative permeability
        sl = np.linspace(0.0, 1.0, n)
        kl, kg = self(sl)

        # Plot
        ax = ax if ax is not None else plt.gca()

        ax.plot(sl, kl, label="Liquid", **kwargs)
        ax.plot(sl, kg, label="Gas", **kwargs)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Saturation (liquid)")
        ax.set_ylabel("Relative permeability")


class Linear(RelativePermeabilityModel):
    """
    Linear function.

    Parameters
    ----------
    slmin : scalar
        Lower liquid saturation threshold (RP(1)).
    sgmin : scalar
        Lower gas saturation threshold (RP(2)).
    slmax : scalar
        Upper liquid saturation threshold (RP(3)).
    sgmax : scalar
        Upper gas saturation threshold (RP(4)).

    """
    def __init__(self, slmin: float, sgmin: float, slmax: float, sgmax: float) -> None:
        """Initialize linear relative permeability model."""
        if slmin >= slmax:
            raise ValueError("slmin must be lower than slmax")

        if sgmin >= sgmax:
            raise ValueError("sgmin must be lower than sgmax")
        
        super().__init__(slmin, sgmin, slmax, sgmax)
        self._id = 1
        self._name = "Linear"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Linear function."""
        slmin, sgmin, slmax, sgmax = args
        sg = 1.0 - sl

        mask = np.logical_and(sl > slmin, sl < slmax)
        kl = np.zeros_like(sl)
        kl[mask] = (sl[mask] - slmin) / (slmax - slmin)
        kl[sl >= slmax] = 1.0

        mask = np.logical_and(sg > sgmin, sg < sgmax)
        kg = np.zeros_like(sl)
        kg[mask] = (sg[mask] - sgmin) / (sgmax - sgmin)
        kg[sg >= sgmax] = 1.0

        return kl, kg


class Pickens(RelativePermeabilityModel):
    """
    Gas perfect mobile function.

    Parameters
    ----------
    x : scalar
        RP(1).

    """
    def __init__(self, x: float) -> None:
        """Initialize Pickens' relative permeability model."""
        super().__init__(x)
        self._id = 2
        self._name = "Pickens"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Gas perfect mobile function."""
        (x,) = args

        kl = np.power(sl, x)
        kg = np.ones_like(sl)

        return kl, kg


class Corey(RelativePermeabilityModel):
    """
    Corey's curve.

    After Corey (1954).

    Parameters
    ----------
    slr : scalar
        Irreducible liquid saturation (RP(1)).
    sgr : scalar
        Irreducible gas saturation (RP(2)).

    """
    def __init__(self, slr: float, sgr: float) -> None:
        """Initialize Corey's relative permeability model."""
        if slr + sgr >= 1.0:
            raise ValueError("slr + sgr must be lower than 1.0")

        super().__init__(slr, sgr)
        self._id = 3
        self._name = "Corey"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Corey's curve."""
        slr, sgr = args
        sg = 1.0 - sl

        mask = np.logical_and(sg >= sgr, sg < 1.0 - slr)
        Shat = (sl[mask] - slr) / (1.0 - slr - sgr)

        kl = np.zeros_like(sl)
        kl[mask] = np.power(Shat, 4)
        kl[sg < sgr] = 1.0

        kg = np.zeros_like(sl)
        kg[mask] = (1.0 - Shat**2) * (1.0 - Shat) ** 2
        kg[sg >= 1.0 - slr] = 1.0

        return kl, kg


class Grant(RelativePermeabilityModel):
    """
    Grant's curve.

    After Grant (1977).

    Parameters
    ----------
    slr : scalar
        Irreducible liquid saturation (RP(1)).
    sgr : scalar
        Irreducible gas saturation (RP(2)).

    """
    def __init__(self, slr: float, sgr: float) -> None:
        """Initialize Grant's relative permeability model."""
        if slr + sgr >= 1.0:
            raise ValueError("slr + sgr must be lower than 1.0")

        super().__init__(slr, sgr)
        self._id = 4
        self._name = "Grant"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Grant's curve."""
        slr, sgr = args
        sg = 1.0 - sl

        mask = np.logical_and(sg >= sgr, sg < 1.0 - slr)
        Shat = (sl[mask] - slr) / (1.0 - slr - sgr)

        kl = np.zeros_like(sl)
        kl[mask] = np.power(Shat, 4)
        kl[sg < sgr] = 1.0

        kg = np.zeros_like(sl)
        kg[mask] = 1.0 - kl[mask]
        kg[sg >= 1.0 - slr] = 1.0

        return kl, kg


class FattKlikoff(RelativePermeabilityModel):
    """
    Fatt and Klikoff's function.

    After Fatt and Klikoff (1959).

    Parameters
    ----------
    slr : scalar
        Irreducible liquid saturation (RP(1)).

    """
    def __init__(self, slr: float) -> None:
        """Initialize Fatt and Klikoff's relative permeability model."""
        if slr >= 1.0:
            raise ValueError("slr must be lower than 1.0")

        super().__init__(slr)
        self._id = 6
        self._name = "Fatt-Klikoff"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Fatt and Klikoff's function."""
        (slr,) = args

        mask = sl > slr
        Seff = np.zeros_like(sl)
        Seff[mask] = (sl[mask] - slr) / (1.0 - slr)

        kl = np.power(Seff, 3)
        kg = np.power(1.0 - Seff, 3)

        return kl, kg


class vanGenuchtenMualem(RelativePermeabilityModel):
    """
    van Genuchten-Mualem's function.

    After Mualem (1976) and van Genuchten (1980).

    Parameters
    ----------
    m : scalar
        Related to pore size distribution (RP(1)).
    slr : scalar
        Irreducible liquid saturation (RP(2)).
    sls : scalar
        Maximum liquid saturation (RP(3)).
    sgr : scalar
        Irreducible gas saturation (RP(4)).

    """
    def __init__(self, m: float, slr: float, sls: float, sgr: float) -> None:
        """Initialize van Genuchten-Mualem's relative permeability model."""
        super().__init__(m, slr, sls, sgr)
        self._id = 7
        self._name = "van Genuchten-Mualem"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """van Genuchten-Mualem's function."""
        m, slr, sls, sgr = args

        Seff = (sl - slr) / (sls - slr)
        mask = Seff > 0.0
        kl = np.zeros_like(sl)
        kl[mask] = Seff[mask] ** 0.5 * (1.0 - (1.0 - Seff[mask] ** (1.0 / m)) ** m) ** 2
        kl[sl >= sls] = 1.0

        Shat = ((sl - slr) / (1.0 - slr - sgr)).clip(0.0, 1.0)
        kg = np.where(sgr > 0.0, (1.0 - Shat**2) * (1.0 - Shat) ** 2, 1.0 - kl)
        kg[sl >= sls] = 0.0

        return kl, kg


class Verma(RelativePermeabilityModel):
    """
    Verma's function.

    After Verma et al. (1985).

    Parameters
    ----------
    slr : scalar, default 0.2
        Irreducible liquid saturation (RP(1)).
    sls : scalar, default 0.895
        Maximum liquid saturation (RP(2)).
    a : scalar, default 1.259
        A (RP(3)).
    b : scalar, default -1.7615
        B (RP(4)).
    c : scalar, default 0.5089
        C (RP(5)).

    """
    def __init__(self, slr: float = 0.2, sls: float = 0.895, a: float = 1.259, b: float = -1.7615, c: float = 0.5089) -> None:
        super().__init__(slr, sls, a, b, c)
        self._id = 8
        self._name = "Verma"

    def _eval(self, sl: ArrayLike, *args) -> tuple[ArrayLike, ArrayLike]:
        """Verma's function."""
        slr, sls, a, b, c = args

        Shat = ((sl - slr) / (sls - slr)).clip(0.0, 1.0)
        kl = np.power(Shat, 3)
        kg = (a + b * Shat + c * Shat**2).clip(0.0, 1.0)

        return kl, kg
