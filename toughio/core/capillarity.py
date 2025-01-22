from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class BaseCapillarity(ABC):
    _id = None
    _name = ""

    def __init__(self, *args) -> None:
        """
        Base class for capillarity models.

        Do not use.

        """
        pass

    def __repr__(self):
        """Display capillarity model informations."""
        out = [f"{self._name} capillarity model (ICP = {self._id}):"]
        out += [
            f"    CP({i + 1}) = {parameter}"
            for i, parameter in enumerate(self.parameters)
        ]
        return "\n".join(out)

    def __call__(self, sl):
        """Calculate capillary pressure given liquid saturation."""
        if np.ndim(sl) == 0:
            if not (0.0 <= sl <= 1.0):
                raise ValueError()
            return self._eval(sl, *self.parameters)
        else:
            sl = np.asarray(sl)
            if not np.logical_and((sl >= 0.0).all(), (sl <= 1.0).all()):
                raise ValueError()
            return np.array([self._eval(sat, *self.parameters) for sat in sl])

    @abstractmethod
    def _eval(self, sl, *args):
        raise NotImplementedError()

    def plot(self, n=100, ax=None, figsize=(10, 8), plt_kws=None):
        """
        Plot capillary pressure curve.

        Parameters
        ----------
        n : int, optional, default 100
            Number of saturation points.
        ax : matplotlib.pyplot.Axes or None, optional, default None
            Matplotlib axes. If `None`, a new figure and axe is created.
        figsize : array_like or None, optional, default None
            New figure size if `ax` is `None`.
        plt_kws : dict or None, optional, default None
            Additional keywords passed to :func:`matplotlib.pyplot.semilogy`.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting capillary pressure curve requires matplotlib to be installed."
            )

        if not (isinstance(n, int) and n > 1):
            raise ValueError()
        if not (ax is None or isinstance(ax, plt.Axes)):
            raise TypeError()
        if not (figsize is None or isinstance(figsize, (tuple, list, np.ndarray))):
            raise TypeError()
        if len(figsize) != 2:
            raise ValueError()
        if not (plt_kws is None or isinstance(plt_kws, dict)):
            raise TypeError()

        # Plot parameters
        plt_kws = plt_kws if plt_kws is not None else {}
        _kwargs = {"linestyle": "-", "linewidth": 2}
        _kwargs.update(plt_kws)

        # Initialize figure
        if ax:
            ax1 = ax
        else:
            figsize = figsize if figsize else (8, 5)
            fig = plt.figure(figsize=figsize, facecolor="white")
            ax1 = fig.add_subplot(1, 1, 1)

        # Calculate capillary pressure
        sl = np.linspace(0.0, 1.0, n)
        pcap = self(sl)

        # Plot
        ax1.semilogy(sl, np.abs(pcap), **_kwargs)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_xlabel("Saturation (liquid)")
        ax1.set_ylabel("Capillary pressure (Pa)")
        ax1.grid(True, linestyle=":")

        plt.draw()
        plt.show()
        return ax1

    @property
    def id(self):
        """Return capillarity model ID in TOUGH."""
        return self._id

    @property
    def name(self):
        """Return capillarity model name."""
        return self._name

    @abstractproperty
    def parameters(self):
        raise NotImplementedError()

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError()


class Linear(BaseCapillarity):
    _id = 1
    _name = "Linear"

    def __init__(self, pmax, smin, smax):
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
        if smax <= smin:
            raise ValueError()
        self.parameters = [pmax, smin, smax]

    def _eval(self, sl, *args):
        """Linear function."""
        pmax, smin, smax = args
        return (
            -pmax
            if sl <= smin
            else 0.0 if sl >= smax else -pmax * (smax - sl) / (smax - smin)
        )

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._pmax, self._smin, self._smax]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 3:
            raise ValueError()
        self._pmax = value[0]
        self._smin = value[1]
        self._smax = value[2]


class Milly(BaseCapillarity):
    _id = 4
    _name = "Milly"

    def __init__(self, slr):
        """
        Milly's function.

        After Milly (1982).

        Parameters
        ----------
        slr : scalar
            Irreducible liquid saturation (CP(1)).

        """
        if slr < 0.0:
            raise ValueError()
        self.parameters = [slr]

    def _eval(self, sl, *args):
        (slr,) = args
        sl = max(sl, 1.001 * slr)
        fac = (
            1.0
            if sl - slr >= 0.371
            else 10 ** (2.26 * (0.371 / (sl - slr) - 1.0) ** 0.25 - 2.0)
        )
        return -97.783 * fac

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slr]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 1:
            raise ValueError()
        self._slr = value[0]


class Pickens(BaseCapillarity):
    _id = 2
    _name = "Pickens"

    def __init__(self, p0, slr, sl0, x):
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
        if not (0.0 < slr < 1.0):
            raise ValueError()
        if sl0 < 1.0:
            raise ValueError()
        if x == 0.0:
            raise ValueError()
        self.parameters = [p0, slr, sl0, x]

    def _eval(self, sl, *args):
        """Pickens et al function."""
        p0, slr, sl0, x = args
        sl = max(sl, 1.001 * slr)
        sl = 0.999 * sl0 if sl > 0.999 * sl0 else sl

        A = (1.0 + sl / sl0) * (sl0 - slr) / (sl0 + slr)
        B = 1.0 - sl / sl0
        return -p0 * (np.log(A / B * (1.0 + (1.0 - B**2 / A**2) ** 0.5))) ** (1.0 / x)

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._p0, self._slr, self._sl0, self._x]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 4:
            raise ValueError()
        self._p0 = value[0]
        self._slr = value[1]
        self._sl0 = value[2]
        self._x = value[3]


class TRUST(BaseCapillarity):
    _id = 3
    _name = "TRUST"

    def __init__(self, p0, slr, eta, pe, pmax):
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
        if slr < 0.0:
            raise ValueError()
        if eta == 0.0:
            raise ValueError()
        self.parameters = [p0, slr, eta, pe, pmax]

    def _eval(self, sl, *args):
        """TRUST capillary pressure."""
        p0, slr, eta, pe, pmax = args
        if sl > slr:
            pcap = -pe - p0 * ((1.0 - sl) / (sl - slr)) ** (1.0 / eta)
        else:
            pcap = -abs(pmax)
        pcap = max(pcap, -abs(pmax))
        pcap *= (1.0 - sl) / 0.001 if sl > 0.999 else 1.0
        return pcap

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._p0, self._slr, self._eta, self._pe, self._pmax]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 5:
            raise ValueError()
        self._p0 = value[0]
        self._slr = value[1]
        self._eta = value[2]
        self._pe = value[3]
        self._pmax = value[4]


class vanGenuchten(BaseCapillarity):
    _id = 7
    _name = "van Genuchten"

    def __init__(self, m, slr, alpha, pmax, sls):
        """
        Van Genuchten function.

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
        if pmax < 0.0:
            raise ValueError()
        self.parameters = [m, slr, alpha, pmax, sls]

    def _eval(self, sl, *args):
        """Van Genuchten function."""
        m, slr, alpha, pmax, sls = args
        if sl == 1.0:
            pcap = 0.0
        else:
            if sl > slr:
                Seff = (sl - slr) / (sls - slr)
                pcap = -1.0 / abs(alpha) * (Seff ** (-1.0 / m) - 1.0) ** (1.0 - m)
            else:
                pcap = -abs(pmax)
            pcap = max(pcap, -abs(pmax))
            pcap *= (1.0 - sl) / 0.001 if sl > 0.999 else 1.0
        return pcap

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._m, self._slr, self._alpha, self._pmax, self._sls]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 5:
            raise ValueError()
        self._m = value[0]
        self._slr = value[1]
        self._alpha = value[2]
        self._pmax = value[3]
        self._sls = value[4]
