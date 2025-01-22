from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class BaseRelativePermeability(ABC):
    _id = None
    _name = ""

    def __init__(self, *args) -> None:
        """
        Base class for relative permeability models.

        Do not use.

        """
        pass

    def __repr__(self):
        """Display relative permeability model informations."""
        out = [f"{self._name} relative permeability model (IRP = {self._id}):"]
        out += [
            f"    RP({i + 1}) = {parameter}"
            for i, parameter in enumerate(self.parameters)
        ]
        return "\n".join(out)

    def __call__(self, sl):
        """Calculate relative permeability given liquid saturation."""
        if np.ndim(sl) == 0:
            if not (0.0 <= sl <= 1.0):
                raise ValueError()
            return self._eval(sl, *self.parameters)
        else:
            sl = np.asarray(sl)
            if not np.logical_and((sl >= 0.0).all(), (sl <= 1.0).all()):
                raise ValueError()
            return np.transpose([self._eval(sat, *self.parameters) for sat in sl])

    @abstractmethod
    def _eval(self, sl, *args):
        raise NotImplementedError()

    def plot(self, n=100, ax=None, figsize=(10, 8), plt_kws=None):
        """
        Plot relative permeability curve.

        Parameters
        ----------
        n : int, optional, default 100
            Number of saturation points.
        ax : matplotlib.pyplot.Axes or None, optional, default None
            Matplotlib axes. If `None`, a new figure and axe is created.
        figsize : array_like or None, optional, default None
            New figure size if `ax` is `None`.
        plt_kws : dict or None, optional, default None
            Additional keywords passed to :func:`matplotlib.pyplot.plot`.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting relative permeability curve requires matplotlib to be installed."
            )

        if not (isinstance(n, int) and n > 1):
            raise TypeError()
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

        # Calculate liquid and gas relative permeability
        sl = np.linspace(0.0, 1.0, n)
        kl, kg = self(sl)

        # Plot
        ax1.plot(sl, kl, label="Liquid", **_kwargs)
        ax1.plot(sl, kg, label="Gas", **_kwargs)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Saturation (liquid)")
        ax1.set_ylabel("Relative permeability")
        ax1.grid(True, linestyle=":")

        plt.draw()
        plt.show()
        return ax1

    @property
    def id(self):
        """Return relative permeability model ID in TOUGH."""
        return self._id

    @property
    def name(self):
        """Return relative permeability model name."""
        return self._name

    @abstractproperty
    def parameters(self):
        raise NotImplementedError()

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError()


class Corey(BaseRelativePermeability):
    _id = 3
    _name = "Corey"

    def __init__(self, slr, sgr):
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
        if slr + sgr >= 1.0:
            raise ValueError()
        self.parameters = [slr, sgr]

    def _eval(self, sl, *args):
        """Corey's curve."""
        slr, sgr = args
        sg = 1.0 - sl
        if sg < sgr:
            kl = 1.0
            kg = 0.0
        elif sg >= 1.0 - slr:
            kl = 0.0
            kg = 1.0
        else:
            Shat = (sl - slr) / (1.0 - slr - sgr)
            kl = Shat**4
            kg = (1.0 - Shat**2) * (1.0 - Shat) ** 2
        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slr, self._sgr]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 2:
            raise ValueError()
        self._slr = value[0]
        self._sgr = value[1]


class FattKlikoff(BaseRelativePermeability):
    _id = 6
    _name = "Fatt-Klikoff"

    def __init__(self, slr):
        """
        Fatt and Klikoff's function.

        After Fatt and Klikoff (1959).

        Parameters
        ----------
        slr : scalar
            Irreducible liquid saturation (RP(1)).

        """
        if slr >= 1.0:
            raise ValueError()
        self.parameters = [slr]

    def _eval(self, sl, *args):
        """Fatt and Klikoff's function."""
        (slr,) = args
        Seff = (sl - slr) / (1.0 - slr) if sl > slr else 0.0
        kl = Seff**3
        kg = (1.0 - Seff) ** 3
        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slr]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 1:
            raise ValueError()
        self._slr = value[0]


class Grant(BaseRelativePermeability):
    _id = 4
    _name = "Grant"

    def __init__(self, slr, sgr):
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
        if slr + sgr >= 1.0:
            raise ValueError()
        self.parameters = [slr, sgr]

    def _eval(self, sl, *args):
        """Grant's curve."""
        slr, sgr = args
        sg = 1.0 - sl
        if sg < sgr:
            kl = 1.0
            kg = 0.0
        elif sg >= 1.0 - slr:
            kl = 0.0
            kg = 1.0
        else:
            Shat = (sl - slr) / (1.0 - slr - sgr)
            kl = Shat**4
            kg = 1.0 - kl
        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slr, self._sgr]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 2:
            raise ValueError()
        self._slr = value[0]
        self._sgr = value[1]


class Linear(BaseRelativePermeability):
    _id = 1
    _name = "Linear"

    def __init__(self, slmin, sgmin, slmax, sgmax):
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
        if slmin >= slmax:
            raise ValueError()
        if sgmin >= sgmax:
            raise ValueError()
        self.parameters = [slmin, sgmin, slmax, sgmax]

    def _eval(self, sl, *args):
        """Linear function."""
        slmin, sgmin, slmax, sgmax = args
        kl = (
            1.0
            if sl >= slmax
            else 0.0 if sl <= slmin else (sl - slmin) / (slmax - slmin)
        )

        sg = 1.0 - sl
        kg = (
            1.0
            if sg >= sgmax
            else 0.0 if sg <= sgmin else (sg - sgmin) / (sgmax - sgmin)
        )

        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slmin, self._sgmin, self._slmax, self._sgmax]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 4:
            raise ValueError()
        self._slmin = value[0]
        self._sgmin = value[1]
        self._slmax = value[2]
        self._sgmax = value[3]


class Pickens(BaseRelativePermeability):
    _id = 2
    _name = "Pickens"

    def __init__(self, x):
        """
        Gas perfect mobile function.

        Parameters
        ----------
        x : scalar
            RP(1).

        """
        self.parameters = [x]

    def _eval(self, sl, *args):
        """Gas perfect mobile function."""
        (x,) = args
        return sl**x, 1.0

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._x]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 1:
            raise ValueError()
        self._x = value[0]


class vanGenuchtenMualem(BaseRelativePermeability):
    _id = 7
    _name = "van Genuchten-Mualem"

    def __init__(self, m, slr, sls, sgr):
        """
        Van Genuchten-Mualem function.

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
        self.parameters = [m, slr, sls, sgr]

    def _eval(self, sl, *args):
        """Van Genuchten-Mualem function."""
        m, slr, sls, sgr = args
        if sl >= sls:
            kl = 1.0
            kg = 0.0

        else:
            Seff = (sl - slr) / (sls - slr)
            kl = (
                Seff**0.5 * (1.0 - (1.0 - Seff ** (1.0 / m)) ** m) ** 2
                if Seff > 0.0
                else 0.0
            )

            Shat = (sl - slr) / (1.0 - slr - sgr)
            Shat = max(Shat, 0.0)
            Shat = min(Shat, 1.0)
            kg = 1.0 - kl if sgr <= 0.0 else (1.0 - Shat**2) * (1.0 - Shat) ** 2

        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._m, self._slr, self._sls, self._sgr]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 4:
            raise ValueError()
        self._m = value[0]
        self._slr = value[1]
        self._sls = value[2]
        self._sgr = value[3]


class Verma(BaseRelativePermeability):
    _id = 8
    _name = "Verma"

    def __init__(self, slr=0.2, sls=0.895, a=1.259, b=-1.7615, c=0.5089):
        """
        Verma's function.

        After Verma et al. (1985).

        Parameters
        ----------
        slr : scalar
            Irreducible liquid saturation (RP(1)).
        sls : scalar
            Maximum liquid saturation (RP(2)).
        a : scalar
            A (RP(3)).
        b : scalar
            B (RP(4)).
        c : scalar
            C (RP(5)).

        """
        self.parameters = [slr, sls, a, b, c]

    def _eval(self, sl, *args):
        """Verma's function."""
        slr, sls, a, b, c = args
        Shat = (sl - slr) / (sls - slr)
        Shat = max(Shat, 0.0)
        Shat = min(Shat, 1.0)
        kl = Shat**3
        kg = a + b * Shat + c * Shat**2
        kg = max(kg, 0.0)
        kg = min(kg, 1.0)
        return kl, kg

    @property
    def parameters(self):
        """Return model parameters."""
        return [self._slr, self._sls, self._a, self._b, self._c]

    @parameters.setter
    def parameters(self, value):
        if len(value) != 5:
            raise ValueError()
        self._slr = value[0]
        self._sls = value[1]
        self._a = value[2]
        self._b = value[3]
        self._c = value[4]
