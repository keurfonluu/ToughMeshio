from __future__ import annotations
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod

import numpy as np


class BaseCurve(ABC):
    """Base class for curve models."""
    _short: str = None

    def __init__(self, *args) -> None:
        """Initialize a curve."""
        self.parameters = args

    def __repr__(self) -> str:
        """Display curve informations."""
        out = [f"{self.name} {self.__class__.__name__.lower()} model (I{self._short} = {self.id}):"]
        out += [
            f"    {self._short}({i + 1}) = {parameter}"
            for i, parameter in enumerate(self.parameters)
        ]

        return "\n".join(out)

    def __call__(self, sl: ArrayLike) -> ArrayLike:
        """Compute curve data given liquid saturation."""
        sl = np.asanyarray(sl)

        if np.logical_and(sl < 0.0, sl > 1.0).any():
            raise ValueError("liquid saturation must be between 0.0 and 1.0")

        return self._eval(sl, *self.parameters)

    @abstractmethod
    def _eval(self, sl: ArrayLike, *args) -> None:
        """Evaluate model at query points."""
        pass

    @abstractmethod
    def plot(*args) -> None:
        """Plot a curve."""
        pass

    @property
    def id(self) -> int:
        """Return model ID."""
        return self._id

    @property
    def name(self) -> str:
        """Return model name."""
        return self._name

    @property
    def parameters(self) -> ArrayLike:
        """Return model parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: ArrayLike) -> None:
        """Set model parameters."""
        self._parameters = value
