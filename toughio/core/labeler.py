from __future__ import annotations
from numpy.typing import ArrayLike

from string import ascii_uppercase

import numpy as np


class Labeler:
    """
    Labeler class.

    Parameters
    ----------
    label_length : int
        Number of characters.

    """
    __name__: str = "Labeler"
    __qualname__: str = "toughio.Labeler"

    def __init__(self, label_length: int) -> None:
        """Initialize a labeler."""
        self.label_length = label_length

    def __call__(self, n: int) -> ArrayLike:
        """
        Generate *n* labels.

        Parameters
        ----------
        n : int:
            Number of labels to generate.

        Returns
        -------
        ArrayLike
            List of *n* labels.
    
        """
        l = self.label_length - 3
        fmt = f"{{: >{l}}}"
        alpha = np.array(list(ascii_uppercase))
        numer = np.array([fmt.format(i) for i in range(10 ** l)])
        nomen = np.concatenate(([f"{i + 1:1}" for i in range(9)], alpha))

        q1, r1 = np.divmod(np.arange(n), numer.size)
        q2, r2 = np.divmod(q1, nomen.size)
        q3, r3 = np.divmod(q2, nomen.size)
        _, r4 = np.divmod(q3, nomen.size)

        return np.array(["".join(name) for name in zip(alpha[r4], nomen[r3], nomen[r2], numer[r1])])

    @property
    def label_length(self) -> int:
        """Return label length."""
        return self._label_length

    @label_length.setter
    def label_length(self, value: int) -> None:
        """Set label length."""
        self._label_length = value
