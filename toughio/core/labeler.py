from __future__ import annotations
from numpy.typing import ArrayLike

from string import ascii_uppercase

import numpy as np


class Labeler:
    def __init__(self, label_length: int) -> None:
        self.label_length = label_length

    def __call__(self, n: int) -> ArrayLike:
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
        return self._label_length

    @label_length.setter
    def label_length(self, value: int) -> None:
        self._label_length = value
