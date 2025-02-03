from __future__ import annotations

import numpy as np


def scientific_notation(x: float, n: int) -> str:
    """
    Scientific notation with fixed number of characters.

    Parameters
    ----------
    x : float
        Value to format.
    n : int
        Length of output string representation.

    Returns
    -------
    str
        The string representation of the floating point value.

    Note
    ----
    This function maximizes accuracy given a fixed number of characters.

    """
    tmp = np.format_float_scientific(
        x,
        unique=True,
        trim="0",
        exp_digits=0,
        sign=False,
    )
    tmp = tmp.replace("+", "")

    if len(tmp) > n:
        significand, exponent = tmp.split("e")
        significand = significand[: n - len(tmp)]

        return f"{significand}e{exponent}"

    else:
        return tmp