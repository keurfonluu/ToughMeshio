from __future__ import annotations
from typing import Any, Optional

import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from io import TextIOWrapper

import numpy as np


def str2format(fmt: str) -> Sequence[str]:
    """Convert a string to a list of formats."""
    token_to_format = {
        "s": "",
        "S": "",
        "d": "g",
        "f": "f",
    }

    base_fmt = "{{:{}}}"
    out = [
        (
            base_fmt.format(f"{token[:-1]}.{token[:-1]}")
            if token[-1].lower() == "s"
            else base_fmt.format(f">{token[:-1]}{token_to_format[token[-1]]}")
        )
        for token in fmt.split(",")
    ]

    return out


def get_label_length(label: str) -> int:
    """Get length of cell label."""
    label_length = 5
    while label_length < len(label) and label[label_length].isdigit():
        label_length += 1

    return max(min(label_length, 9), 5)


def register_format(
    fmt: str,
    ext_to_fmt: dict,
    reader_map: dict,
    writer_map: dict,
    extensions: Sequence[str],
    reader: Optional[Callable] = None,
    writer: Optional[Callable] = None,
) -> None:
    """Register a new format."""
    for ext in extensions:
        ext_to_fmt[ext] = fmt

    if reader is not None:
        reader_map[fmt] = reader

    if writer is not None:
        writer_map[fmt] = writer


def filetype_from_filename(
    filename: str,
    ext_to_fmt: dict,
    default: str = "",
) -> str:
    """Determine file type from its extension."""
    if not isinstance(filename, TextIOWrapper):
        ext = os.path.splitext(filename)[1].lower()

        return ext_to_fmt[ext] if ext in ext_to_fmt else default

    else:
        return default


@contextmanager
def open_file(path_or_buffer: str | os.PathLike | TextIOWrapper, mode: Literal["r", "w"]) -> TextIOWrapper:
    """Open file or buffer."""
    def is_buffer(obj: Any, mode: Literal["r", "w"]) -> bool:
        return ("r" in mode and hasattr(obj, "read")) or (
            "w" in mode and hasattr(obj, "write")
        )

    if is_buffer(path_or_buffer, mode):
        yield path_or_buffer

    else:
        with open(path_or_buffer, mode) as f:
            yield f


def prune_values(data: dict | Sequence[float], value: Optional[float] = None) -> dict | Sequence[float]:
    """Remove values from dict or trailing values from list."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v != value}

    elif isinstance(data, (list, tuple, np.ndarray)):
        return [x for i, x in enumerate(data) if any(xx != value for xx in data[i:])]

    else:
        return data


block_to_format = {
    "DIMEN": ",".join(8 * ["10d"]),
    "ROCKS": {
        1: "5s,5d,10f,10f,10f,10f,10f,10f,10f",
        2: "10f,10f,10f,10f,10f,10f,10f,10f,10f",  # Tortuosity can be <0 in TOUGHREACT
        # TOUGHREACT
        3: "5d,5s,10f,10f,10f",
        4: "5d,5s,14f,14f,14f,14f",
    },
    "RPCAP": "5d,5s,10f,10f,10f,10f,10f,10f,10f",
    "FLAC": {
        1: ",".join(16 * ["5d"]),
        2: "10d,10f,10f,10f,10f,10f,10f,10f",
        3: "5d,5s,10f,10f,10f,10f,10f,10f,10f",
    },
    "CHEMP": {
        1: "5d",
        2: "20s",
        3: ",".join(5 * ["10f"]),
    },
    "NCGAS": {1: "5d", 2: "10s"},
    "MULTI": ",".join(5 * ["5d"]),
    "SELEC": {1: ",".join(16 * ["5d"]), 2: ",".join(8 * ["10f"])},
    "SOLVR": "1d,2s,2s,3s,2s,10f,10f",
    "PARAM": {
        1: "2d,2d,4d,4d,4d,24S,10s,10f,10f",
        2: "10f,10f,10f,10f,5s,5s,10f,10f,10f",
        3: ",".join(8 * ["10f"]),
        4: "10f,10f,10s,10f,10f,10f",
        5: ",".join(4 * ["20f"]),
    },
    "INDOM": {0: ",".join(4 * ["20f"]), 5: "5s,5d"},  # 5d is for ECO2M and TMVOC
    "MOMOP": "50S",
    "TIMES": {1: "5d,5d,10f,10f", 2: ",".join(8 * ["10f"])},
    "HYSTE": ",".join(3 * ["5d"]),
    "FOFT": {5: "5s,5s,5d", 6: "6s,4s,5d", 7: "7s,3s,5d", 8: "8s,2s,5d", 9: "9s,1s,5d"},
    "COFT": {
        5: "10s,10s,5d",
        6: "12s,8s,5d",
        7: "14s,6s,5d",
        8: "16s,4s,5d",
        9: "18s,2s,5d",
    },
    "GOFT": {5: "5s,5s,5s,5d", 6: "6s,4s,5s,5d", 7: "7s,3s,5s,5d", 8: "8s,2s,5s,5d", 9: "9s,1s,5s,5d"},
    "ROFT": "5s,5s",
    "GENER": {
        0: ",".join(4 * ["14f"]),
        # Last integer is for KTAB value in TOUGHREACT
        5: "5s,5s,5d,5d,5d,5d,5s,4s,1s,10f,10f,10f,2d",
        6: "6s,5s,6d,4d,4d,5d,5s,4s,1s,10f,10f,10f,2d",
        7: "7s,5s,5d,4d,4d,5d,5s,4s,1s,10f,10f,10f,2d",
        8: "8s,5s,4d,4d,4d,5d,5s,4s,1s,10f,10f,10f,2d",
        9: "9s,5s,5d,3d,3d,5d,5s,4s,1s,10f,10f,10f,2d",
    },
    "DIFFU": ",".join(8 * ["10f"]),
    "OUTPU": {1: "20s", 2: "15s", 3: "20s,5d,5d"},
    "ELEME": {
        5: "5s,5d,5d,5s,10f,10f,10f,10f,10f,10f",
        6: "6s,5d,4d,5s,10f,10f,10f,10f,10f,10f",
        7: "7s,4d,4d,5s,10f,10f,10f,10f,10f,10f",
        8: "8s,4d,3d,5s,10f,10f,10f,10f,10f,10f",
        9: "9s,3d,3d,5s,10f,10f,10f,10f,10f,10f",
    },
    "COORD": ",".join(3 * ["20f"]),
    "CONNE": {
        5: "10s,5d,5d,5d,5d,10f,10f,10f,10f,10f",
        6: "12s,5d,4d,4d,5d,10f,10f,10f,10f,10f",
        7: "14s,5d,3d,3d,5d,10f,10f,10f,10f,10f",
        8: "16s,3d,3d,3d,5d,10f,10f,10f,10f,10f",
        9: "18s,3d,2d,2d,5d,10f,10f,10f,10f,10f",
    },
    "INCON": {
        0: ",".join(4 * ["20f"]),
        "default": {
            5: "5s,5d,5d,15f,10f,10f,10f,10f,10f,10f",
            6: "6s,5d,4d,15f,10f,10f,10f,10f,10f,10f",
            7: "7s,4d,4d,15f,10f,10f,10f,10f,10f,10f",
            8: "8s,4d,3d,15f,10f,10f,10f,10f,10f,10f",
            9: "9s,3d,3d,15f,10f,10f,10f,10f,10f,10f",
        },
        "eco2m": {
            5: "5s,5d,5d,15f,2d",
            6: "6s,5d,4d,15f,2d",
            7: "7s,4d,4d,15f,2d",
            8: "8s,4d,3d,15f,2d",
            9: "9s,3d,3d,15f,2d",
        },
        "tmvoc": {
            5: "5s,5d,5d,15f,2d",
            6: "6s,5d,4d,15f,2d",
            7: "7s,4d,4d,15f,2d",
            8: "8s,4d,3d,15f,2d",
            9: "9s,3d,3d,15f,2d",
        },
        "toughreact": {
            5: "5s,5d,5d,15f,15f,15f,15f",
            6: "6s,5d,4d,15f,15f,15f,15f",
            7: "7s,4d,4d,15f,15f,15f,15f",
            8: "8s,4d,3d,15f,15f,15f,15f",
            9: "9s,3d,3d,15f,15f,15f,15f",
        },
    },
    "MESHM": {
        1: "5s",
        "XYZ": {
            1: "10f",
            2: "5s,5d,10f",
            3: ",".join(8 * ["10f"]),
        },
        "RZ2D": {
            1: "5s",
            "RADII": {
                1: "5d",
                2: ",".join(8 * ["10f"]),
            },
            "EQUID": "5d,5s,10f",
            "LOGAR": "5d,5s,10f,10f",
            "LAYER": {
                1: "5d",
                2: ",".join(8 * ["10f"]),
            },
        },
        "MINC": {
            1: "5s,5s,5s,5s",
            2: "3d,3d,4s,10f,10f,10f,10f,10f,10f,10f",
            3: ",".join(8 * ["10f"]),
        },
    },
    "REACT": "25S",
}
