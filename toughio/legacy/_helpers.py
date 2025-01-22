from __future__ import annotations
from typing import Callable, Optional
from numpy.typing import ArrayLike

import os

import meshio
import numpy as np
import pvgridder as pvg
import pyvista as pv

from ..core import Mesh


def extrude_to_3d(
    mesh: pv.StructuredGrid | pv.UnstructuredGrid,
    height: Optional[ArrayLike] = None,
    axis: int = 2,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    from pvgridder import get_dimension, MeshExtrude

    if get_dimension(mesh) != 2:
        raise ValueError("could not extrude 3D mesh")

    height = height if height is not None else 1.0
    height = [height] if np.ndim(height) == 0 else height

    extrude = pvg.MeshExtrude(mesh)
    for h in height:
        extrude = extrude.add(np.insert(np.zeros(2), axis, h))

    return extrude.generate_mesh()


def from_meshio(mesh: meshio.Mesh) -> Mesh:
    return Mesh(mesh)


def from_pyvista(mesh: pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid) -> Mesh:
    return Mesh(mesh)


def get_material_key(cell_data: dict) -> str:
    """Get key of material data in cell_data."""
    from meshio._common import _pick_first_int_data

    key, _ = _pick_first_int_data(cell_data)

    return key


def read_mesh(filename: str | os.PathLike, file_format: Optional[str] = None) -> Mesh | dict:
    """
    Read mesh from file.

    Parameters
    ----------
    filename : str | PathLike
        Input file name.
    file_format : str, optional
        Output file format.

    Returns
    -------
    :class:`toughio.Mesh` | dict
        Imported mesh.

    """
    from .. import read_input

    if file_format == "tough":
        return read_input(filename, blocks=["ELEME", "CONNE", "COORD", "INCON"])

    else:
        mesh = (
            meshio.read(filename, file_format=file_format)
            if file_format
            else pv.read(filename)
        )

        return Mesh(mesh)


def read_time_series(filename: str | os.PathLike) -> tuple[Mesh, list[dict], ArrayLike]:
    """
    Read time series from XDMF file.

    Parameters
    ----------
    filename : str | PathLike
        Input file name.

    Returns
    -------
    :class:`toughio.Mesh`
        Output mesh.
    sequence of dict
        List of data for each time step.
    ArrayLike
        Time steps.

    """
    data, time_steps = [], []

    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()

        for k in range(reader.num_steps):
            t, _, cd = reader.read_data(k)
            data.append({k: np.concatenate(v) for k, v in cd.items()})
            time_steps.append(t)

    time_steps = np.array(time_steps)

    return Mesh(points, cells), data, time_steps


def register_mesh(file_format: str, extensions: list[str], reader: Callable, writer: Optional[Callable] = None):
    meshio.register_format(file_format, extensions, reader, {file_format: writer} if writer else {})


def write_mesh(filename: str | os.PathLike, mesh: Mesh, file_format: Optional[str] = None) -> None:
    """
    Write mesh to file.

    Parameters
    ----------
    filename : str | PathLike
        Output file name.
    mesh : :class:`toughio.Mesh`
        Mesh to export.
    file_format : str, optional
        Output file format.
    
    """
    mesh.write(filename, file_format=file_format)


def write_time_series(
    filename: str | os.PathLike,
    mesh: Mesh,
    data: list[dict],
    time_steps: Optional[ArrayLike] = None,
) -> None:
    """
    Write time series given data.

    Parameters
    ----------
    filename : str | PathLike
        Output file name.
    mesh : :class:`toughio.Mesh`
        Input mesh.
    data : sequence of dict
        List of data for each time step.
    time_steps : ArrayLike, optional
        Time steps.

    """
    mesh = mesh.copy()
    
    for k, v in data[0].items():
        mesh.add_data(k, v)

    mesh.to_xdmf(filename, other_data=data[1:], time_steps=time_steps)
    