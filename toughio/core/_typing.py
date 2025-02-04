from __future__ import annotations
from typing import Union

import pyvista as pv


GridLike = Union[pv.ExplicitStructuredGrid, pv.RectilinearGrid, pv.StructuredGrid, pv.UnstructuredGrid]
