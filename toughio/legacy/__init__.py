from . import flac3d
from ._helpers import (
    extrude_to_3d,
    from_meshio,
    from_pyvista,
    read_mesh,
    read_time_series,
    register_mesh,
    write_mesh,
    write_time_series,
)


def _patch_meshio():
    from meshio import register_format

    register_format("flac3d", [".f3grid"], flac3d.read, {"flac3d": flac3d.write})


_patch_meshio()
