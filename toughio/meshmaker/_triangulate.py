import numpy as np

__all__ = [
    "triangulate",
]


def triangulate(points, material="dfalt"):
    """
    Generate a triangulated mesh from cloud points.

    Parameters
    ----------
    points : array_like
        Cooordinates of points.
    material : str, optional, default 'dfalt'
        Default material name.

    Returns
    -------
    toughio.Mesh
        Output triangulated mesh.

    """
    from ..core import CellBlock, Mesh
    
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("Triangulation requires scipy >= 0.9 to be installed.")

    if not isinstance(points, (list, tuple, np.ndarray)):
        raise TypeError()
    if np.ndim(points) != 2:
        raise ValueError()
    if np.shape(points)[1] not in {2, 3}:
        raise ValueError()
    if not isinstance(material, str):
        raise TypeError()

    points = np.asarray(points)
    connectivity = Delaunay(points).simplices
    cell_type = "triangle" if points.shape[1] == 2 else "tetra"
    cells = [CellBlock(cell_type, connectivity)]
    mesh = Mesh(points, cells)
    mesh.add_cell_data("material", np.ones(mesh.n_cells, dtype=np.int64))
    mesh.add_material(material, 1)

    return mesh
