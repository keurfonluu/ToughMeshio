from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional
from typing_extensions import Self

import os
import pathlib
from abc import ABC, abstractmethod

import meshio
import numpy as np
import pyvista as pv
import pvgridder as pvg


class BaseMesh(ABC):
    __name__: str = "BaseMesh"
    __qualname__: str = "toughio.BaseMesh"

    def __init__(
        self,
        *args,
        material: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if len(args) == 1:
            mesh, = args

            if isinstance(mesh, BaseMesh):
                self._pyvista = mesh.pyvista

            elif isinstance(mesh, (pv.StructuredGrid, pv.UnstructuredGrid)):
                self._pyvista = mesh.copy()

            elif isinstance(mesh, pv.RectilinearGrid):
                self._pyvista = mesh.cast_to_structured_grid()

            elif isinstance(mesh, pv.ExplicitStructuredGrid):
                self._pyvista = mesh.cast_to_unstructured_grid()

            elif isinstance(mesh, meshio.Mesh):
                if mesh.cell_sets:
                    mesh.cell_data["Material"] = [np.full(len(c.data), -1, dtype=int) for c in mesh.cells]

                    for i, (k, v) in enumerate(mesh.cell_sets.items()):
                        mesh.field_data[k] = np.array([i + 1, 3])

                        for ii, vv in enumerate(v):
                            if vv is not None and len(vv):
                                mesh.cell_data["Material"][ii][vv] = i + 1
                
                self._pyvista = pv.from_meshio(mesh)

                if mesh.field_data:
                    self.metadata["Material"] = {k: int(v[0]) for k, v in mesh.field_data.items()}

            elif isinstance(mesh, (str, os.PathLike)):
                if pathlib.Path(mesh).suffix == ".f3grid":
                    self._pyvista = Mesh(meshio.read(mesh)).pyvista

                else:
                    self._pyvista = pv.read(mesh)

            else:
                raise ValueError(f"could not initialize mesh from '{type(mesh)}'")

        elif len(args) == 2:
            self._pyvista = pv.from_meshio(meshio.Mesh(*args))

        else:
            raise ValueError()

        if isinstance(self.pyvista, pv.UnstructuredGrid):
            self._pyvista = pvg.extract_cells_by_dimension(self.pyvista)

        if not material:
            for k, v in self.data.items():
                if v.dtype.kind == "i":
                    material = k
                    break

        if material:
            if self.data[material].dtype.kind != "i":
                raise ValueError("could not set material to non-integer data")

            self.rename_data(material, "Material")

            try:
                self.metadata["Material"] = self.metadata.pop(material)

            except KeyError:
                pass

        if metadata is not None:
            self.metadata.update(metadata)

        self._labels = None
        self.label_length = None

    @abstractmethod
    def __getitem__(self, key: tuple[int | slice | ArrayLike]) -> None:
        pass

    def copy(self, deep=True) -> Self:
        mesh = self.__class__(self.pyvista.copy(deep=deep))
        mesh._labels = self._labels
        mesh.label_length = self.label_length
        mesh.metadata.update(self.metadata)

        return mesh

    def add_data(self, name: str, data: ArrayLike) -> None:
        self.data[name] = data[:self.n_cells]

    add_cell_data = add_data

    def find_enclosing_cell(self, points: ArrayLike) -> ArrayLike:
        """
        Find cell(s) that contains query point(s).

        Parameters
        ----------
        points : ArrayLike
            Coordinates of point(s) to query.
        
        Returns
        -------
        ArrayLike
            Indices of cells containing point(s).

        """
        return self.pyvista.find_containing_cell(points)

    def find_nearest_cell(self, points: ArrayLike) -> ArrayLike:
        """
        Find cells(s) nearest to query point(s).

        Parameters
        ----------
        points : ArrayLike
            Coordinates of point(s) to query.
        
        Returns
        -------
        ArrayLike
            Indices of cells nearest to point(s).

        """
        return self.pyvista.find_closest_cell(points)

    near = find_nearest_cell

    def rename_data(self, old: str, new: str) -> None:
        self.pyvista.rename_array(old, new, preference="cell")

    def set_material(self, material: str, ind: Optional[ArrayLike] = None):
        ind = ind if ind is not None else np.ones(self.n_cells, dtype=bool)

        if "Material" not in self.metadata:
            self.metadata["Material"] = {material: 0}
        
        try:
            imat = self.metadata["Material"][material]
        
        except KeyError:
            imat = len(self.metadata["Material"])
            self.metadata["Material"][material] = imat

        self.materials_digitized[ind] = imat

    def to_meshio(self) -> meshio.Mesh:
        return pv.to_meshio(self.pyvista)

    def to_pyvista(self) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return self.pyvista.copy(deep=True)

    def to_tough(
        self,
        filename: Optional[str | os.PathLike] = None,
        nodal_distance: Literal["line", "orthogonal"] = "line",
        material_name: Optional[dict] = None,
        gravity: Optional[ArrayLike] = None,
        incon: bool = False,
        **kwargs
    ) -> dict | None:
        def dot(A: ArrayLike, B: ArrayLike) -> ArrayLike:
            """Calculate the dot product when arrays A and B have the same shape."""
            return (A * B).sum(axis=1)

        def intersection_line_plane(centers: ArrayLike, lines: ArrayLike, points: ArrayLike, normals: ArrayLike) -> ArrayLike:
            """Calculate the intersection point between a line and a plane."""
            tmp = dot(points - centers, normals) / dot(lines, normals)

            return centers + lines * tmp[:, None]

        def distance_point_plane(centers: ArrayLike, points: ArrayLike, normals: ArrayLike, mask: ArrayLike) -> ArrayLike:
            """Calculate the orthogonal distance of a point to a plane."""
            return np.where(mask, 1.0e-9, np.abs(dot(centers - points, normals)))

        from .. import write_input

        material_name = material_name if material_name else {}
        gravity = gravity if gravity is not None else np.array([0.0, 0.0, -1.0])

        # Shallow copy current mesh
        mesh = self.copy(deep=False)
        labels = mesh.labels
        materials = mesh.materials
        dirichlet = mesh.dirichlet
        volumes = np.where(dirichlet, 1.0e50, mesh.volumes)
        centers = mesh.centers

        # Connection data
        connections, face_centers, face_normals, face_areas = mesh._compute_connection_properties()
        labels_1 = labels[connections[:, 0]]
        labels_2 = labels[connections[:, 1]]
        centers_1 = centers[connections[:, 0]]
        centers_2 = centers[connections[:, 1]]
        bounds_1 = dirichlet[connections[:, 0]]
        bounds_2 = dirichlet[connections[:, 1]]

        # Direction vectors of connection lines
        lines = centers_2 - centers_1
        lines /= np.linalg.norm(lines, axis=1)[:, None]

        # Permeability directions
        mask = lines != 0.0
        permability_directions = np.where(mask.sum(axis=1) == 1, mask.argmax(axis=1) + 1, 1)

        # Gravity angles
        angles = lines @ gravity

        # Nodal distances
        if nodal_distance == "line":
            fp = intersection_line_plane(centers_1, lines, face_centers, face_normals)
            distances_1 = np.where(bounds_1, 1.0e-9, np.linalg.norm(centers_1 - fp, axis=1))
            distances_2 = np.where(bounds_2, 1.0e-9, np.linalg.norm(centers_2 - fp, axis=1))

        elif nodal_distance == "orthogonal":
            distances_1 = distance_point_plane(centers_1, face_centers, face_normals, bounds_1)
            distances_2 = distance_point_plane(centers_2, face_centers, face_normals, bounds_2)

        # Write MESH file
        parameters = {
            "elements": {
                label: {
                    "material": (
                        material_name[material]
                        if material in material_name
                        else material
                    ),
                    "volume": volume,
                    "center": center,
                }
                for label, material, volume, center in zip(labels, materials, volumes, centers)
            },
            "connections": {
                f"{l1}{l2}": {
                    "permeability_direction": isot,
                    "nodal_distances": [d1, d2],
                    "interface_area": face_area,
                    "gravity_cosine_angle": angle,
                }
                for l1, l2, isot, d1, d2, face_area, angle in zip(
                    labels_1,
                    labels_2,
                    permability_directions,
                    distances_1,
                    distances_2,
                    face_areas,
                    angles,
                )
            }
        }

        # Initial conditions
        if incon:
            porosities = mesh.porosities
            permeabilities = mesh.permeabilities
            permeabilities = (
                np.expand_dims(permeabilities, axis=1)
                if permeabilities.ndim == 1
                else permeabilities
            )
            phase_compositions = mesh.phase_compositions
            initial_conditions = mesh.initial_conditions
            initial_conditions = (
                np.expand_dims(initial_conditions, axis=1)
                if initial_conditions.ndim == 1
                else initial_conditions
            )
            parameters["initial_conditions"] = {}

            for label, phi, k, index, values in zip(labels, porosities, permeabilities, phase_compositions, initial_conditions):
                tmp = {}

                if phi != 0.0:
                    tmp["porosity"] = phi

                if (k != 0.0).any():
                    tmp["userx"] = k[:3]

                if index != 0:
                    tmp["phase_composition"] = index

                if (values != 0.0).any():
                    tmp["values"] = values

                if tmp:
                    parameters["initial_conditions"][label] = tmp

        if filename:
            write_input(filename, parameters, block="mesh", **kwargs)

            if incon:
                write_input("INCON", parameters, block="incon", **kwargs)

        else:
            return parameters

    def to_xdmf(
        self,
        filename: str | os.PathLike,
        other_data: Optional[list[dict]] = None,
        time_steps: Optional[ArrayLike] = None,
    ) -> None:
        import shutil

        filename = pathlib.Path(filename)
        other_data = other_data if other_data else []
        data = [self.data, *other_data]
        
        nt = len(data)
        time_steps = np.asanyarray(time_steps) if time_steps is not None else np.arange(nt)

        if time_steps is not None and len(time_steps) != nt:
            raise ValueError(f"inconsitent number of data ({nt}) and time steps ({len(time_steps)})")

        # Convert to meshio
        mesh = pv.to_meshio(self.pyvista)
        points = mesh.points
        cells = mesh.cells

        # Split cell data arrays
        sizes = np.cumsum([len(c.data) for c in cells[:-1]])
        data = [{k: np.split(v, sizes) for k, v in cd.items()} for cd in data]

        # Sort data with time steps
        idx = time_steps.argsort()
        data = [data[i] for i in idx]
        time_steps = time_steps[idx]

        # Create parent folder if any
        parent_is_root = str(filename.parent) == "."

        if not parent_is_root:
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Write XDMF
        with meshio.xdmf.TimeSeriesWriter(filename) as writer:
            writer.write_points_cells(points, cells)

            for t, cd in zip(time_steps, data):
                writer.write_data(t, cell_data=cd)

        # Bug in meshio v5: H5 file is written in the current working directory
        if not parent_is_root:
            source = filename.with_suffix(".h5")

            if source.is_file():
                os.remove(source)

            shutil.move(source.name, filename.parent)

    def write_tough(
        self,
        filename: str | os.PathLike = "MESH",
        nodal_distance: Literal["line", "orthogonal"] = "line",
        material_name: Optional[dict] = None,
        gravity: Optional[ArrayLike] = None,
        incon: bool = False,
        **kwargs
    ) -> None:
        self.to_tough(
            filename,
            nodal_distance,
            material_name,
            gravity,
            incon,
        )

    def write(self, filename: str | os.PathLike, file_format: Optional[str] = None) -> None:
        if file_format:
            self.to_meshio().write(filename, file_format=file_format)

        else:
            self.pyvista.save(filename)

    def plot(self, **kwargs) -> None:
        if "scalars" not in kwargs:
            kwargs["scalars"] = self.materials
            
        self.pyvista.plot(**kwargs)

    def _compute_connection_properties(self) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        poly = (
            pvg.extract_cell_geometry(self.pyvista)
            .compute_cell_sizes(length=True, area=True, volume=False)
        )
        mask = (poly["vtkOriginalCellIds"] >= 0).all(axis=1)
        connections = poly["vtkOriginalCellIds"][mask]
        centers = poly.cell_centers().points[mask]

        if self.ndim == 3:
            normals = poly.compute_normals(point_normals=False)["Normals"][mask]
            lengths_or_areas = poly["Area"][mask]

        else:
            normals = np.diff(
                poly.points[np.delete(poly.lines.reshape((poly.n_lines, 3)), 0, axis=1)],
                axis=1,
            ).squeeze()
            normals = np.column_stack((normals[:, 2], np.zeros(poly.n_lines), -normals[:, 0]))
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
            normals = normals[mask]
            lengths_or_areas = poly["Length"][mask]

        return connections, centers, normals, lengths_or_areas

    def _get_property(self, name: str, default: Optional[ArrayLike] = None) -> ArrayLike:
        if name not in self.data:
            data = np.zeros(self.n_cells, dtype=float) if default is None else default
            self.add_data(name, data)

        return self.data[name]

    @property
    def centers(self) -> ArrayLike:
        return self.pyvista.cell_centers().points

    @property
    def data(self) -> dict:
        return self.pyvista.cell_data

    cell_data = data

    @property
    def dirichlet(self) -> ArrayLike:
        return self._get_property("Dirichlet", np.zeros(self.n_cells, dtype=bool))

    @dirichlet.setter
    def dirichlet(self, value: ArrayLike) -> None:
        self.add("Dirichlet", np.asanyarray(value).astype(bool))

    @property
    def initial_conditions(self) -> ArrayLike:
        return self._get_property("Initial Conditions")

    @initial_conditions.setter
    def initial_conditions(self, value: ArrayLike) -> None:
        self.add_data("Initial Conditions", np.asanyarray(value).astype(float))

    @property
    def labels(self) -> ArrayLike:
        from string import ascii_uppercase

        if self._labels is not None:
            labels = self._labels

        else:
            if not self.label_length:
                bins = 3185000 * 10 ** np.arange(5, dtype=np.int64) + 1
                self.label_length = np.digitize(self.n_cells, bins) + 5

            n = self.label_length - 3
            fmt = f"{{: >{n}}}"
            alpha = np.array(list(ascii_uppercase))
            numer = np.array([fmt.format(i) for i in range(10 ** n)])
            nomen = np.concatenate(([f"{i + 1:1}" for i in range(9)], alpha))

            q1, r1 = np.divmod(np.arange(self.n_cells), numer.size)
            q2, r2 = np.divmod(q1, nomen.size)
            q3, r3 = np.divmod(q2, nomen.size)
            _, r4 = np.divmod(q3, nomen.size)

            labels = np.array(["".join(name) for name in zip(alpha[r4], nomen[r3], nomen[r2], numer[r1])])

        return labels

    @labels.setter
    def labels(self, value: ArrayLike) -> None:
        value = np.array(value)

        if value.ndim != 1 or value.size != self.n_cells:
            raise ValueError(f"could not set label arrays of shape {value.shape} (expected shape ({self.n_cells},))")

        self._labels = value

    @property
    def label_length(self) -> int:
        return self._label_length

    @label_length.setter
    def label_length(self, value: int) -> None:
        self._label_length = value

    @property
    def materials(self) -> ArrayLike:
        metadata = self.metadata

        try:
            material_map = {v: k for k, v in metadata["Material"].items()}

            return np.array(
                [
                    material_map[material]
                    if material in material_map
                    else material
                    for material in self.materials_digitized
                ]
            )

        except KeyError:
            return self.materials_digitized

    @property
    def materials_digitized(self) -> ArrayLike:
        return self._get_property("Material", -np.ones(self.n_cells, dtype=int))

    @materials_digitized.setter
    def materials_digitized(self, value: ArrayLike) -> None:
        self.add_data("Material", np.asanyarray(value).astype(int))

    @property
    def metadata(self) -> dict:
        return self.pyvista.user_dict

    @property
    def ndim(self) -> int:
        return pvg.get_dimension(self.pyvista)

    @property
    def n_cells(self) -> int:
        return self.pyvista.n_cells

    @property
    def n_points(self) -> int:
        return self.pyvista.n_points

    @property
    def permeabilities(self) -> ArrayLike:
        return self._get_property("Permeability")

    @permeabilities.setter
    def permeabilities(self, value: ArrayLike) -> None:
        self.add_data("Permeability", np.asanyarray(value).astype(float))

    @property
    def phase_compositions(self) -> ArrayLike:
        return self._get_property("Phase Composition", np.zeros(self.n_cells, dtype=int))

    @phase_compositions.setter
    def phase_compositions(self, value: ArrayLike) -> None:
        self.add_data("Phase Composition", np.asanyarray(value).astype(int))

    @property
    def points(self) -> ArrayLike:
        return self.pyvista.points

    @property
    def porosities(self) -> ArrayLike:
        return self._get_property("Porosity")

    @porosities.setter
    def porosities(self, value: ArrayLike) -> None:
        self.add_data("Porosity", np.asanyarray(value).astype(float))

    @property
    def pyvista(self) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return self._pyvista

    @property
    def volumes(self) -> ArrayLike:
        is3d = self.ndim == 3
        key = "Volume" if is3d else "Area"

        return np.abs(self.pyvista.compute_cell_sizes(length=False, area=not is3d, volume=is3d)[key])


class Mesh(BaseMesh):
    __name__: str = "Mesh"
    __qualname__: str = "toughio.Mesh"

    def __init__(
        self,
        *args,
        material: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(*args, material=material, metadata=metadata)

    def __getitem__(self, key: tuple[int | slice | ArrayLike]) -> Mesh:
        return Mesh(self.pyvista.cast_to_unstructured_grid().extract_cells(key))

    def extrude_to_3d(self, height: ArrayLike = 1.0, axis: int = 2) -> Mesh:
        from ..legacy import extrude_to_3d

        return Mesh(extrude_to_3d(self.pyvista))

    def prune_duplicates(self) -> Mesh:
        return Mesh(self.pyvista.clean(produce_merge_map=False))


class CylindricMesh(BaseMesh):
    __name__: str = "CylindricMesh"
    __qualname__: str = "toughio.CylindricMesh"

    def __init__(
        self,
        *args,
        material: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(*args, material=material, metadata=metadata)

        if self.ndim != 2:
            raise ValueError("could not initialize a cylindric mesh from a 3D mesh")

        if not isinstance(self.pyvista, pv.StructuredGrid):
            raise ValueError("could not initialize a cylindric mesh from an unstructured mesh")

        if self.pyvista.dimensions[1] != 1:
            raise ValueError("could not initialize a cylindric mesh from a non vertical mesh")

        x = np.unique(self.points[:, 0])
        z = np.unique(self.points[:, 2])

        if x.size * z.size != np.prod(self.pyvista.dimensions):
            raise ValueError("could not initialize a cylindric mesh from a non rectilinear mesh")

        self.points[:, 0] -= x[0]
        self.points[:, 1] = 0.0

    def __getitem__(self, key: tuple[int | slice | ArrayLike]) -> CylindricMesh:
        raise NotImplementedError("could not slice cylindric mesh")

    def _compute_connection_properties(self) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        connections, centers, normals, lengths = super()._compute_connection_properties()

        # This is counterintuitive but the areas can be calculated with the same
        # instruction given the definitions of x (center) and L (length) for horizontal
        # and vertical connections:
        #  - horizontal: x is the coordinate of the vertical interface (i.e., radius)
        #    and L its "height"
        #       A = (2 * pi * x) * L
        #  - vertical: x is the center of the horizontal interface and L its "width"
        #       A = pi * (rout ** 2 - rin ** 2)
        #         = pi * ((x + L / 2) ** 2 - (x - L / 2) ** 2)
        #         = pi * (2 * x * L)
        areas = 2.0 * np.pi * centers[:, 0] * lengths

        return connections, centers, normals, areas

    @property
    def volumes(self) -> ArrayLike:
        centers = self.centers
        x = np.unique(self.points[:, 0])
        z = np.unique(self.points[:, 2])
        ix = np.searchsorted(x, centers[:, 0]) - 1
        iz = np.searchsorted(z, centers[:, 2]) - 1

        return 2.0 * np.pi * centers[:, 0] * np.diff(x)[ix] * np.diff(z)[iz]
