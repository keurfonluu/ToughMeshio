from __future__ import annotations
from collections.abc import Sequence
from numpy.typing import ArrayLike
from typing import Literal, Optional

import meshio
import numpy as np
import pyvista as pv

from .mesh import Mesh


class ParticleTracker:
    """
    Particle tracker class.
    
    Parameters
    ----------
    mesh : toughio.Mesh | meshio.Mesh | pyvista.UnstructuredGrid
        Input mesh.
    velocity : str | ArrayLike
        Velocity field data.

    """
    __name__: str = "ParticleTracker"
    __qualname__: str = "toughio.ParticleTracker"

    def __init__(
        self,
        mesh: Mesh | meshio.Mesh | pv.UnstructuredGrid,
        velocity: str | ArrayLike,
    ):
        """Initialize a particle tracker."""
        self.mesh = mesh
        self.velocity = velocity

    def get_velocity(self, point: ArrayLike) -> ArrayLike:
        """
        Get velocity vector at any point in mesh.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of query point.
        
        Returns
        -------
        ArrayLike
            Velocity vector at query point.

        """
        return pv.PolyData(point).sample(
            self.mesh,
            locator="cell",
            mark_blank=False,
        )["velocity"][0]

    def track(
        self,
        particles: ArrayLike,
        direction: Literal["forward", "backward"] = "forward",
        step_size: float = 1.0,
        time_ini: float = 0.0,
        max_step: int = 10000,
        max_time: float = np.inf,
        max_length: float = np.inf,
        end_points: Optional[ArrayLike] = None,
        radius: float = 1.0,
        window_length: int = 32,
        check_bounds: bool = False,
    ) -> ArrayLike | Sequence[ArrayLike]:
        """
        Track particle(s) given starting point coordinates.

        Parameters
        ----------
        particles : ArrayLike
            Starting point coordinates of particle(s) to track.
        direction : {'forward', 'backward'}, default 'forward'
            Direction of integration.
        step_size : scalar, default 1.0
            Pathline step size at each time step (in m).
        time_ini : scalar, default 0.0
            Initial time step (in second).
        max_step : int, default 10000
            Maximum number of steps.
        max_time : scalar, default Infinity
            Maximum time of pathline ending point.
        max_length : scalar, default Infinity
            Maximum length of pathline.
        end_points : ArrayLike, optional
            Ending point coordinates.
        radius : scalar, default 1.0
            Ending point radius (in m). The tracking algorithm stops if the pathline is
            within *radius* meters of an ending point.
        window_length : int, default 32
            Window length for oscillation detection.
        check_bounds : bool, default False
            If True, a particle is out of bound if it is not contained by any cell of
            the mesh, otherwise a simple box condition is applied (faster but less accurate).

        Returns
        -------
        ArrayLike | Sequence[ArrayLike]
            Output pathline(s).

        """
        step_size = step_size if step_size is not None else 1.0
        time_ini = time_ini if time_ini is not None else 0.0
        max_step = max_step if max_step is not None else 10000
        max_time = max_time if max_time is not None else np.inf
        max_length = max_length if max_length is not None else np.inf
        end_points = np.asarray(end_points) if end_points is not None else []
        radius = radius if radius is not None else 1.0
        window_length = window_length if window_length is not None else 32

        if direction == "forward":
            direction = 1.0

        elif direction == "backward":
            direction = -1.0

        else:
            raise ValueError(f"invalid direction '{direction}'")

        args = (
            direction,
            step_size,
            time_ini,
            max_step,
            max_time,
            max_length,
            end_points,
            radius,
            window_length,
            check_bounds,
        )

        if np.ndim(particles) == 1:
            return self._track(particles, *args)

        elif np.ndim(particles) == 2:
            return [self._track(particle, *args) for particle in particles]

        else:
            raise ValueError()

    def _track(
        self,
        particle: ArrayLike,
        direction: float,
        step_size: float,
        time_ini: float,
        max_step: int,
        max_time: float,
        max_length: float,
        end_points: ArrayLike,
        radius: float,
        window_length: int,
        check_bounds: bool,
    ) -> ArrayLike:
        """Track a particle."""
        x, y, z = particle
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds

        path = np.empty((max_step + 1, 3))
        time = np.empty(max_step + 1)
        path[0] = x, y, z
        time[0] = time_ini
        length = 0.0

        count = 1
        while True:
            in_end_point = False
            exceed_max_time = False
            exceed_max_length = False

            v = self.get_velocity(path[count - 1]) * direction
            vn = np.linalg.norm(v)

            # Stop if zero velocity
            if vn > 0.0:
                dx = step_size
                dt = dx / vn

            else:
                break

            if time[-1] + dt >= max_time:
                exceed_max_time = True
                dt = max_time - time[-1]
                v *= dt

            elif length + dx >= max_length:
                exceed_max_length = True
                dx = max_length - length
                v *= dx / vn

            else:
                v *= dt

            x += v[0]
            y += v[1]
            z += v[2]
            path[count] = x, y, z
            time[count] = time[count - 1] + dt
            length += dx

            # Stop is maximum tracking time is reached
            if exceed_max_time:
                break

            # Stop if maximum path length is reached
            if exceed_max_length:
                break

            # Stop if maximum number of iteration is reached
            if count >= max_step:
                break

            # Stop if particle is out of bound
            if check_bounds:
                if self.mesh.find_containing_cell(path[count]) == -1:
                    break

            else:
                if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
                    break

            # Stop if oscillation is detected
            # Oscillation is here characterized by fast changes in particle's direction (the particle is going back and forth)
            # This usually results in rapid changes in the sign of the difference of its position between two consecutive steps
            if count >= window_length:
                if (
                    np.abs(
                        np.sign(
                            np.diff(path[count - window_length : count], axis=0)
                        ).mean(axis=0)
                    )
                    < 0.5
                ).all():
                    break

            # Stop if particle is within distance of specified ending points
            for end_point in end_points:
                if np.linalg.norm(path[-1] - end_point) <= radius:
                    in_end_point = True
                    break

            if in_end_point:
                break

            count += 1

        return path[: count + 1]

    @property
    def mesh(self) -> pv.UnstructuredGrid:
        """Return mesh."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Mesh | meshio.Mesh | pv.UnstructuredGrid) -> None:
        """Set mesh."""
        if isinstance(value, Mesh):
            self._mesh = value.to_pyvista()

        elif isinstance(value, meshio.Mesh):
            self._mesh = pv.from_meshio(value)

        elif isinstance(value, pv.UnstructuredGrid):
            self._mesh = value.copy(deep=True)

        else:
            raise ValueError("invalid input mesh")

    @property
    def velocity(self) -> ArrayLike:
        """Return velocity field data."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: str | ArrayLike) -> None:
        """Set velocity field data."""
        if isinstance(value, str):
            if value not in self.mesh.cell_data:
                raise ValueError(f"invalid velocity data '{value}'")

            velocity = self.mesh.cell_data[value]

        elif isinstance(value, np.ndarray):
            if value.shape != (self.mesh.n_cells, 3):
                raise ValueError(f"velocity array size mismatch (expected {(self.mesh.n_cells, 3)}, got {value.shape})")

            velocity = value

        else:
            raise ValueError("invalid velocity data")

        self.mesh.clear_data()
        self.mesh.cell_data["velocity"] = velocity
        self.mesh = self.mesh.cell_data_to_point_data()
