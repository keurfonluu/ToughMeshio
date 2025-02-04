from __future__ import annotations
from collections.abc import Sequence
from numpy.typing import ArrayLike
from typing import Optional

from abc import ABC, abstractmethod

import os
import numpy as np


class Output(ABC):
    """
    Base class for output data.

    Parameters
    ----------
    data : dict
        Data arrays.
    time : scalar, optional
        Time step (in seconds).
    labels : ArrayLike, optional
        Labels of elements.

    """
    __name__: str = "Output"
    __qualname__: str = "toughio.Output"

    def __init__(
        self,
        data: dict,
        time: Optional[float] = None,
        labels: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize an output."""
        self.time = time
        self.data = data
        self.labels = labels

    @abstractmethod
    def __getitem__(self, islice: tuple) -> None:
        """Slice an output."""
        pass

    @abstractmethod
    def index(self, label: str, *args, **kwargs) -> None:
        """Get index of element or connection."""
        if self.labels is None:
            raise AttributeError("could not find labels")

    @property
    def n_data(self) -> int:
        """Return number of data points."""
        return len(self.data[list(self.data)[0]])

    @property
    def time(self) -> float:
        """Return time step (in seconds)."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        """Set time step."""
        self._time = value

    @property
    def data(self) -> dict:
        """Return data arrays."""
        return self._data

    @data.setter
    def data(self, value: dict) -> None:
        """Set data arrays."""
        self._data = value

    @property
    def labels(self) -> ArrayLike | None:
        """Return labels."""
        return self._labels

    @labels.setter
    def labels(self, value: ArrayLike) -> None:
        """Set labels."""
        if value is not None:
            if len(value) != self.n_data:
                raise ValueError()

            self._labels = np.asarray(value)

        else:
            self._labels = None


class ElementOutput(Output):
    """
    Element output data.

    Parameters
    ----------
    data : dict
        Data arrays.
    time : scalar, optional
        Time step (in seconds).
    labels : ArrayLike, optional
        Labels of elements.

    """
    __name__: str = "ElementOutput"
    __qualname__: str = "toughio.ElementOutput"

    def __init__(
        self,
        data: dict,
        time: Optional[float] = None,
        labels: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize an element output."""
        super().__init__(data=data, time=time, labels=labels)

    def __getitem__(
        self,
        islice: int | str | slice | Sequence[int | str],
    ) -> dict | ElementOutput:
        """
        Slice an element output.

        Parameters
        ----------
        islice : int | str | slice | Sequence[int | str]
            Indices or labels of elements to slice.

        Returns
        -------
        dict | toughio.ElementOutput
            Sliced element outputs.

        """
        if self.labels is None:
            raise AttributeError()

        if np.ndim(islice) == 0:
            if isinstance(islice, slice):
                islice = np.arange(self.n_data)[islice]

            else:
                islice = self.index(islice) if isinstance(islice, str) else islice

                return {k: v[islice] for k, v in self.data.items()}

        elif np.ndim(islice) == 1:
            islice = [self.index(i) if isinstance(i, str) else i for i in islice]

        else:
            raise ValueError()

        return ElementOutput(
            data={k: v[islice] for k, v in self.data.items()},
            time=self.time,
            labels=[self._labels[i] for i in islice],
        )

    def index(self, label: str) -> int:
        """
        Get index of element.

        Parameters
        ----------
        label : str
            Label of element.

        Returns
        -------
        int
            Index of element.

        """
        super().index(label)

        return np.flatnonzero(self.labels == label)[0]


class ConnectionOutput(Output):
    """
    Connection output data.

    Parameters
    ----------
    data : dict
        Data arrays.
    time : scalar, optional
        Time step (in seconds).
    labels : ArrayLike, optional
        Labels of connections.

    """
    __name__: str = "ConnectionOutput"
    __qualname__: str = "toughio.ConnectionOutput"

    def __init__(
        self,
        data: dict,
        time: Optional[float] = None,
        labels: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize a connection output."""
        super().__init__(data=data, time=time, labels=labels)

    def __getitem__(
        self,
        islice: int | str | slice | Sequence[int | str],
    ) -> dict | ConnectionOutput:
        """
        Slice a connection output.

        Parameters
        ----------
        islice : int | str | slice | Sequence[int | str]
            Indices or labels of connections to slice.

        Returns
        -------
        dict | toughio.ConnectionOutput
            Sliced connection outputs.

        """
        if self.labels is None:
            raise AttributeError()

        if np.ndim(islice) == 0:
            if isinstance(islice, str):
                islice = np.flatnonzero((self.labels == islice).any(axis=1))

            elif isinstance(islice, slice):
                islice = np.arange(self.n_data)[islice]

            else:
                return {k: v[islice] for k, v in self.data.items()}

        elif np.ndim(islice) <= 2:
            islice = [self.index(*i) if np.ndim(i) == 1 else i for i in islice]

        else:
            raise ValueError()

        return ConnectionOutput(
            data={k: v[islice] for k, v in self.data.items()},
            time=self.time,
            labels=[self._labels[i] for i in islice],
        )

    def index(self, label: str, label2: Optional[str] = None) -> int:
        """
        Get index of connection.

        Parameters
        ----------
        label : str
            Label of connection or label of first element of connection.
        label2 : str, optional
            Label of second element of connection (if *label* is the label of the first element).

        Returns
        -------
        int
            Index of connection.

        """
        super().index(label)
        labels = ["".join(label) for label in self.labels]

        if label2 is not None:
            label = f"{label}{label2}"

        return labels.index(label)

    def to_element(
        self,
        mesh,
        linear_data: Optional[Sequence[str]] = None,
        ignore_elements: Optional[Sequence[str]] = None,
    ) -> ElementOutput:
        """
        Project connection data to element centers.

        Parameters
        ----------
        mesh : dict | PathLike
            Mesh parameters or file name.
        linear_data : Sequence[str], optional
            Data keys corresponding to linear quantities. Linear quantities (e.g., Darcy's velocity) are not scaled by the interface area.
        ignore_elements : Sequence[str], optional
            Labels of elements to ignore.

        Returns
        -------
        toughio.ElementOutput
            Element output with projected data.

        References
        ----------
        .. [1] Painter, S. L., Gable, C. W., and Kelkar, S. (2012). "Pathline tracing on fully unstructured control-volume grids". Computational Geosciences, 16(4), 1125-1134
        
        """
        from .. import read_input

        if isinstance(mesh, (str, os.PathLike)):
            mesh = read_input(mesh, file_format="tough", blocks=["ELEME", "CONNE"])

        linear_data = list(linear_data) if linear_data is not None else {"VEL_L", "VEL_G", "V(LIQ.)", "V(GAS)"}
        ignore_elements = set(ignore_elements) if ignore_elements is not None else set()
        centers = {
            k: np.asarray(v["center"]) for k, v in mesh["elements"].items()
            if k not in ignore_elements
        }
        face_areas = {k: v["interface_area"] for k, v in mesh["connections"].items()}
        labels = list(centers)

        # Gather all data to build linear systems to solve
        connections = {
            label: {
                "index": [],
                "areas": [],
                "normals": [],
            }
            for label in labels
        }

        for i, (l1, l2) in enumerate(self.labels):
            if l1 in ignore_elements or l2 in ignore_elements:
                continue

            area = face_areas[f"{l1}{l2}"]
            normal = centers[l1] - centers[l2]
            normal /= np.linalg.norm(normal)

            connections[l1]["index"].append(i)
            connections[l1]["areas"].append(area)
            connections[l1]["normals"].append(normal)

            connections[l2]["index"].append(i)
            connections[l2]["areas"].append(area)
            connections[l2]["normals"].append(normal)

        connections = [
            {k: np.array(v) for k, v in connection.items()}
            for connection in connections.values()
        ]

        # Identify linear and volumetric data
        data = np.row_stack(list(self.data.values()))
        linear, volume = [], []

        for i, k in enumerate(self.data):
            if k in linear_data:
                linear.append(i)

            else:
                volume.append(i)

        if linear:
            data_linear = data[linear]

        if volume:
            data_volume = data[volume]

        # Approximate connection data
        Q = np.zeros((len(centers), 3, len(data)))

        for i, connection in enumerate(connections):
            if linear:
                G = connection["normals"]
                Q[i][:, linear] = np.linalg.pinv(G.T @ G) @ G.T @ data_linear[:, connection["index"]].T

            if volume:
                G = connection["areas"][:, np.newaxis] * connection["normals"]
                Q[i][:, volume] = np.linalg.pinv(G.T @ G) @ G.T @ data_volume[:, connection["index"]].T

        return ElementOutput(
            data={
                k: v for k, v in zip(self.data, Q.transpose((2, 0, 1)))
                if k not in {"X", "Y", "Z"}
            },
            time=self.time,
            labels=labels,
        )
