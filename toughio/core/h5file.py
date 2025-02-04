from __future__ import annotations
from collections.abc import Sequence
from numpy.typing import ArrayLike
from typing import Literal, Optional
from types import TracebackType

import os
import pathlib

import h5py
import numpy as np

from .history_output import HistoryOutput
from .output import ConnectionOutput, ElementOutput


class H5File:
    """
    Open an H5 container.

    Parameters
    ----------
    filename : str | PathLike
        H5 container file name.
    mode : {'r', 'w'}, default 'r'
        File opening mode.
    compression_opts : int, default 4
        Compression level for gzip compression. May be an integer from 0 to 9.
    exist_ok : bool, default False
        If True and *mode* = 'w', overwrite *filename* if it already exists.

    """
    __name__: str = "H5File"
    __qualname__: str = "toughio.H5File"

    def __init__(
        self,
        filename: str | os.PathLike,
        mode: Literal["r", "w"] = "r",
        compression_opts: int = 4,
        exist_ok: bool = False,
    ) -> None:
        """Initialize an H5 file."""
        self.filename = filename
        self.exist_ok = exist_ok

    def __enter__(self) -> H5File:
        """On context opening."""
        if self.mode == "w" and self.filename.is_file():
            if self.exist_ok:
                os.remove(self.filename)

            else:
                raise FileExistsError(f"could not create file '{self.filename.name}'")

        self._h5file = h5py.File(self.filename, self.mode)

        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException] = None,
        exc_val: Optional[BaseException] = None,
        exc_tb: Optional[TracebackType] = None,
    ) -> None:
        """On context closing."""
        self._h5file.close()

        if self.mode == "w" and (exc_type or exc_val or exc_tb):
            os.remove(self.filename)

    def __contains__(self, node: str) -> bool:
        """Return True if file contains a node."""
        return node in self._h5file

    def dump(
        self,
        obj: ConnectionOutput | ElementOutput | HistoryOutput,
    ) -> None:
        """
        Dump an output to container.

        Parameters
        ----------
        obj : toughio.ConnectionOutput | toughio.ElementOutput | toughio.HistoryOutput
            Output to dump to container.

        """
        def check_name(name: str, node: h5py.Group) -> str:
            """Check if name exists in node, increment otherwise."""
            count = 0
            name_ = name

            while name_ in node:
                count += 1
                name_ = f"{name} ({count})"

            return name_

        if self.mode == "r":
            raise ValueError("could not dump in read mode")

        try:
            name = obj.__class__.__name__

        except AttributeError:
            name = type(obj)
        
        if isinstance(obj, (ConnectionOutput, ElementOutput)):
            # /Output
            node = self._get_node("Output")

            # /Output/{type}
            if isinstance(obj, ConnectionOutput):
                # /Output/Connection
                node = self._get_node("Connection", node=node)

            else:
                # /Output/Element
                node = self._get_node("Element", node=node)

            if obj.time is not None:
                time = (
                    f"{int(obj.time)} s"
                    if obj.time < 3600.0
                    else f"{obj.time / 3600.0:.2f} hr"
                    if obj.time < 86400.0
                    else f"{obj.time / 86400.0:.2f} d"
                    if obj.time < 31536000.0
                    else f"{obj.time / 31536000.0:.2f} yr"
                )
                name = f"Time = {time}"

            else:
                name = f"Dataset {len(node)}"

            name = check_name(name, node)

            # /Output/{type}/{name}
            node = self._get_node(name, node=node)

            self._dump_dict("data", obj.data, node=node)
            self._dump_dict(
                "metadata",
                {
                    "time": obj.time,
                    "labels": (
                        obj.labels.tolist()
                        if obj.labels is not None
                        else obj.labels
                    ),
                },
                node=node,
            )

        elif isinstance(obj, HistoryOutput):
            # /History
            node = self._get_node("History")

            type_ = obj.type.capitalize()

            if type_ not in {"Connection", "Element", "Generator", "Rock"}:
                raise ValueError(f"could not dump history output with type '{obj.type}'")

            # /History/{type_}
            node = self._get_node(type_, node)

            if obj.label:
                name = obj.label

            else:
                name = f"Dataset {len(node)}"

            name = check_name(name, node)

            # /History/{type_}/{name}
            node = self._get_node(name, node=node)

            self._dump_dict("data", obj.to_dict(unit=True), node=node)
            self._dump_dict("metadata", {k: v for k, v in obj.metadata.items() if k not in {"label", "type"}}, node=node)

        else:
            raise ValueError(f"could not dump {name} to file")

    def load_connection_output(
        self,
        name_or_ind: str | int,
    ) -> ConnectionOutput:
        """
        Load a connection output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.ConnectionOutput
            Connection output.

        """
        return self._load_output("connection", name_or_ind)

    def load_element_output(
        self,
        name_or_ind: str | int,
    ) -> ElementOutput:
        """
        Load an element output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.ElementOutput
            Element output.

        """
        return self._load_output("element", name_or_ind)

    def load_connection_history(
        self,
        name_or_ind: str | int,
    ) -> HistoryOutput:
        """
        Load a connection history output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.HistoryOutput
            Connection history output.

        """
        return self._load_history("connection", name_or_ind)

    def load_element_history(
        self,
        name_or_ind: str | int,
    ) -> HistoryOutput:
        """
        Load an element history output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.HistoryOutput
            Element history output.

        """
        return self._load_history("element", name_or_ind)

    def load_generator_history(
        self,
        name_or_ind: str | int,
    ) -> HistoryOutput:
        """
        Load a generator history output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.HistoryOutput
            Generator history output.

        """
        return self._load_history("generator", name_or_ind)

    def load_rock_history(
        self,
        name_or_ind: str | int,
    ) -> HistoryOutput:
        """
        Load a rock history output.

        Parameters
        ----------
        name_or_ind : str | int
            Name or index of node in container.

        Returns
        -------
        toughio.HistoryOutput
            Rock history output.

        """
        return self._load_history("rock", name_or_ind)

    def list_connection_output(self) -> Sequence[str]:
        """
        List connection output names in container.

        Returns
        -------
        Sequence[str]
            List of connection output names.

        """
        return self._list("Output/Connection")

    def list_element_output(self) -> Sequence[str]:
        """
        List element output names in container.

        Returns
        -------
        Sequence[str]
            List of element output names.

        """
        return self._list("Output/Element")

    def list_connection_history(self) -> Sequence[str]:
        """
        List connection history output names in container.

        Returns
        -------
        Sequence[str]
            List of connection history output names.

        """
        return self._list("History/Connection")

    def list_element_history(self) -> Sequence[str]:
        """
        List element history output names in container.

        Returns
        -------
        Sequence[str]
            List of element history output names.

        """
        return self._list("History/Element")

    def list_generator_history(self) -> Sequence[str]:
        """
        List generator history output names in container.

        Returns
        -------
        Sequence[str]
            List of generator history output names.

        """
        return self._list("History/Generator")

    def list_rock_history(self) -> Sequence[str]:
        """
        List rock history output names in container.

        Returns
        -------
        Sequence[str]
            List of rock history output names.

        """
        return self._list("History/Rock")

    def _load_output(
        self,
        type_: Literal["connection", "element"],
        name_or_ind: str | int,
    ) -> ConnectionOutput | ElementOutput:
        """Load an output."""
        path = f"Output/{type_.capitalize()}"

        if isinstance(name_or_ind, str):
            if f"{path}/{name_or_ind}" not in self:
                raise ValueError(f"invalid {type_} output name '{name_or_ind}'")

            name = name_or_ind

        else:
            name = self._list(path)[name_or_ind]

        data = self._load_dict(f"{path}/{name}/data")
        metadata = self._load_dict(f"{path}/{name}/metadata")
        kwargs = {"data": data, **metadata}

        return (
            ConnectionOutput(**kwargs)
            if type_ == "connection"
            else ElementOutput(**kwargs)
        )

    def _load_history(
        self,
        type_: Literal["connection", "element", "generator", "rock"],
        name_or_ind: str | int,
    ) -> HistoryOutput:
        """Load an history output."""
        path = f"History/{type_.capitalize()}"

        if isinstance(name_or_ind, str):
            if f"{path}/{name_or_ind}" not in self:
                raise ValueError(f"invalid {type_} history name '{name_or_ind}'")

            name = name_or_ind

        else:
            name = self._list(path)[name_or_ind]

        data = self._load_dict(f"{path}/{name}/data")
        metadata = self._load_dict(f"{path}/{name}/metadata")
        metadata["label"] = name
        metadata["type"] = type_

        return HistoryOutput(data, metadata=metadata)

    def _list(self, name: str) -> Sequence[str]:
        """List output names."""
        node = self._get_node(name)
        node = node if node else {}

        return list(node)

    def _get_node(
        self,
        name: str,
        node: Optional[h5py.Group] = None,
    ) -> h5py.Dataset | h5py.Group | None:
        """Get or create a node."""
        node = node if node else self._h5file

        if name in node:
            return node.get(name)

        else:
            if self.mode != "r":
                return node.create_group(name, track_order=True)

            else:
                return None

    def _dump_data(
        self,
        name: str,
        data: str | ArrayLike,
        node: Optional[h5py.Group] = None,
    ) -> None:
        """Dump any data."""
        name = name.replace("/", "-")
        node = node if node else self._h5file

        if isinstance(data, str):
            node.create_dataset(
                name,
                data=data if data else "",
                dtype=h5py.string_dtype("utf-8"),
            )

        elif data is None:
            node.create_dataset(name, data=np.nan)

        elif np.ndim(data) == 0:
            node.create_dataset(name, data=data)

        elif isinstance(data, (list, tuple, np.ndarray)):
            node.create_dataset(
                name,
                data=data,
                compression="gzip",
                compression_opts=self.compression_opts,
            )

        else:
            raise ValueError()

    def _load_data(
        self,
        name: str,
        node: Optional[h5py.Group] = None,
        nan_is_none: bool = True,
    ) -> str | ArrayLike | None:
        """Load any data."""
        node = node if node else self._h5file
        node = node.get(name)

        if node is None:
            return None

        elif isinstance(node, h5py.Dataset):
            if node.ndim == 0:
                node = node[()]

                if isinstance(node, bytes):
                    return node.decode("utf-8")

                elif np.isnan(node):
                    return None if nan_is_none else np.nan

                else:
                    return node

            else:
                return (
                    np.array(node).astype("U")
                    if node.dtype.kind == "O"
                    else np.array(node)
                )

        else:
            raise ValueError(f"could not load '{name}' as array")

    def _dump_dict(
        self,
        name: str,
        data: dict,
        node: Optional[h5py.Group] = None,
        **kwargs
    ) -> None:
        """Dump a dict."""
        node = node if node else self._h5file
        node = self._get_node(name, node=node)

        for k, v in data.items():
            if isinstance(v, dict):
                self._dump_dict(k, v, node=node, **kwargs)

            elif isinstance(v, str):
                self._dump_data(k, v, node=node)

            else:
                self._dump_data(k, v, node=node, **kwargs)

    def _load_dict(
        self,
        name: str,
        node: Optional[h5py.Group] = None,
        nan_is_none: bool = True,
    ) -> dict:
        """Load a dict."""
        node = node if node else self._h5file
        node = node.get(name)
        out = {}

        for k, v in node.items():
            try:
                out[k] = self._load_data(k, node, nan_is_none=nan_is_none)

            except ValueError:
                out[k] = self._load_dict(k, node, nan_is_none=nan_is_none)

        return out

    @property
    def filename(self) -> pathlib.Path:
        """Return file name."""
        return self._filename

    @filename.setter
    def filename(self, value: str | os.PathLike) -> None:
        """Set file name."""
        self._filename = pathlib.Path(value)

    @property
    def mode(self) -> str:
        """Return mode."""
        return self._mode

    @mode.setter
    def mode(self, value: Literal["r", "w"]) -> None:
        """Set mode."""
        self._mode = value

    @property
    def compression_opts(self) -> int:
        """Return compression level."""
        return self._compression_opts

    @compression_opts.setter
    def compression_opts(self, value: int) -> None:
        """Set compression level."""
        self._compression_opts = value

    @property
    def exist_ok(self) -> bool:
        """Return exist_ok."""
        return self._exist_ok

    @exist_ok.setter
    def exist_ok(self, value: bool) -> None:
        """Set exist_ok."""
        self._exist_ok = value
