from ..._common import filetype_from_filename, register_format

__all__ = [
    "register",
    "read",
    "write",
]


_extension_to_filetype = {}
_reader_map = {}
_writer_map = {}


def register(file_format, extensions, reader, writer=None):
    """
    Register a new input format.

    Parameters
    ----------
    file_format : str
        File format to register.
    extensions : array_like
        List of extensions to associate to the new format.
    reader : callable
        Read fumction.
    writer : callable or None, optional, default None
        Write function.

    """
    register_format(
        fmt=file_format,
        ext_to_fmt=_extension_to_filetype,
        reader_map=_reader_map,
        writer_map=_writer_map,
        extensions=extensions,
        reader=reader,
        writer=writer,
    )


def read(filename, file_format=None, **kwargs):
    """
    Read TOUGH input file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Input file name or buffer.
    file_format : str ('tough', 'toughreact-flow', 'json') or None, optional, default None
        Input file format.

    Other Parameters
    ----------------
    label_length : int or None, optional, default None
        Only if ``file_format = "tough"``. Number of characters in cell labels.
    eos : str or None, optional, default None
        Only if ``file_format = "tough"``. Equation of State.

    Returns
    -------
    dict
        TOUGH input parameters.

    Note
    ----
    If ``file_format == 'tough'``, can also read `MESH`, `INCON` and `GENER` files.

    """
    if not (file_format is None or file_format in _reader_map):
        raise ValueError()

    fmt = (
        file_format
        if file_format
        else filetype_from_filename(filename, _extension_to_filetype)
    )
    fmt = fmt if fmt else "tough"

    return _reader_map[fmt](filename, **kwargs)


def write(filename, parameters, file_format=None, **kwargs):
    """
    Write TOUGH input file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Output file name or buffer.
    parameters : dict
        Parameters to export.
    file_format : str ('tough', 'toughreact-flow', 'json') or None, optional, default None
        Output file format.

    Other Parameters
    ----------------
    block : str {'all', 'gener', 'mesh', 'incon'} or None, optional, default None
        Only if ``file_format = "tough"``. Blocks to be written:
         - 'all': write all blocks,
         - 'gener': only write block GENER,
         - 'mesh': only write blocks ELEME, COORD and CONNE,
         - 'incon': only write block INCON,
         - None: write all blocks except blocks defined in `ignore_blocks`.
    ignore_blocks : list of str or None, optional, default None
        Only if ``file_format = "tough"`` and `block` is None. Blocks to ignore.
    eos : str or None, optional, default None
        Only if ``file_format = "tough"``. Equation of State.
        If `eos` is defined in `parameters`, this option will be ignored.

    """
    if not isinstance(parameters, dict):
        raise TypeError()
    if not (file_format is None or file_format in _writer_map):
        raise ValueError()

    fmt = (
        file_format
        if file_format
        else filetype_from_filename(filename, _extension_to_filetype)
    )
    fmt = fmt if fmt else "tough"

    _writer_map[fmt](filename, parameters, **kwargs)
