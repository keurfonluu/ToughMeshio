import numpy as np

from .._io.input._helpers import read
from ._meshmaker import cylindric_grid, structured_grid


def from_meshmaker(filename_or_dict, material="dfalt"):
    """
    Generate a mesh from a block MESHM.

    Parameters
    ----------
    filename_or_dict: str, pathlike, buffer or dict
        Input file name, buffer or parameters dict with key "meshmaker".
    material : str, optional, default 'dfalt'
        Default material name.

    """
    if isinstance(filename_or_dict, str):
        parameters = read(filename_or_dict, file_format="tough")

    else:
        parameters = filename_or_dict

    if "meshmaker" not in parameters:
        raise ValueError()

    if "type" not in parameters["meshmaker"]:
        raise ValueError()

    if parameters["meshmaker"]["type"] not in {"xyz", "rz2d", "rz2dl"}:
        raise ValueError()

    # XYZ
    if parameters["meshmaker"]["type"] == "xyz":
        dx_, dy_, dz_ = parse_xyz(parameters["meshmaker"]["parameters"])

        dx, dy, dz = [], [], []
        for increment in dx_:
            append(dx, **increment)
        for increment in dy_:
            append(dy, **increment)
        for increment in dz_:
            append(dz, **increment)

        if not len(dx):
            dx = [1.0]
        if not len(dy):
            dy = [1.0]
        if not len(dz):
            dz = [1.0]

        return structured_grid(dx, dy, dz, material=material)

    # RZ2D
    else:
        dr_, dz_ = parse_rz2d(parameters["meshmaker"]["parameters"])

        dr, dz = [], []
        for increment in dr_:
            append(dr, **increment)
        for increment in dz_:
            append(dz, **increment)

        if not len(dr):
            dr = [1.0]
        if not len(dz):
            dz = [1.0]

        return cylindric_grid(
            dr, dz, layer=parameters["meshmaker"]["type"] == "rz2dl", material=material
        )


def parse_xyz(parameters):
    """Parse input for XYZ mesh."""

    def parse(parameter):
        n = get_value(parameter, "n_increment", 1)
        if n <= 0:
            raise ValueError()

        sizes = get_value(parameter, "sizes", 0.0)
        if np.ndim(sizes) == 0:
            if sizes < 0.0:
                raise ValueError()

            tmp = [
                {
                    "n_increment": n,
                    "size": sizes,
                    "type": "uniform",
                    "radius_ref": None,
                }
            ]

        elif np.ndim(sizes) == 1:
            if (np.array(sizes) < 0.0).any():
                raise ValueError()

            tmp = [
                {
                    "n_increment": count,
                    "size": value,
                    "type": "uniform",
                    "radius_ref": None,
                }
                for count, value in zip(*squeeze(sizes))
            ]

        return tmp

    dx, dy, dz = [], [], []
    for parameter in parameters:
        if parameter["type"] == "nx":
            dx += parse(parameter)

        elif parameter["type"] == "ny":
            dy += parse(parameter)

        elif parameter["type"] == "nz":
            dz += parse(parameter)

    return dx, dy, dz


def parse_rz2d(parameters):
    """Parse input for RZ2D mesh."""
    rmax = 0.0
    dr, dz = [], []
    for parameter in parameters:
        if parameter["type"] == "radii":
            if len(parameter["radii"]) < 2:
                continue

            tmp = np.diff(parameter["radii"])
            if (tmp < 0.0).any():
                raise ValueError("radii must be sorted in ascending order.")

            rmax += tmp.sum()
            dr += [
                {
                    "n_increment": count,
                    "size": value,
                    "type": "uniform",
                    "radius_ref": None,
                }
                for count, value in zip(*squeeze(tmp))
            ]

        elif parameter["type"] == "equid":
            n = get_value(parameter, "n_increment", 1)
            if n <= 0:
                raise ValueError()

            size = get_value(parameter, "size", 0.0)
            if size < 0.0:
                raise ValueError()

            rmax += n * size
            dr.append(
                {
                    "n_increment": n,
                    "size": size,
                    "type": "uniform",
                    "radius_ref": None,
                }
            )

        elif parameter["type"] == "logar":
            n = get_value(parameter, "n_increment", 1)
            if n <= 0:
                raise ValueError()

            radius = get_value(parameter, "radius", 0.0)
            if radius <= rmax:
                raise ValueError()

            else:
                size = radius - rmax

            radius_ref = get_value(parameter, "radius_ref", 0.0)
            if radius_ref < 0.0:
                raise ValueError()

            rmax += size
            dr.append(
                {
                    "n_increment": n,
                    "size": size,
                    "type": "logarithmic",
                    "radius_ref": radius_ref,
                }
            )

        elif parameter["type"] == "layer":
            dz += [
                {
                    "n_increment": count,
                    "size": value,
                    "type": "uniform",
                    "radius_ref": None,
                }
                for count, value in zip(*squeeze(parameter["thicknesses"]))
            ]

    return dr, dz


def get_value(parameter, key, default):
    """Helper function to get value in dictionary."""
    return (
        default
        if key not in parameter
        or (isinstance(parameter[key], list) and not len(parameter[key]))
        or parameter[key] is None
        else parameter[key]
    )


def squeeze(data):
    """Squeeze a sequence."""
    count = [1]
    values = [data[0]]

    for value in data[1:]:
        if value == values[-1]:
            count[-1] += 1

        else:
            values.append(value)
            count.append(1)

    return count, values


def append(sizes, n_increment, size, type="uniform", radius_ref=None):
    """Append next increment size(s)."""
    if type == "uniform":
        sizes += [size] * n_increment

    elif type == "logarithmic":
        if not radius_ref:
            if not len(sizes):
                raise ValueError()

            radius_ref = sizes[-1]

        f = get_factor(n_increment, size, radius_ref)

        sizes.append(f * radius_ref)
        for _ in range(n_increment - 1):
            sizes.append(f * sizes[-1])


def get_factor(n_increment, radius, radius_ref):
    """
    Calculate f factor.

    Note
    ----
    Solve the equation DR * f + DR * f ** 2 + ... + DR * f ** n = RMAX.
    This polynomial should have only one positive root.

    """
    roots = np.roots(np.append(np.ones(n_increment), -radius / radius_ref))
    roots = roots[::-1]  # The real positive root seems to be the last one

    # Still check that it's real and positive
    f = None
    for root in roots:
        if np.isreal(root) and root > 0.0:
            f = root.real
            break

    if f is None:
        raise ValueError("no real positive root found.")

    return f
