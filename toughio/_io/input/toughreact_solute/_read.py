from ...._common import open_file, prune_values
from ....core import FileIterator, ReadError
from ..._common import read_record, to_float
from .._common import read_end_comments

__all__ = [
    "read",
]


def read(filename, mopr_11=0):
    """
    Read TOUGHREACT solute input file.

    Parameters
    ----------
    filename : str
        Input file name.
    mopr_11 : int, optional, default 0
        MOPR(11) value in file 'flow.inp'.

    """
    with open_file(filename, "r") as f:
        out = read_buffer(f, mopr_11)

    return out


def read_buffer(f, mopr_11=0):
    """Read TOUGHREACT solute input file."""
    parameters = {}
    fiter = FileIterator(f)

    try:
        parameters.update(_read_title(fiter))
        parameters.update(_read_options(fiter))
        parameters.update(_read_filenames(fiter))
        parameters["options"].update(_read_weights_coefficients(fiter))
        parameters["options"].update(_read_convergence(fiter))

        output, numbers = _read_output(fiter)
        parameters["options"].update(output["options"])
        parameters["flags"].update(output["flags"])
        parameters["output"] = {}

        elements = _read_elements(fiter, numbers["NWNOD"])
        if elements:
            parameters["output"]["elements"] = elements

        components = _read_indices_names(fiter, numbers["NWCOM"])
        if components:
            parameters["output"]["components"] = components

        minerals = _read_indices_names(fiter, numbers["NWMIN"])
        if minerals:
            parameters["output"]["minerals"] = minerals

        aqueous_species = _read_indices_names(fiter, numbers["NWAQ"])
        if aqueous_species:
            parameters["output"]["aqueous_species"] = aqueous_species

        surface_complexes = _read_indices_names(fiter, numbers["NWADS"])
        if surface_complexes:
            parameters["output"]["surface_complexes"] = surface_complexes

        exchange_species = _read_indices_names(fiter, numbers["NWEXC"])
        if exchange_species:
            parameters["output"]["exchange_species"] = exchange_species

        parameters.update(_read_default(fiter, mopr_11))
        parameters.update(_read_zones(fiter, mopr_11))

        convergence_bounds = _read_convergence_bounds(fiter)
        if convergence_bounds:
            parameters["options"].update(convergence_bounds)

        # Look for file ending
        while fiter.line.strip() != "end":
            try:
                _ = fiter.next()

            except StopIteration:
                break

        # End comments
        end_comments = read_end_comments(fiter)
        if end_comments:
            parameters["end_comments"] = end_comments

    except:
        raise ReadError(f"failed to parse line {fiter.count}.")

    return parameters


def _read_title(f):
    """Read title."""
    line = f.next(skip_empty=True, comments="#").strip()

    return {"title": line}


def _read_options(f):
    """Read options."""
    # Record 2
    line = f.next(skip_empty=True, comments="#").strip()
    data = [int(x) for x in line.split()]
    if len(data) < 8:
        raise ReadError()

    options = {
        "flags": {
            "iteration_scheme": data[0],
            "reactive_surface_area": data[1],
            "solver": data[2],
            "n_subiteration": data[3],
            "gas_transport": data[4],
            "verbosity": data[5],
            "feedback": data[6],
            "coupling": data[7],
            # ITDS_REACT is not used
        }
    }

    # Record 3
    line = f.next(skip_empty=True, comments="#").strip()
    data = [to_float(x) for x in line.split()]
    if len(data) < 4:
        raise ReadError()

    options["options"] = {
        "sl_min": data[0],
        "rcour": data[1],
        "ionic_strength_max": data[2],
        "mineral_gas_factor": data[3],
    }

    return options


def _read_filenames(f):
    """Read file names."""
    filenames = {"files": {}}

    line = f.next(skip_empty=True, comments="#")
    filenames["files"]["thermodynamic_input"] = line[:20].strip()

    for keys in ["iteration", "plot", "solid", "gas", "time"]:
        line = f.next()
        filenames["files"]["{}_output".format(keys)] = line[:20].strip()

    return filenames


def _read_weights_coefficients(f):
    """Read weights and diffusion coefficients."""
    line = f.next(skip_empty=True, comments="#").strip()
    data = [to_float(x) for x in line.split()]
    if len(data) < 4:
        raise ReadError()

    options = {
        "w_time": data[0],
        "w_upstream": data[1],
        "aqueous_diffusion_coefficient": data[2],
        "molecular_diffusion_coefficient": data[3],
    }

    return options


def _read_convergence(f):
    """Read convergence criterion."""
    line = f.next(skip_empty=True, comments="#").strip()
    data = line.split()
    if len(data) < 7:
        raise ReadError()

    options = {
        "n_iteration_tr": int(data[0]),
        "eps_tr": to_float(data[1]),
        "n_iteration_ch": int(data[2]),
        "eps_ch": to_float(data[3]),
        "eps_mb": to_float(data[4]),
        "eps_dc": to_float(data[5]),
        "eps_dr": to_float(data[6]),
    }

    return options


def _read_output(f):
    """Read output control variables."""
    output = {"options": {}, "flags": {}}

    line = f.next(skip_empty=True, comments="#").strip()
    data = [int(x) for x in line.split()]
    if len(data) < 10:
        raise ReadError()

    output = {
        "options": {"n_cycle_print": data[0]},
        "flags": {
            "aqueous_concentration_unit": data[7],
            "mineral_unit": data[8],
            "gas_concentration_unit": data[9],
        },
    }

    numbers = {
        "NWNOD": data[1],
        "NWCOM": data[2],
        "NWMIN": data[3],
        "NWAQ": data[4],
        "NWADS": data[5],
        "NWEXC": data[6],
    }

    return output, numbers


def _read_elements(f, n):
    """Read list of grid blocks."""
    elements = []

    if n > 0:
        fmt = ",".join(15 * ["5s"])
        line = f.next(skip_empty=True, comments="#").strip()
        data = read_record(line, fmt)
        elements += data

        while len(elements) < n:
            line = f.next().strip()
            data = read_record(line, fmt)
            elements += data

        return prune_values(elements[:n])

    elif n < 0:
        line = f.next(skip_empty=True, comments="#").strip()
        elements.append(line[:5].strip())
        while True:
            line = f.next().strip()

            if line:
                elements.append(line[:5].strip())

            else:
                break

        return elements

    else:
        _ = f.next(comments="#")  # Skip next blank line


def _read_indices_names(f, n):
    """Read indices or names."""
    if n > 0:
        line = f.next(comments="#").strip()
        data = [int(x) for x in line.split()]
        out = data

        return out[:n]

    elif n < 0:
        line = f.next(comments="#").strip()
        out = [line[:20].strip()]
        while True:
            line = f.next().strip()

            if line:
                out.append(line[:20].strip())

            else:
                break

        return out

    else:
        _ = f.next(comments="#")  # Skip next blank line


def _read_default(f, mopr_11=0):
    """Read default chemical property zones."""
    line = f.next(skip_empty=True, comments="#").strip()
    data = line.split()
    if len(data) < 9:
        raise ReadError()
    if mopr_11 == 1 and len(data) < 10:
        raise ReadError()

    default = {
        "default": {
            "initial_water": int(data[0]),
            "injection_water": int(data[1]),
            "mineral": int(data[2]),
            "initial_gas": int(data[3]),
            "adsorption": int(data[4]),
            "cation_exchange": int(data[5]),
            "permeability_porosity": int(data[6]),
            "linear_kd": int(data[7]),
            "injection_gas": int(data[8]),
        }
    }

    if len(data) == 10:
        if mopr_11 != 1:
            default["default"]["element"] = int(data[9])

        else:
            default["default"]["sedimentation_velocity"] = to_float(data[9])

    return default


def _read_zones(f, mopr_11=0):
    """Read chemical property zones."""
    zones = {"zones": {}}

    line = f.next(skip_empty=True, comments="#").rstrip()
    while True:
        # Normally, the list should end with a blank record
        # But some sample files don't yet end with an 'end' statement
        if not line.strip() or line.startswith("end"):
            break

        label = line[:5]
        data = line[5:].split()
        if len(data) < 11:
            raise ReadError()

        zones["zones"][label] = _parse_zones(data[2:], mopr_11)
        zones["zones"][label]["nseq"] = int(data[0])
        zones["zones"][label]["nadd"] = int(data[1])
        line = f.next(comments="#").rstrip()

    return zones


def _read_convergence_bounds(f):
    """Read convergence bounds."""
    while True:
        try:
            line = f.next().rstrip()

            if line.startswith("end"):
                return

            if line.startswith("CONVP"):
                break

        except StopIteration:
            return

    line = f.next(skip_empty=True, comments="#").strip()
    data = line.split()
    if len(data) < 4:
        raise ReadError()

    options = {
        "n_iteration_1": int(data[0]),
        "n_iteration_2": int(data[1]),
        "n_iteration_3": int(data[2]),
        "n_iteration_4": int(data[3]),
    }

    line = f.next(skip_empty=True, comments="#").strip()
    data = [to_float(x) for x in line.split()]
    if len(data) < 6:
        raise ReadError()

    options.update(
        {
            "t_increase_factor_1": data[0],
            "t_increase_factor_2": data[1],
            "t_increase_factor_3": data[2],
            "t_reduce_factor_1": data[3],
            "t_reduce_factor_2": data[4],
            "t_reduce_factor_3": data[5],
        }
    )

    return options


def _parse_zones(data, mopr_11):
    """Parse zones."""
    out = {
        "initial_water": int(data[0]),
        "injection_water": int(data[1]),
        "mineral": int(data[2]),
        "initial_gas": int(data[3]),
        "adsorption": int(data[4]),
        "cation_exchange": int(data[5]),
        "permeability_porosity": int(data[6]),
        "linear_kd": int(data[7]),
        "injection_gas": int(data[8]),
    }

    if len(data) == 10:
        if mopr_11 != 1:
            out["element"] = int(data[9])

        else:
            out["sedimentation_velocity"] = to_float(data[9])

    return out
