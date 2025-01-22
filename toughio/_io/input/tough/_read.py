from ...._common import block_to_format, get_label_length, open_file, prune_values
from ....core import FileIterator, ReadError
from ..._common import read_record
from .._common import read_end_comments
from ._helpers import read_model_record, read_primary_variables

__all__ = [
    "read",
]


oft_to_key = {
    "FOFT": "element_history",
    "COFT": "connection_history",
    "GOFT": "generator_history",
}


def read(
    filename,
    blocks=None,
    label_length=None,
    n_variables=None,
    eos=None,
    simulator="tough",
):
    """
    Read TOUGH input file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Input file name or buffer.
    blocks : list of str or None, optional, default None
        Blocks to read. If None, all blocks are read.
    label_length : int or None, optional, default None
        Number of characters in cell labels.
    n_variables : int or None, optional, default None
        Number of primary variables.
    eos : str or None, optional, default None
        Equation of State.

    Returns
    -------
    dict
        TOUGH input parameters.

    """
    if not (label_length is None or isinstance(label_length, int)):
        raise TypeError()
    if isinstance(label_length, int) and not 5 <= label_length < 10:
        raise ValueError()
    if simulator not in {"tough", "toughreact"}:
        raise ValueError()

    with open_file(filename, "r") as f:
        out = read_buffer(f, blocks, label_length, n_variables, eos, simulator)

    return out


def read_buffer(f, block_stack, label_length, n_variables, eos, simulator="tough"):
    """Read TOUGH input file."""
    from ._common import blocks

    # Block filters
    block_stack = block_stack if block_stack is not None else blocks
    block_stack = set(block_stack)

    parameters = {}
    flag = False

    # Title
    if "TITLE" in block_stack:
        block_stack.remove("TITLE")

        title = []
        while True:
            if len(title) >= 100:
                raise ValueError()

            line = f.readline().strip()

            if line[:5].rstrip().upper() not in blocks:
                title.append(line)

            else:
                break

        if title:
            title = title[0] if len(title) == 1 else title

        if title:
            parameters["title"] = title

        f.seek(0)

    # Loop over blocks
    # Some blocks (INCON, INDOM, PARAM) need to rewind to previous line but tell and seek are disabled by next
    # See <https://stackoverflow.com/questions/22688505/is-there-a-way-to-go-back-when-reading-a-file-using-seek-and-calls-to-next>
    fiter = FileIterator(f)

    try:
        for line in fiter:
            if line.startswith("DIMEN") and "DIMEN" in block_stack:
                block_stack.remove("DIMEN")
                parameters.update(_read_dimen(fiter))

            elif line.startswith("ROCKS") and "ROCKS" in block_stack:
                block_stack.remove("ROCKS")
                parameters.update(_read_rocks(fiter, simulator))

            elif line.startswith("RPCAP") and "RPCAP" in block_stack:
                block_stack.remove("RPCAP")
                rpcap = _read_rpcap(fiter)

                if "default" in parameters:
                    parameters["default"].update(rpcap)

                else:
                    parameters["default"] = rpcap

            elif line.startswith("REACT") and "REACT" in block_stack:
                block_stack.remove("REACT")
                react = _read_react(fiter)

                if "react" in parameters:
                    parameters["react"].update(react["react"])

                else:
                    parameters.update(react)

            elif line.startswith("FLAC") and "FLAC" in block_stack:
                block_stack.remove("FLAC")
                flac = _read_flac(fiter, parameters["rocks"])
                parameters["flac"] = flac["flac"]

                for k, v in flac["rocks"].items():
                    parameters["rocks"][k].update(v)

            elif line.startswith("CHEMP") and "CHEMP" in block_stack:
                block_stack.remove("CHEMP")
                parameters.update(_read_chemp(fiter))

            elif line.startswith("NCGAS") and "NCGAS" in block_stack:
                block_stack.remove("NCGAS")
                parameters.update(_read_ncgas(fiter))

            elif line.startswith("MULTI") and "MULTI" in block_stack:
                block_stack.remove("MULTI")
                parameters.update(_read_multi(fiter))

                if not n_variables:
                    n_variables = parameters["n_component"] + 1

            elif line.startswith("SOLVR") and "SOLVR" in block_stack:
                block_stack.remove("SOLVR")
                parameters.update(_read_solvr(fiter))

            elif line.startswith("INDEX") and "INDEX" in block_stack:
                block_stack.remove("INDEX")
                parameters["index"] = True

            elif line.startswith("START") and "START" in block_stack:
                block_stack.remove("START")
                parameters["start"] = True

            elif line.startswith("PARAM") and "PARAM" in block_stack:
                block_stack.remove("PARAM")
                param, n_variables = _read_param(fiter, n_variables, eos)
                parameters["options"] = param["options"]
                parameters["extra_options"] = param["extra_options"]

                if "default" in parameters:
                    parameters["default"].update(param["default"])

                else:
                    parameters["default"] = param["default"]

            elif line.startswith("SELEC") and "SELEC" in block_stack:
                block_stack.remove("SELEC")
                parameters.update(_read_selec(fiter))

            elif line.startswith("INDOM") and "INDOM" in block_stack:
                block_stack.remove("INDOM")
                indom, n_variables = _read_indom(fiter, n_variables, eos)

                for k, v in indom["rocks"].items():
                    parameters["rocks"][k].update(v)

            elif line.startswith("MOMOP") and "MOMOP" in block_stack:
                block_stack.remove("MOMOP")
                parameters.update(_read_momop(fiter))

            elif line.startswith("TIMES") and "TIMES" in block_stack:
                block_stack.remove("TIMES")
                parameters.update(_read_times(fiter))

            elif line.startswith("HYSTE") and "HYSTE" in block_stack:
                block_stack.remove("HYSTE")
                parameters.update(_read_hyste(fiter))

            elif line.startswith("FOFT") and "FOFT" in block_stack:
                block_stack.remove("FOFT")
                oft, label_length = _read_oft(fiter, "FOFT", label_length)
                parameters.update(oft)

            elif line.startswith("COFT") and "COFT" in block_stack:
                block_stack.remove("COFT")
                oft, label_length = _read_oft(fiter, "COFT", label_length)
                parameters.update(oft)

            elif line.startswith("GOFT") and "GOFT" in block_stack:
                block_stack.remove("GOFT")
                oft, label_length = _read_oft(fiter, "GOFT", label_length)
                parameters.update(oft)

            elif line.startswith("ROFT") and "ROFT" in block_stack:
                block_stack.remove("ROFT")
                parameters.update(_read_roft(fiter))

            elif line.startswith("GENER") and "GENER" in block_stack:
                block_stack.remove("GENER")
                gener, flag, label_length = _read_gener(fiter, label_length, simulator)
                parameters.update(gener)

                if flag:
                    break

            elif line.startswith("TIMBC") and "TIMBC" in block_stack:
                block_stack.remove("TIMBC")
                parameters.update(_read_timbc(fiter))

            elif line.startswith("DIFFU") and "DIFFU" in block_stack:
                block_stack.remove("DIFFU")
                parameters.update(_read_diffu(fiter))

            elif line.startswith("OUTPT") and "OUTPT" in block_stack:
                block_stack.remove("OUTPT")
                outpt = _read_outpt(fiter)

                if "react" in parameters:
                    parameters["react"].update(outpt["react"])

                else:
                    parameters.update(outpt)

            elif line.startswith("OUTPU") and "OUTPU" in block_stack:
                block_stack.remove("OUTPU")
                parameters.update(_read_outpu(fiter))

            elif line.startswith("ELEME") and "ELEME" in block_stack:
                block_stack.remove("ELEME")
                eleme, label_length = _read_eleme(fiter, label_length)
                parameters.update(eleme)
                parameters["coordinates"] = False

            elif line.startswith("COORD") and "COORD" in block_stack:
                block_stack.remove("COORD")
                coord = _read_coord(fiter)

                for k, v in zip(parameters["elements"], coord):
                    parameters["elements"][k]["center"] = v

                parameters["coordinates"] = True

            elif line.startswith("CONNE") and "CONNE" in block_stack:
                block_stack.remove("CONNE")
                conne, flag, label_length = _read_conne(fiter, label_length)
                parameters.update(conne)

                if flag:
                    break

            elif line.startswith("INCON") and "INCON" in block_stack:
                block_stack.remove("INCON")
                incon, flag, label_length, n_variables = _read_incon(
                    fiter, label_length, n_variables, eos, simulator
                )
                parameters.update(incon)

                if flag:
                    break

            elif line.startswith("MESHM") and "MESHM" in block_stack:
                block_stack.remove("MESHM")
                parameters.update(_read_meshm(fiter))

            elif line.startswith("POISE") and "POISE" in block_stack:
                block_stack.remove("POISE")
                poise = _read_poise(fiter)

                if "react" in parameters:
                    parameters["react"].update(poise["react"])

                else:
                    parameters.update(poise)

            elif line.startswith("NOVER") and "NOVER" in block_stack:
                block_stack.remove("NOVER")
                parameters["nover"] = True

            elif line.startswith("ENDCY") and "ENDCY" in block_stack:
                block_stack.remove("ENDCY")
                end_comments = read_end_comments(fiter)

                if end_comments:
                    parameters["end_comments"] = end_comments

            # Stop reading if block stack is empty
            if not block_stack:
                break

    except:
        raise ReadError(f"failed to parse line {fiter.count}.")

    if flag:
        end_comments = read_end_comments(fiter)

        if end_comments:
            if isinstance(end_comments, str):
                end_comments = [end_comments]

            parameters["end_comments"] = ["+++", *end_comments]

    return parameters


def _read_dimen(f):
    """Read DIMEN block data."""
    fmt = block_to_format["DIMEN"]
    dimen = {"array_dimensions": {}}

    # Record 1
    line = f.next()
    data = read_record(line, fmt)

    dimen["array_dimensions"].update(
        {
            "n_rocks": data[0],
            "n_times": data[1],
            "n_generators": data[2],
            "n_rates": data[3],
            "n_increment_x": data[4],
            "n_increment_y": data[5],
            "n_increment_z": data[6],
            "n_increment_rad": data[7],
        }
    )

    # Record 2
    line = f.next()
    data = read_record(line, fmt)

    dimen["array_dimensions"].update(
        {
            "n_properties": data[0],
            "n_properties_times": data[1],
            "n_regions": data[2],
            "n_regions_parameters": data[3],
            "n_ltab": data[4],
            "n_rpcap": data[5],
            "n_elements_timbc": data[6],
            "n_timbc": data[7],
        }
    )

    return dimen


def _read_rocks(f, simulator="tough"):
    """Read ROCKS block data."""
    fmt = block_to_format["ROCKS"]
    rocks = {"rocks": {}}

    while True:
        line = f.next()

        if line.strip():
            # Record 1
            data = read_record(line, fmt[1])
            rock = data[0]
            rocks["rocks"][rock] = {
                "density": data[2],
                "porosity": data[3],
                "permeability": data[4] if len(set(data[4:7])) == 1 else data[4:7],
                "conductivity": data[7],
                "specific_heat": data[8],
            }

            nad = data[1] if data[1] else 0
            if nad:
                # Record 2
                line = f.next()
                data = read_record(line, fmt[2])
                rocks["rocks"][rock].update(
                    {
                        "compressibility": data[0],
                        "expansivity": data[1],
                        "conductivity_dry": data[2],
                        "tortuosity": data[3],
                        "klinkenberg_parameter": data[4],
                        "distribution_coefficient_3": data[5],
                        "distribution_coefficient_4": data[6],
                        "tortuosity_exponent": data[7],
                        "porosity_crit": data[8],
                    }
                )

            if nad >= 2:
                # TOUGHREACT
                if simulator == "toughreact" and nad >= 4:
                    line = f.next()
                    if line.strip():
                        rocks["rocks"][rock]["react_tp"] = read_model_record(
                            line, fmt[3]
                        )

                    line = f.next()
                    if line.strip():
                        rocks["rocks"][rock]["react_hcplaw"] = read_model_record(
                            line, fmt[4]
                        )

                rocks["rocks"][rock].update(_read_rpcap(f))

        else:
            break

    rocks["rocks"] = {k: prune_values(v) for k, v in rocks["rocks"].items()}

    return rocks


def _read_rpcap(f):
    """Read RPCAP block data."""
    fmt = block_to_format["RPCAP"]
    rpcap = {}

    for key in ["relative_permeability", "capillarity"]:
        line = f.next()
        if line.strip():
            rpcap[key] = read_model_record(line, fmt)

    return rpcap


def _read_react(f):
    """Read REACT block data."""
    fmt = block_to_format["REACT"]

    line = f.next()
    data = read_record(line, fmt)
    react = {
        "react": {
            "options": {i + 1: int(x) for i, x in enumerate(data[0]) if x.isdigit()}
        }
    }

    return react


def _read_flac(f, rocks_order):
    """Read FLAC block data."""
    fmt = block_to_format["FLAC"]
    flac = {"rocks": {}, "flac": {}}

    # Record 1
    line = f.next()
    data = read_record(line, fmt[1])
    flac["flac"]["creep"] = data[0]
    flac["flac"]["porosity_model"] = data[1]
    flac["flac"]["version"] = data[2]

    # Additional records
    for rock in rocks_order:
        flac["rocks"][rock] = {}

        line = f.next()
        data = read_record(line, fmt[2])
        flac["rocks"][rock]["permeability_model"] = {
            "id": data[0],
            "parameters": prune_values(data[1:]),
        }

        line = f.next()
        data = read_record(line, fmt[3])
        flac["rocks"][rock]["equivalent_pore_pressure"] = {
            "id": data[0],
            "parameters": prune_values(data[2:]),
        }

    flac["flac"] = prune_values(flac["flac"])

    return flac


def _read_chemp(f):
    """Read CHEMP block data."""
    fmt = block_to_format["CHEMP"]
    chemp = {"chemical_properties": {}}

    # Record 1
    line = f.next()
    n = read_record(line, fmt[1])[0]

    # Record 2
    for _ in range(n):
        tmp = {}

        line = f.next()
        data = read_record(line, fmt[2])
        chem = data[0]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["temperature_crit"] = data[0]
        tmp["pressure_crit"] = data[1]
        tmp["compressibility_crit"] = data[2]
        tmp["pitzer_factor"] = data[3]
        tmp["dipole_moment"] = data[4]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["boiling_point"] = data[0]
        tmp["vapor_pressure_a"] = data[1]
        tmp["vapor_pressure_b"] = data[2]
        tmp["vapor_pressure_c"] = data[3]
        tmp["vapor_pressure_d"] = data[4]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["molecular_weight"] = data[0]
        tmp["heat_capacity_a"] = data[1]
        tmp["heat_capacity_b"] = data[2]
        tmp["heat_capacity_c"] = data[3]
        tmp["heat_capacity_d"] = data[4]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["napl_density_ref"] = data[0]
        tmp["napl_temperature_ref"] = data[1]
        tmp["gas_diffusivity_ref"] = data[2]
        tmp["gas_temperature_ref"] = data[3]
        tmp["exponent"] = data[4]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["napl_viscosity_a"] = data[0]
        tmp["napl_viscosity_b"] = data[1]
        tmp["napl_viscosity_c"] = data[2]
        tmp["napl_viscosity_d"] = data[3]
        tmp["volume_crit"] = data[4]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["solubility_a"] = data[0]
        tmp["solubility_b"] = data[1]
        tmp["solubility_c"] = data[2]
        tmp["solubility_d"] = data[3]

        line = f.next()
        data = read_record(line, fmt[3])
        tmp["oc_coeff"] = data[0]
        tmp["oc_fraction"] = data[1]
        tmp["oc_decay"] = data[2]

        chemp["chemical_properties"][chem] = tmp

    return chemp


def _read_ncgas(f):
    """Read NCGAS block data."""
    fmt = block_to_format["NCGAS"]
    ncgas = {"non_condensible_gas": []}

    # Record 1
    line = f.next()
    n = read_record(line, fmt[1])[0]

    # Record 2
    for _ in range(n):
        line = f.next()
        data = read_record(line, fmt[2])
        ncgas["non_condensible_gas"].append(data[0])

    return ncgas


def _read_multi(f):
    """Read MULTI block data."""
    fmt = block_to_format["MULTI"]
    multi = {}

    line = f.next()
    data = read_record(line, fmt)
    multi["n_component"] = data[0]
    multi["isothermal"] = data[1] == data[0]
    multi["n_phase"] = data[2]
    multi["do_diffusion"] = data[3] == 8
    multi["n_component_incon"] = data[4]

    return multi


def _read_solvr(f):
    """Read SOLVR block data."""
    fmt = block_to_format["SOLVR"]
    solvr = {}

    line = f.next()
    data = read_record(line, fmt)
    solvr["solver"] = {
        "method": data[0],
        "z_precond": data[2],
        "o_precond": data[4],
        "rel_iter_max": data[5],
        "eps": data[6],
    }

    return solvr


def _read_param(f, n_variables, eos=None):
    """Read PARAM block data."""
    fmt = block_to_format["PARAM"]
    param = {}

    # Record 1
    line = f.next()
    data = read_record(line, fmt[1])
    param["options"] = {
        "n_iteration": data[0],
        "verbosity": data[1],
        "n_cycle": data[2],
        "n_second": data[3],
        "n_cycle_print": data[4],
        "temperature_dependence_gas": data[7],
        "effective_strength_vapor": data[8],
    }
    param["extra_options"] = {
        i + 1: int(x) for i, x in enumerate(data[5]) if x.isdigit()
    }

    # Record 2
    line = f.next()
    data = read_record(line, fmt[2])
    param["options"].update(
        {
            "t_ini": data[0],
            "t_max": data[1],
            "t_steps": data[2],
            "t_step_max": data[3],
            "gravity": data[6],
            "t_reduce_factor": data[7],
            "mesh_scale_factor": data[8],
        }
    )
    wdata = data[4]

    t_steps = data[2]
    if t_steps:
        if t_steps >= 0.0:
            param["options"]["t_steps"] = t_steps

        else:
            t_steps = int(-t_steps)
            param["options"]["t_steps"] = []

            for _ in range(t_steps):
                line = f.next()
                data = read_record(line, fmt[3])
                param["options"]["t_steps"] += prune_values(data)

            if len(param["options"]["t_steps"]) == 1:
                param["options"]["t_steps"] = param["options"]["t_steps"][0]

    # TOUGHREACT
    if wdata == "wdata":
        line = f.next()
        n = int(line.strip().split()[0])

        if n:
            param["options"]["react_wdata"] = []
            for _ in range(n):
                line = f.next()
                param["options"]["react_wdata"].append(line.strip()[:5])

    # Record 3
    line = f.next()
    data = read_record(line, fmt[4])
    param["options"].update(
        {
            "eps1": data[0],
            "eps2": data[1],
            "w_upstream": data[3],
            "w_newton": data[4],
            "derivative_factor": data[5],
        }
    )

    # Record 4 (ECO2M, TMVOC)
    if eos in {"eco2m", "tmvoc"}:
        line = f.next()
        data = read_record(line, "5d")
        param["default"] = {"phase_composition": data[0]}

    # Record 5
    data = read_primary_variables(f, fmt[5], n_variables)

    if "default" not in param:
        param["default"] = {}

    if any(x is not None for x in data):
        data = prune_values(data)
        param["default"]["initial_condition"] = data

    if not n_variables:
        n_variables = len(data)

    # Remove Nones
    param["options"] = prune_values(param["options"])
    param["extra_options"] = prune_values(param["extra_options"])

    return param, n_variables


def _read_selec(f):
    """Read SELEC block data."""
    fmt = block_to_format["SELEC"]
    selec = {"selections": {}}

    line = f.next()
    data = read_record(line, fmt[1])
    selec["selections"]["integers"] = {k + 1: v for k, v in enumerate(data)}

    if selec["selections"]["integers"][1]:
        selec["selections"]["floats"] = []
        for _ in range(selec["selections"]["integers"][1]):
            line = f.next()
            data = read_record(line, fmt[2])
            selec["selections"]["floats"].append(prune_values(data))

    selec["selections"]["integers"] = prune_values(selec["selections"]["integers"])
    if selec["selections"]["integers"][1] == 1:
        selec["selections"]["floats"] = selec["selections"]["floats"][0]

    return selec


def _read_indom(f, n_variables, eos=None):
    """Read INDOM block data."""
    fmt = block_to_format["INDOM"]
    indom = {"rocks": {}}

    line = f.next()
    while True:
        if line.strip():
            # Record 1
            data = read_record(line, fmt[5])
            rock = data[0]
            phase_composition = data[1]  # ECO2M, TMVOC

            # Record 2
            data = read_primary_variables(f, fmt[0], n_variables)
            data = prune_values(data)
            indom["rocks"][rock] = {"initial_condition": data}

            if not n_variables:
                n_variables = len(data)

            if eos in {"eco2m", "tmvoc"}:
                indom["rocks"][rock]["phase_composition"] = phase_composition

        else:
            break

        line = f.next()

    return indom, n_variables


def _read_momop(f):
    """Read MOMOP block data."""
    fmt = block_to_format["MOMOP"]

    line = f.next()
    data = read_record(line, fmt)
    momop = {
        "more_options": {i + 1: int(x) for i, x in enumerate(data[0]) if x.isdigit()}
    }

    return momop


def _read_times(f):
    """Read TIMES block data."""
    fmt = block_to_format["TIMES"]
    times = {"times": []}

    # Record 1
    line = f.next()
    data = read_record(line, fmt[1])
    n_times = data[0]

    # Record 2
    while len(times["times"]) < n_times:
        line = f.next()
        data = read_record(line, fmt[2])
        times["times"] += prune_values(data)

    return times


def _read_hyste(f):
    """Read HYSTE block data."""
    fmt = block_to_format["HYSTE"]
    hyste = {"hysteresis_options": {}}

    line = f.next()
    data = read_record(line, fmt)
    hyste["hysteresis_options"] = {k + 1: v for k, v in enumerate(data)}
    hyste["hysteresis_options"] = prune_values(hyste["hysteresis_options"])

    return hyste


def _read_oft(f, oft, label_length):
    """Read FOFT, COFT and GOFT blocks data."""
    key = oft_to_key[oft]
    fmt = block_to_format[oft]
    history = {key: []}

    line = f.next()
    if not label_length:
        label_length = get_label_length(line[:9])
    label_format = f"{{:>{label_length if oft != 'COFT' else 2 * label_length}}}"
    iflag = 2 if oft != "GOFT" else 3

    while True:
        if line.strip():
            data = read_record(line, fmt[label_length])
            tmp = {"label": label_format.format(data[0])}

            if data[iflag] is not None:
                tmp["flag"] = data[iflag]

            history[key].append(tmp)

        else:
            break

        line = f.next()

    return history, label_length


def _read_roft(f):
    """Read ROFT block data."""
    fmt = block_to_format["ROFT"]
    history = {"rock_history": []}

    line = f.next()
    while True:
        if line.strip():
            data = read_record(line, fmt)
            rock1 = data[0] if data[0] else ""
            rock2 = data[1] if data[1] else ""
            history["rock_history"].append([rock1, rock2])

        else:
            break

        line = f.next()

    return history


def _read_gener(f, label_length, simulator="tough"):
    """Read GENER block data."""

    def read_table(f, n, fmt):
        table = []
        while len(table) < n:
            line = f.next()
            data = read_record(line, fmt)
            table += prune_values(data)

        return table

    fmt = block_to_format["GENER"]
    gener = {"generators": []}

    line = f.next()
    if not label_length:
        label_length = get_label_length(line[:9])
    label_format = f"{{:>{label_length}}}"

    flag = False
    while True:
        if line.strip() and not line.startswith("+++"):
            data = read_record(line, fmt[label_length])
            tmp = {
                "label": label_format.format(data[0]),
                "name": data[1],
                "nseq": data[2],
                "nadd": data[3],
                "nads": data[4],
                "type": data[7],
                "layer_thickness": data[11],
            }
            ktab = data[12]  # TOUGHREACT

            ltab = data[5]
            if ltab and ltab > 1 and tmp["type"] != "DELV":
                itab = data[8]
                keys = ["times", "rates"]
                keys += (
                    ["specific_enthalpy"] if itab else []
                )  # Specific enthalpy must be provided for time dependent injection
                for key in keys:
                    tmp[key] = read_table(f, ltab, fmt[0])

            else:
                tmp.update(
                    {
                        "times": None,
                        "rates": data[9],
                        "specific_enthalpy": data[10],
                    }
                )

            if ltab and tmp["type"] == "DELV":
                tmp["n_layer"] = ltab

            if simulator == "toughreact" and ktab:
                tmp["conductivity_times"] = read_table(f, ktab, fmt[0])
                tmp["conductivity_factors"] = read_table(f, ktab, fmt[0])

            gener["generators"].append(tmp)

        else:
            flag = line.startswith("+++")
            break

        line = f.next()

    return (
        {"generators": [prune_values(generator) for generator in gener["generators"]]},
        flag,
        label_length,
    )


def _read_timbc(f):
    """Read TIMBC block data."""
    timbc = {"boundary_conditions": []}

    # Record 1
    line = f.next().strip()
    ntptab = int(line)

    for _ in range(ntptab):
        # Record 2
        line = f.next().strip()
        data = [int(x) for x in line.split()]
        if len(data) < 2:
            raise ReadError()

        nbcp = data[0]
        nbcpv = data[1]

        # Record 3
        bcelm = f.next().strip()

        # Record 4
        times = []
        values = []
        for _ in range(nbcp):
            line = f.next().strip()
            data = [float(x) for x in line.split()]
            times.append(data[0])
            values.append(data[1])

        tmp = {
            "label": bcelm,
            "variable": nbcpv,
            "times": times,
            "values": values,
        }
        timbc["boundary_conditions"].append(tmp)

    return timbc


def _read_diffu(f):
    """Read DIFFU block data."""
    fmt = block_to_format["DIFFU"]
    diffu = {"diffusion": []}

    while True:
        i = f.tell()
        line = f.next()

        if line.split():
            try:
                data = read_record(line, fmt)
                diffu["diffusion"].append(prune_values(data))
            except ValueError:
                f.seek(i, increment=-1)
                break
        else:
            break

    return diffu


def _read_outpt(f):
    """Read OUTPT block data."""
    outpt = {"react": {"output": {}}}

    line = f.next().strip()
    data = [int(x) for x in line.split()]  # Free-format

    if len(data):
        outpt["react"]["output"]["format"] = data[0]

    if len(data) > 1:
        outpt["react"]["output"]["shape"] = data[1:]

    return outpt


def _read_outpu(f):
    """Read OUTPU block data."""
    fmt = block_to_format["OUTPU"]
    outpu = {"output": {}}

    # Format
    line = f.next().strip()
    if line and not line.isdigit():
        outpu["output"]["format"] = line if line else None
        line = f.next().strip()

    # Variables
    if line.isdigit():
        num_vars = int(line)
        outpu["output"]["variables"] = []

        for _ in range(num_vars):
            line = f.next()
            data = read_record(line, fmt[3])
            name = data[0].lower()

            tmp = prune_values(data[1:])
            options = None if len(tmp) == 0 else tmp[0] if len(tmp) == 1 else tmp

            outpu["output"]["variables"].append({"name": name})
            if options is not None:
                outpu["output"]["variables"][-1]["options"] = options

    return outpu


def _read_eleme(f, label_length):
    """Read ELEME block data."""
    fmt = block_to_format["ELEME"]
    eleme = {"elements": {}}

    line = f.next()
    if not label_length:
        label_length = get_label_length(line[:9])
    label_format = f"{{:>{label_length}}}"

    while True:
        if line.strip():
            data = read_record(line, fmt[label_length])
            label = label_format.format(data[0])
            label = label.lstrip() if label.lstrip().isalpha() else label
            rock = data[3]
            if rock:
                rock = rock.strip()
                rock = int(rock) if rock.isdigit() else rock
            eleme["elements"][label] = {
                "nseq": data[1],
                "nadd": data[2],
                "material": rock,
                "volume": data[4],
                "heat_exchange_area": data[5],
                "permeability_modifier": data[6],
                "center": data[7:10],
            }

        else:
            break

        line = f.next()

    eleme["elements"] = {k: prune_values(v) for k, v in eleme["elements"].items()}

    return eleme, label_length


def _read_coord(f):
    """Read COORD block data."""
    fmt = block_to_format["COORD"]
    coord = []

    line = f.next()
    while True:
        if line.strip():
            data = read_record(line, fmt)
            coord.append(data)
        else:
            break

        line = f.next()

    return coord


def _read_conne(f, label_length):
    """Read CONNE block data."""
    fmt = block_to_format["CONNE"]
    conne = {"connections": {}}

    line = f.next()
    if not label_length:
        label_length = get_label_length(line[:9])
    label_format = f"{{:>{2 * label_length}}}"

    flag = False
    while True:
        if line.strip() and not line.startswith("+++"):
            data = read_record(line, fmt[label_length])
            label = label_format.format(data[0])
            conne["connections"][label] = {
                "nseq": data[1],
                "nadd": data[2:4],
                "permeability_direction": data[4],
                "nodal_distances": data[5:7],
                "interface_area": data[7],
                "gravity_cosine_angle": data[8],
                "radiant_emittance_factor": data[9],
            }

        else:
            flag = line.startswith("+++")
            break

        line = f.next()

    conne["connections"] = {k: prune_values(v) for k, v in conne["connections"].items()}

    return conne, flag, label_length


def _read_incon(f, label_length, n_variables, eos=None, simulator="tough"):
    """Read INCON block data."""
    fmt = block_to_format["INCON"]
    fmt2 = (
        fmt[simulator]
        if simulator == "toughreact"
        else fmt[eos] if eos in fmt else fmt["default"]
    )
    incon = {"initial_conditions": {}}

    line = f.next()
    if not label_length:
        label_length = get_label_length(line[:9])
    label_format = f"{{:>{label_length}}}"

    flag = False
    while True:
        if line.strip() and not line.startswith("+++"):
            # Record 1
            data = read_record(line, fmt2[label_length])
            label = label_format.format(data[0])
            incon["initial_conditions"][label] = {"porosity": data[3]}

            if simulator == "toughreact":
                permeability = data[4] if len(set(data[4:7])) == 1 else data[4:7]
                incon["initial_conditions"][label]["permeability"] = (
                    permeability if permeability else None
                )

            elif eos in {"eco2m", "tmvoc"}:
                incon["initial_conditions"][label]["phase_composition"] = data[4]

            else:
                userx = prune_values(data[4:])
                incon["initial_conditions"][label]["userx"] = userx if userx else None

            # Record 2
            data = read_primary_variables(f, fmt[0], n_variables)
            data = prune_values(data)
            incon["initial_conditions"][label]["values"] = data

            if not n_variables:
                n_variables = len(data)

        else:
            flag = line.startswith("+++")
            break

        line = f.next()

    incon["initial_conditions"] = {
        k: prune_values(v) for k, v in incon["initial_conditions"].items()
    }

    return incon, flag, label_length, n_variables


def _read_meshm(f):
    """Read MESHM block data."""

    def read_minc(f, data, fmt):
        """Read MINC data."""
        minc = {}

        # Record 1
        line = f.next()
        data = read_record(line, fmt[1])
        minc["type"] = data[1].lower()
        minc["dual"] = data[3].lower()

        # Record 2
        line = f.next()
        data = read_record(line, fmt[2])
        minc["n_minc"] = data[0]
        n_volume = data[1]
        minc["where"] = data[2].lower()
        minc["parameters"] = prune_values(data[3:])

        # Record 3
        minc["volumes"] = []

        while len(minc["volumes"]) < n_volume:
            line = f.next()
            data = read_record(line, fmt[3])
            minc["volumes"] += prune_values(data)

        return minc

    fmt = block_to_format["MESHM"]
    fmt_minc = fmt["MINC"]

    # Mesh type
    line = f.next().strip()
    data = read_record(line, fmt[1])
    mesh_type = data[0].upper()

    if mesh_type in {"XYZ", "RZ2D", "RZ2DL"}:
        meshm = {"meshmaker": {"type": mesh_type.lower()}}

    else:
        meshm = {"minc": {}}

    # XYZ
    if mesh_type == "XYZ":
        fmt = fmt["XYZ"]

        # Record 1
        line = f.next().strip()
        data = read_record(line, fmt[1])
        meshm["meshmaker"]["angle"] = data[0]

        # Record 2
        meshm["meshmaker"]["parameters"] = []

        while True:
            line = f.next()

            if line.strip():
                data = read_record(line, fmt[2])
                tmp = {
                    "type": data[0].lower(),
                    "n_increment": data[1],
                }

                if data[2]:
                    tmp["sizes"] = data[2]

                else:
                    sizes = []
                    while len(sizes) < tmp["n_increment"]:
                        line = f.next()
                        data = read_record(line, fmt[3])
                        sizes += prune_values(data)

                    tmp["sizes"] = sizes[: tmp["n_increment"]]

                meshm["meshmaker"]["parameters"].append(tmp)

            else:
                break

    # RZ2D
    elif mesh_type in {"RZ2D", "RZ2DL"}:
        fmt = fmt["RZ2D"]

        # Record 1
        meshm["meshmaker"]["parameters"] = []

        while True:
            line = f.next()

            if line.strip():
                data = read_record(line, fmt[1])
                data_type = data[0].upper()

                if data_type == "RADII":
                    line = f.next()
                    data = read_record(line, fmt["RADII"][1])
                    n = data[0]

                    radii = []
                    while len(radii) < n:
                        line = f.next()
                        data = read_record(line, fmt["RADII"][2])
                        radii += prune_values(data)

                    tmp = {
                        "type": data_type.lower(),
                        "radii": radii[:n],
                    }
                    meshm["meshmaker"]["parameters"].append(tmp)

                elif data_type == "EQUID":
                    line = f.next()
                    data = read_record(line, fmt["EQUID"])

                    tmp = {
                        "type": data_type.lower(),
                        "n_increment": data[0],
                        "size": data[2],
                    }
                    meshm["meshmaker"]["parameters"].append(tmp)

                elif data_type == "LOGAR":
                    line = f.next()
                    data = read_record(line, fmt["LOGAR"])

                    tmp = {
                        "type": data_type.lower(),
                        "n_increment": data[0],
                        "radius": data[2],
                        "radius_ref": data[3],
                    }
                    meshm["meshmaker"]["parameters"].append(tmp)

                elif data_type == "LAYER":
                    # Record 1
                    line = f.next()
                    data = read_record(line, fmt["LAYER"][1])
                    n = data[0]

                    # Record 2
                    thicknesses = []
                    while len(thicknesses) < n:
                        line = f.next()
                        data = read_record(line, fmt["LAYER"][2])
                        thicknesses += prune_values(data)

                    tmp = {"type": data_type.lower(), "thicknesses": thicknesses[:n]}
                    meshm["meshmaker"]["parameters"].append(tmp)

                    # LAYER closes the block RZ2D
                    # RZ2D can be followed by MINC
                    line = f.next()

                    if line.startswith("MINC"):
                        meshm["minc"] = read_minc(f, data, fmt_minc)

                    else:
                        break

            else:
                break

    # MINC
    elif mesh_type == "MINC":
        meshm["minc"] = read_minc(f, data, fmt_minc)

    return meshm


def _read_poise(f):
    """Read POISE block data."""
    poise = {"react": {"poiseuille": {}}}

    line = f.next().strip()
    data = [float(x) for x in line.split()]  # Free-format

    if len(data) < 5:
        raise ReadError()

    poise["react"]["poiseuille"]["start"] = data[:2]
    poise["react"]["poiseuille"]["end"] = data[2:4]
    poise["react"]["poiseuille"]["aperture"] = data[4]

    return poise
