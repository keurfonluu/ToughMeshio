import logging
from copy import deepcopy

import numpy as np

from ...._common import block_to_format, open_file, prune_values, str2format
from ..._common import write_record
from .._common import write_ffrecord
from ._common import default
from ._helpers import block, write_model_record

__all__ = [
    "write",
]


def write(
    filename,
    parameters,
    block=None,
    ignore_blocks=None,
    eos=None,
    space_between_blocks=False,
    space_between_values=True,
    simulator="tough",
):
    """
    Write TOUGH input file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Output file name or buffer.
    parameters : dict
        Parameters to export.
    block : str {'all', 'gener', 'mesh', 'incon'} or None, optional, default None
        Blocks to be written:

         - 'all': write all blocks,
         - 'gener': only write block GENER,
         - 'mesh': only write blocks ELEME, COORD and CONNE,
         - 'incon': only write block INCON,
         - None: write all blocks except blocks defined in `ignore_blocks`.

    ignore_blocks : list of str or None, optional, default None
        Blocks to ignore. Only if `block` is None.
    space_between_blocks : bool, optional, default False
        Add an empty record between blocks.
    space_between_values : bool, optional, default True
        Add a white space between floating point values.
    eos : str or None, optional, default None
        Equation of State. If `eos` is defined in `parameters`, this option will be ignored.

    """
    if simulator not in {"tough", "toughreact"}:
        raise ValueError()

    buffer = write_buffer(
        parameters,
        block,
        ignore_blocks,
        space_between_blocks,
        space_between_values,
        eos,
        simulator,
    )
    with open_file(filename, "w") as f:
        for record in buffer:
            f.write(record)


def write_buffer(
    params,
    block,
    ignore_blocks=None,
    space_between_blocks=False,
    space_between_values=True,
    eos_=None,
    simulator="tough",
):
    """Write TOUGH input file as a list of 80-character long record strings."""
    from ._common import Parameters
    from ._common import blocks as blocks_
    from ._common import default, eos

    # Block filters
    if block is not None:
        if block == "all":
            blocks = blocks_.copy()

        elif block == "gener":
            blocks = {"GENER", "END COMMENTS"}

        elif block == "mesh":
            blocks = {"ELEME", "COORD", "CONNE", "END COMMENTS"}

        elif block == "incon":
            blocks = {"INCON", "END COMMENTS"}

        else:
            raise ValueError()

    else:
        blocks = blocks_.copy()

        if ignore_blocks is not None:
            if not isinstance(ignore_blocks, (list, tuple, np.ndarray)):
                raise TypeError()

            ignore_blocks = set(ignore_blocks)
            for ignore_block in ignore_blocks:
                if ignore_block not in blocks_:
                    raise ValueError(f"unknown block '{ignore_block}'.")

                else:
                    blocks.remove(ignore_block)

    # Some preprocessing
    parameters = deepcopy(Parameters)
    parameters.update(deepcopy(params))

    parameters["title"] = (
        [parameters["title"]]
        if isinstance(parameters["title"], str)
        else parameters["title"]
    )

    parameters["end_comments"] = (
        None
        if not parameters["end_comments"]
        else (
            [parameters["end_comments"]]
            if isinstance(parameters["end_comments"], str)
            else parameters["end_comments"]
        )
    )

    for k, v in default.items():
        if k not in parameters["default"]:
            parameters["default"][k] = v

    for rock in parameters["rocks"]:
        for k, v in parameters["default"].items():
            cond1 = k not in parameters["rocks"][rock]
            cond2 = k not in {
                "initial_condition",
                "relative_permeability",
                "capillarity",
                "phase_composition",
                "react_tp",
                "react_hcplaw",
            }
            if cond1 and cond2:
                parameters["rocks"][rock][k] = v

    tmp = parameters["default"]["permeability"]
    if np.ndim(tmp) == 0:
        tmp = 3 * [tmp]

    for v in parameters["rocks"].values():
        if v["permeability"] is None:
            v["permeability"] = tmp

        elif np.ndim(v["permeability"]):
            if len(v["permeability"]) != 3:
                raise ValueError()

            for i, k in enumerate(tmp):
                if v["permeability"][i] is None:
                    v["permeability"][i] = k

    # Make sure that some keys are integers and not strings
    if "more_options" in parameters:
        parameters["more_options"] = {
            int(k): v for k, v in parameters["more_options"].items()
        }

    if "extra_options" in parameters:
        parameters["extra_options"] = {
            int(k): v for k, v in parameters["extra_options"].items()
        }

    if "selections" in parameters and "integers" in parameters["selections"]:
        parameters["selections"]["integers"] = {
            int(k): v for k, v in parameters["selections"]["integers"].items()
        }

    if "react" in parameters and "options" in parameters["react"]:
        parameters["react"]["options"] = {
            int(k): v for k, v in parameters["react"]["options"].items()
        }

    # Check that EOS is defined (for block MULTI)
    if parameters["eos"] and parameters["eos"] not in eos:
        raise ValueError(f"EOS '{parameters['eos']}' is unknown or not supported.")

    if parameters["eos"]:
        eos_ = parameters["eos"]

    else:
        parameters["eos"] = eos_

    # Set some flags
    cond1 = (
        "relative_permeability" in parameters["default"]
        and parameters["default"]["relative_permeability"]["id"] is not None
    )
    cond2 = (
        "capillarity" in parameters["default"]
        and parameters["default"]["capillarity"]["id"] is not None
    )
    rpcap = cond1 or cond2

    param = False
    if prune_values(parameters["default"]["initial_condition"]):
        param = True
    if (
        eos_ in {"eco2m", "tmvoc"}
        and parameters["default"]["phase_composition"] is not None
    ):
        param = True

    for option in parameters["options"].values():
        if param:
            break

        if (isinstance(option, (list, tuple, np.ndarray)) and len(option)) or option:
            param = True

    if parameters["extra_options"]:
        param = True

    indom = False
    for rock in parameters["rocks"].values():
        if "initial_condition" in rock:
            if any(x is not None for x in rock["initial_condition"][:4]):
                indom = True
                break

    multi = (
        parameters["eos"]
        or parameters["n_component"]
        or parameters["n_phase"]
        or parameters["n_component_incon"]
    )

    output = False
    for key in ["format", "variables"]:
        if key in parameters["output"] and parameters["output"][key]:
            output = True
            break

    # TOUGHREACT related flags
    react = "options" in parameters["react"] and parameters["react"]["options"]
    outpt = (
        "output" in parameters["react"]
        and "format" in parameters["react"]["output"]
        and parameters["react"]["output"]["format"] is not None
    )
    poise = "poiseuille" in parameters["react"] and parameters["react"]["poiseuille"]

    # Check that start is True if indom is True
    if indom and not parameters["start"]:
        logging.warning("Option 'START' is needed to use 'INDOM' conditions.")

    # Check diffusion
    if parameters["do_diffusion"] is None:
        parameters["do_diffusion"] = len(parameters["diffusion"]) > 0

    if parameters["do_diffusion"] and not len(parameters["diffusion"]):
        logging.warning("Diffusion is enabled but 'diffusion' data is missing.")

    # Define input file contents
    out = []
    if "TITLE" in blocks:
        out += [f"{title:80}\n" for title in parameters["title"]]
        out += ["\n"] if space_between_blocks else []

    if "DIMEN" in blocks and parameters["array_dimensions"]:
        out += _write_dimen(parameters, space_between_values)

    if "ROCKS" in blocks and parameters["rocks"]:
        out += _write_rocks(parameters, space_between_values, simulator)

    if "RPCAP" in blocks and rpcap:
        out += _write_rpcap(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "REACT" in blocks and react and simulator == "toughreact":
        out += _write_react(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "FLAC" in blocks and parameters["flac"]:
        out += _write_flac(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "CHEMP" in blocks and parameters["chemical_properties"]:
        out += _write_chemp(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "NCGAS" in blocks and len(parameters["non_condensible_gas"]):
        out += _write_ncgas(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "MULTI" in blocks and multi:
        out += _write_multi(parameters)
        out += ["\n"] if space_between_blocks else []

    if "SOLVR" in blocks and parameters["solver"]:
        out += _write_solvr(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "INDEX" in blocks and parameters["index"]:
        out += _write_index()
        out += ["\n"] if space_between_blocks else []

    if "START" in blocks and parameters["start"]:
        out += _write_start()
        out += ["\n"] if space_between_blocks else []

    if "PARAM" in blocks and param:
        out += _write_param(parameters, space_between_values, eos_, simulator)
        out += ["\n"] if space_between_blocks else []

    if "SELEC" in blocks and parameters["selections"]:
        out += _write_selec(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "INDOM" in blocks and indom:
        out += _write_indom(parameters, space_between_values, eos_)

    if "MOMOP" in blocks and parameters["more_options"]:
        out += _write_momop(parameters)
        out += ["\n"] if space_between_blocks else []

    if "TIMES" in blocks and len(parameters["times"]):
        out += _write_times(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "HYSTE" in blocks and parameters["hysteresis_options"]:
        out += _write_hyste(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "FOFT" in blocks and len(parameters["element_history"]):
        out += _write_foft(parameters, space_between_values)

    if "COFT" in blocks and len(parameters["connection_history"]):
        out += _write_coft(parameters, space_between_values)

    if "GOFT" in blocks and len(parameters["generator_history"]):
        out += _write_goft(parameters, space_between_values)

    if "ROFT" in blocks and len(parameters["rock_history"]):
        out += _write_roft(parameters)

    if "GENER" in blocks and parameters["generators"]:
        out += _write_gener(parameters, space_between_values, simulator)

    if "TIMBC" in blocks and parameters["boundary_conditions"]:
        out += _write_timbc(parameters)

    if "DIFFU" in blocks and len(parameters["diffusion"]):
        out += _write_diffu(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "OUTPU" in blocks and output:
        out += _write_outpu(parameters, space_between_values)
        out += ["\n"] if space_between_blocks else []

    if "OUTPT" in blocks and outpt and simulator == "toughreact":
        out += _write_outpt(parameters)
        out += ["\n"] if space_between_blocks else []

    if "ELEME" in blocks and parameters["elements"]:
        out += _write_eleme(parameters, space_between_values)

    if "COORD" in blocks and parameters["coordinates"]:
        out += _write_coord(parameters, space_between_values)

    if "CONNE" in blocks and (parameters["connections"] or parameters["elements"]):
        out += _write_conne(parameters, space_between_values)

    if "INCON" in blocks and parameters["initial_conditions"]:
        out += _write_incon(parameters, space_between_values, eos_, simulator)

    if "MESHM" in blocks and (parameters["meshmaker"] or parameters["minc"]):
        out += _write_meshm(parameters, space_between_values)

    if "POISE" in blocks and poise and simulator == "toughreact":
        out += _write_poise(parameters)
        out += ["\n"] if space_between_blocks else []

    if "NOVER" in blocks and parameters["nover"]:
        out += _write_nover()
        out += ["\n"] if space_between_blocks else []

    if "ENDCY" in blocks:
        out += _write_endcy()

    if "END COMMENTS" in blocks and parameters["end_comments"]:
        if block in {"gener", "mesh", "incon"}:
            while out[-1] == "\n":
                out = out[:-1]

        else:
            out += ["\n"] if space_between_blocks else []

        out += [f"{comment}\n" for comment in parameters["end_comments"]]

    return out


@block("DIMEN")
def _write_dimen(parameters, space_between_values):
    """Write DIMEN block data."""
    # Format
    fmt = block_to_format["DIMEN"]
    fmt1 = str2format(fmt)

    data = parameters["array_dimensions"]
    values = [
        data["n_rocks"],
        data["n_times"],
        data["n_generators"],
        data["n_rates"],
        data["n_increment_x"],
        data["n_increment_y"],
        data["n_increment_z"],
        data["n_increment_rad"],
        data["n_properties"],
        data["n_properties_times"],
        data["n_regions"],
        data["n_regions_parameters"],
        data["n_ltab"],
        data["n_rpcap"],
        data["n_elements_timbc"],
        data["n_timbc"],
    ]
    out = write_record(values, fmt1, space_between_values, multi=True)

    return out


@block("ROCKS", multi=True)
def _write_rocks(parameters, space_between_values, simulator="tough"):
    """Write ROCKS block data."""
    # Formats
    fmt = block_to_format["ROCKS"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])
    fmt3 = str2format(fmt[3])
    fmt4 = str2format(fmt[4])

    fmt = block_to_format["RPCAP"]
    fmt5 = str2format(fmt)

    out = []
    for k, v in parameters["rocks"].items():
        # Number of additional lines to write per rock
        cond = any(
            v[k] is not None
            for k in [
                "compressibility",
                "expansivity",
                "conductivity_dry",
                "tortuosity",
                "klinkenberg_parameter",
                "distribution_coefficient_3",
                "distribution_coefficient_4",
                "tortuosity_exponent",
                "porosity_crit",
            ]
        )
        nad = 2 if "relative_permeability" in v or "capillarity" in v else int(cond)

        if simulator == "toughreact":
            nad = 4 if "react_tp" in v else nad
            nad = 5 if "react_hcplaw" in v else nad

        # Permeability
        per = v["permeability"]
        per = [per] * 3 if not np.ndim(per) else per
        if not (isinstance(per, (list, tuple, np.ndarray)) and len(per) == 3):
            raise TypeError()

        # Record 1
        values = [
            k,
            nad if nad else "",
            v["density"],
            v["porosity"],
            per[0],
            per[1],
            per[2],
            v["conductivity"],
            v["specific_heat"],
        ]
        out += write_record(values, fmt1, space_between_values)

        # Record 2
        if cond:
            values = [
                v["compressibility"],
                v["expansivity"],
                v["conductivity_dry"],
                v["tortuosity"],
                v["klinkenberg_parameter"],
                v["distribution_coefficient_3"],
                v["distribution_coefficient_4"],
                v["tortuosity_exponent"],
                v["porosity_crit"],
            ]
            out += write_record(values, fmt2, space_between_values)

        else:
            out += write_record([], [], space_between_values) if nad >= 2 else []

        # TOUGHREACT
        if nad >= 4:
            out += write_model_record(v, "react_tp", fmt3, space_between_values)
            out += write_model_record(v, "react_hcplaw", fmt4, space_between_values)

        # Relative permeability / Capillary pressure
        if nad >= 2:
            out += write_model_record(
                v, "relative_permeability", fmt5, space_between_values
            )
            out += write_model_record(v, "capillarity", fmt5, space_between_values)

    return out


@block("RPCAP")
def _write_rpcap(parameters, space_between_values):
    """Write RPCAP block data."""
    data = deepcopy(default)
    data.update(parameters["default"])

    # Formats
    fmt = block_to_format["RPCAP"]
    fmt = str2format(fmt)

    out = []
    for key in ["relative_permeability", "capillarity"]:
        if key in data:
            values = [data[key]["id"], None]
            values += list(data[key]["parameters"])
            out += write_record(values, fmt, space_between_values)

        else:
            out += write_record([], [], space_between_values)

    return out


@block("REACT")
def _write_react(parameters, space_between_values):
    """Write REACT block data."""
    from ._common import react_options

    # Formats
    fmt = block_to_format["REACT"]
    fmt = str2format(fmt)

    _react = deepcopy(react_options)
    _react.update(parameters["react"]["options"])

    tmp = [" " if _react[k] is None else str(_react[k]) for k in sorted(_react)]
    out = write_record(["".join(tmp)], fmt, space_between_values)

    return out


@block("FLAC", multi=True)
def _write_flac(parameters, space_between_values):
    """Write FLAC block data."""
    # Load data
    from ._common import flac

    data = deepcopy(flac)
    data.update(parameters["flac"])

    # Formats
    fmt = block_to_format["FLAC"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])
    fmt3 = str2format(fmt[3])

    # Record 1
    values = [
        bool(data["creep"]),
        data["porosity_model"],
        data["version"],
    ]
    out = write_record(values, fmt1, space_between_values)

    # Additional records
    for v in parameters["rocks"].values():
        # Load data
        data = deepcopy(default)
        data.update(parameters["default"])
        data.update(v)

        # Permeability model
        values = [data["permeability_model"]["id"]]
        values += list(data["permeability_model"]["parameters"])
        out += write_record(values, fmt2, space_between_values)

        # Equivalent pore pressure
        values = [data["equivalent_pore_pressure"]["id"], None]
        values += list(data["equivalent_pore_pressure"]["parameters"])
        out += write_record(values, fmt3, space_between_values)

    return out


@block("CHEMP")
def _write_chemp(parameters, space_between_values):
    """Write CHEMP block data."""
    # Load data
    from ._common import chemical_properties

    data = parameters["chemical_properties"]

    # Formats
    fmt = block_to_format["CHEMP"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])
    fmt3 = str2format(fmt[3])

    # Record 1
    out = write_record([len(data)], fmt1, space_between_values)

    # Record 2
    for k, v in data.items():
        vv = deepcopy(chemical_properties)
        vv.update(v)

        out += write_record([k], fmt2, space_between_values)

        values = [
            vv["temperature_crit"],
            vv["pressure_crit"],
            vv["compressibility_crit"],
            vv["pitzer_factor"],
            vv["dipole_moment"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["boiling_point"],
            vv["vapor_pressure_a"],
            vv["vapor_pressure_b"],
            vv["vapor_pressure_c"],
            vv["vapor_pressure_d"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["molecular_weight"],
            vv["heat_capacity_a"],
            vv["heat_capacity_b"],
            vv["heat_capacity_c"],
            vv["heat_capacity_d"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["napl_density_ref"],
            vv["napl_temperature_ref"],
            vv["gas_diffusivity_ref"],
            vv["gas_temperature_ref"],
            vv["exponent"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["napl_viscosity_a"],
            vv["napl_viscosity_b"],
            vv["napl_viscosity_c"],
            vv["napl_viscosity_d"],
            vv["volume_crit"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["solubility_a"],
            vv["solubility_b"],
            vv["solubility_c"],
            vv["solubility_d"],
        ]
        out += write_record(values, fmt3, space_between_values)

        values = [
            vv["oc_coeff"],
            vv["oc_fraction"],
            vv["oc_decay"],
        ]
        out += write_record(values, fmt3, space_between_values)

    return out


@block("NCGAS")
def _write_ncgas(parameters, space_between_values):
    """Write NCGAS block data."""
    data = parameters["non_condensible_gas"]

    # Formats
    fmt = block_to_format["NCGAS"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])

    # Record 1
    out = write_record([len(data)], fmt1, space_between_values)

    # Record 2
    out += write_record(data, fmt2, space_between_values, multi=True)

    return out


@block("MULTI")
def _write_multi(parameters):
    """Write MULTI block data."""
    from ._common import eos

    if "eos" in parameters and parameters["eos"] == "tmvoc":
        for keyword in {"n_component", "n_phase"}:
            if parameters[keyword] is None:
                raise ValueError(
                    "for 'tmvoc', at least 'n_component' and 'n_phase' must be specified"
                )

    # Formats
    fmt = block_to_format["MULTI"]
    fmt = str2format(fmt)

    values = (
        list(eos[parameters["eos"]])
        if parameters["eos"] and parameters["eos"] != "tmvoc"
        else [0, 0, 0, 6]
    )
    values[0] = parameters["n_component"] if parameters["n_component"] else values[0]
    values[1] = values[0] if parameters["isothermal"] else values[0] + 1
    values[2] = parameters["n_phase"] if parameters["n_phase"] else values[2]

    # Handle diffusion
    if parameters["do_diffusion"]:
        values[3] = 8
        parameters["n_phase"] = values[2]  # Save for later check

    # Number of mass components
    if parameters["n_component_incon"]:
        values.append(parameters["n_component_incon"])

    out = write_record(values, fmt)

    return out


@block("SOLVR")
def _write_solvr(parameters, space_between_values):
    """Write SOLVR block data."""
    from ._common import solver

    data = deepcopy(solver)
    data.update(parameters["solver"])

    # Formats
    fmt = block_to_format["SOLVR"]
    fmt = str2format(fmt)

    values = [
        data["method"],
        None,
        data["z_precond"],
        None,
        data["o_precond"],
        data["rel_iter_max"],
        data["eps"],
    ]
    out = write_record(values, fmt, space_between_values)

    return out


@block("INDEX")
def _write_index():
    """Write INDEX block data."""
    return []


@block("START")
def _write_start():
    """Write START block data."""
    return []


def _write_param(parameters, space_between_values, eos_=None, simulator="tough"):
    """Write PARAM block data."""
    # Load data
    from ._common import extra_options, header, options

    data = deepcopy(options)
    data.update(parameters["options"])

    # Special header
    out = f"PARAM{header}\n"
    out = [out[:11] + "MOP: 123456789*123456789*1234" + out[40:]]

    # Table
    if not isinstance(data["t_steps"], (list, tuple, np.ndarray)):
        data["t_steps"] = [data["t_steps"]]

    # Formats
    fmt = block_to_format["PARAM"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])
    fmt3 = str2format(fmt[3])
    fmt4 = str2format(fmt[4])
    fmt5 = str2format(fmt[5])

    # Record 1
    _mop = deepcopy(extra_options)
    _mop.update(parameters["extra_options"])
    mop = [" " if _mop[k] is None else str(_mop[k]) for k in sorted(_mop)]

    values = [
        data["n_iteration"],
        data["verbosity"],
        data["n_cycle"],
        data["n_second"],
        data["n_cycle_print"],
        f"{''.join(mop)}",
        None,
        data["temperature_dependence_gas"],
        data["effective_strength_vapor"],
    ]
    out += write_record(values, fmt1, space_between_values)

    # Time steps
    t_steps = data["t_steps"]
    ndlt = len(t_steps)
    if ndlt < 2:
        delten = t_steps[0] if ndlt else None

    else:
        delten = -((ndlt - 1) // 8 + 1)

    # Record 2
    values = [
        data["t_ini"],
        data["t_max"],
        delten,
        data["t_step_max"],
        "wdata" if data["react_wdata"] and simulator == "toughreact" else None,
        None,
        data["gravity"],
        data["t_reduce_factor"],
        data["mesh_scale_factor"],
    ]
    out += write_record(values, fmt2, space_between_values)

    # Record 2.1
    if ndlt > 1:
        values = [x for x in t_steps]
        out += write_record(values, fmt3, space_between_values, multi=True)

    # TOUGHREACT
    if data["react_wdata"] and simulator == "toughreact":
        n = len(data["react_wdata"])
        out += [f"{n}\n"]
        out += [f"{x}\n" for x in data["react_wdata"]]

    # Record 3
    values = [
        data["eps1"],
        data["eps2"],
        None,
        data["w_upstream"],
        data["w_newton"],
        data["derivative_factor"],
    ]
    out += write_record(values, fmt4, space_between_values)

    # Record 4 (ECO2M, TMVOC)
    if eos_ in {"eco2m", "tmvoc"}:
        out += write_record(
            [parameters["default"]["phase_composition"]],
            str2format("5d"),
            space_between_values,
        )

    # Record 5
    out += write_record(
        parameters["default"]["initial_condition"],
        fmt5,
        space_between_values,
        multi=True,
    )

    return out


@block("SELEC")
def _write_selec(parameters, space_between_values):
    """Write SELEC block data."""
    # Load data
    from ._common import selections

    data = deepcopy(selections)

    if parameters["selections"]["integers"]:
        data["integers"].update(parameters["selections"]["integers"])

    if "floats" in parameters["selections"] and len(parameters["selections"]["floats"]):
        data["floats"] = parameters["selections"]["floats"]

    # Check floats and overwrite IE(1)
    if data["floats"] is not None and len(data["floats"]):
        if isinstance(data["floats"][0], (list, tuple, np.ndarray)):
            for x in data["floats"]:
                if len(x) > 8:
                    raise ValueError()

            data["integers"][1] = len(data["floats"])
            ndim = 2

        else:
            if len(data["floats"]) > 8:
                raise ValueError()

            data["integers"][1] = 1
            ndim = 1

    else:
        ndim = None

    # Formats
    fmt = block_to_format["SELEC"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])

    # Record 1
    values = [data["integers"][k] for k in sorted(data["integers"])]
    out = write_record(values, fmt1, space_between_values)

    # Record 2
    if ndim == 1:
        out += write_record(data["floats"], fmt2, space_between_values)

    elif ndim == 2:
        for x in data["floats"]:
            out += write_record(x, fmt2, space_between_values)

    else:
        out += write_record([], fmt2, space_between_values)

    return out


@block("INDOM", multi=True)
def _write_indom(parameters, space_between_values, eos_):
    """Write INDOM block data."""
    # Formats
    fmt = block_to_format["INDOM"]
    fmt1 = str2format(fmt[5])
    fmt2 = str2format(fmt[0])

    out = []
    for k, v in parameters["rocks"].items():
        cond1 = "initial_condition" in v
        cond2 = (
            eos_ in {"eco2m", "tmvoc"}
            and "phase_composition" in v
            and v["phase_composition"] is not None
        )

        if cond1 or cond2:
            # Record 1
            values = [k]

            if eos_ in {"eco2m", "tmvoc"}:
                values.append(v["phase_composition"])

            else:
                values.append(None)

            out += write_record(values, fmt1, space_between_values)

            if cond1:
                out += write_record(
                    v["initial_condition"], fmt2, space_between_values, multi=True
                )

            else:
                out += ["\n"]

    return out


@block("MOMOP")
def _write_momop(parameters):
    """Write MOMOP block data."""
    from ._common import more_options

    # Formats
    fmt = block_to_format["MOMOP"]
    fmt = str2format(fmt)

    _momop = deepcopy(more_options)
    _momop.update(parameters["more_options"])

    tmp = [" " if _momop[k] is None else str(_momop[k]) for k in sorted(_momop)]
    out = write_record(["".join(tmp)], fmt)

    return out


@block("TIMES")
def _write_times(parameters, space_between_values):
    """Write TIMES block data."""
    data = parameters["times"]
    data = data if np.ndim(data) else [data]

    # Formats
    fmt = block_to_format["TIMES"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])

    # Record 1
    out = write_record([len(data)], fmt1, space_between_values)

    # Record 2
    out += write_record(data, fmt2, space_between_values, multi=True)

    return out


@block("HYSTE")
def _write_hyste(parameters, space_between_values):
    """Write HYSTE block data."""
    from ._common import hysteresis_options

    # Formats
    fmt = block_to_format["HYSTE"]
    fmt = str2format(fmt)

    _hyste = deepcopy(hysteresis_options)
    _hyste.update(parameters["hysteresis_options"])

    values = [_hyste[k] for k in sorted(_hyste)]
    out = write_record(values, fmt, space_between_values)

    return out


def _write_oft(oft, parameters, space_between_values):
    """Write FOFT, COFT, and GOFT blocks data."""
    from ._common import histories

    keys = {
        "FOFT": "element",
        "COFT": "connection",
        "GOFT": "generator",
    }

    # Formats
    fmt = block_to_format[oft]
    fmt = str2format(fmt[5])

    out = []
    for v in parameters[f"{keys[oft]}_history"]:
        data = deepcopy(histories)

        if isinstance(v, dict):
            data.update(v)

        else:
            data["label"] = v

        values = [data["label"]]
        values += [None] if oft != "GOFT" else [None, None]
        values += [data["flag"]]
        out += write_record(values, fmt, space_between_values)

    return out


@block("FOFT", multi=True)
def _write_foft(parameters, space_between_values):
    """Write FOFT block data."""
    return _write_oft("FOFT", parameters, space_between_values)


@block("COFT", multi=True)
def _write_coft(parameters, space_between_values):
    """Write COFT block data."""
    return _write_oft("COFT", parameters, space_between_values)


@block("GOFT", multi=True)
def _write_goft(parameters, space_between_values):
    """Write GOFT block data."""
    return _write_oft("GOFT", parameters, space_between_values)


@block("ROFT", multi=True)
def _write_roft(parameters):
    """Write ROFT block data."""
    # Format
    fmt = block_to_format["ROFT"]
    fmt = str2format(fmt)

    out = []
    for values in parameters["rock_history"]:
        out += write_record(values, fmt)

    return out


@block("GENER", multi=True)
def _write_gener(parameters, space_between_values, simulator="tough"):
    """Write GENER block data."""
    from ._common import generators

    # Format
    label_length = max(
        [
            len(generator["label"]) if "label" in generator else 0
            for generator in parameters["generators"]
        ]
    )
    label_length = max(label_length, 5)
    fmt = block_to_format["GENER"]
    fmt1 = str2format(fmt[label_length])
    fmt2 = str2format(fmt[0])

    out = []
    for v in parameters["generators"]:
        # Load data
        data = deepcopy(generators)
        data.update(v)

        # Table
        ltab = 1
        if data["times"] is not None and isinstance(
            data["times"], (list, tuple, np.ndarray)
        ):
            ltab = len(data["times"])

            for key in ["rates", "specific_enthalpy"]:
                if data[key] is not None:
                    if ltab == 1 and np.ndim(data[key]) == 1:
                        if len(data[key]) > 1:
                            raise ValueError()

                        data[key] = data[key][0]

                    else:
                        if np.ndim(data[key]) != 1:
                            raise TypeError()

                        if ltab != len(data[key]):
                            raise ValueError()

        elif data["type"] == "DELV" and data["n_layer"]:
            ltab = data["n_layer"]

        else:
            for key in ["rates", "specific_enthalpy"]:
                if key in data and np.ndim(data[key]) > 0:
                    if len(data[key]) > 1:
                        raise ValueError()

                    data[key] = data[key][0]

        itab = (
            "1"
            if isinstance(data["specific_enthalpy"], (list, tuple, np.ndarray))
            else None
        )

        # TOUGHREACT
        ktab = None
        if (
            simulator == "toughreact"
            and data["conductivity_times"] is not None
            and data["conductivity_factors"] is not None
        ):
            ktab = len(data["conductivity_times"])

            if len(data["conductivity_factors"]) != ktab:
                raise ValueError()

            ktab = ktab if ktab else None

        # Record 1
        values = [
            data["label"] if "label" in data else "",
            data["name"],
            data["nseq"],
            data["nadd"],
            data["nads"],
            ltab if ltab > 1 else None,
            None,
            data["type"],
            itab,
            None if ltab > 1 and data["type"] != "DELV" else data["rates"],
            None if ltab > 1 and data["type"] != "DELV" else data["specific_enthalpy"],
            data["layer_thickness"],
            ktab,
        ]
        out += write_record(values, fmt1, space_between_values)

        if ltab > 1 and data["type"] != "DELV":
            # Record 2
            out += write_record(data["times"], fmt2, space_between_values, multi=True)

            # Record 3
            out += write_record(data["rates"], fmt2, space_between_values, multi=True)

            # Record 4
            if data["specific_enthalpy"] is not None:
                if isinstance(data["specific_enthalpy"], (list, tuple, np.ndarray)):
                    specific_enthalpy = data["specific_enthalpy"]

                else:
                    specific_enthalpy = np.full(ltab, data["specific_enthalpy"])

                out += write_record(
                    specific_enthalpy, fmt2, space_between_values, multi=True
                )

        # TOUGHREACT
        if ktab:
            out += write_record(
                data["conductivity_times"], fmt2, space_between_values, multi=True
            )
            out += write_record(
                data["conductivity_factors"], fmt2, space_between_values, multi=True
            )

    return out


@block("TIMBC", multi=True)
def _write_timbc(parameters):
    """Write TIMBC block data."""
    from ._common import boundary_conditions

    out = []

    # Record 1
    ntptab = len(parameters["boundary_conditions"])
    out += write_ffrecord([ntptab], end="\n")

    for v in parameters["boundary_conditions"]:
        # Load data
        data = deepcopy(boundary_conditions)
        data.update(v)

        # Times
        times = data["times"] if data["times"] is not None else []
        values = data["values"] if data["values"] is not None else []
        nbcp = len(times)

        if len(values) < nbcp:
            raise ValueError()

        # Record 2
        out += write_ffrecord([nbcp, data["variable"]], end="\n")

        # Record 3
        out += write_ffrecord([data["label"]], end="\n")

        # Record 4
        for time, value in zip(times, values):
            out += write_ffrecord([time, value], end="\n")

    return out


@block("DIFFU")
def _write_diffu(parameters, space_between_values):
    """Write DIFFU block data."""
    # Format
    fmt = block_to_format["DIFFU"]
    fmt = str2format(fmt)

    out = []
    for mass in parameters["diffusion"]:
        out += write_record(mass, fmt, space_between_values, multi=True)

    return out


@block("OUTPT")
def _write_outpt(parameters):
    """Write OUTPT block data."""
    outpt = parameters["react"]["output"]

    values = [outpt["format"]]
    values += [x for x in outpt["shape"][:3]] if "shape" in outpt else []
    out = [f"{' '.join(str(x) for x in values)}\n"]

    return out


@block("OUTPU")
def _write_outpu(parameters, space_between_values):
    """Write OUTPU block data."""
    # Load data
    from ._common import output

    data = deepcopy(output)
    data.update(parameters["output"])

    # Format
    fmt = block_to_format["OUTPU"]
    fmt1 = str2format(fmt[1])
    fmt2 = str2format(fmt[2])
    fmt3 = str2format(fmt[3])

    out = []

    # Output format
    out += (
        write_record([data["format"].upper()], fmt1, space_between_values)
        if data["format"]
        else []
    )

    # Variables
    if data["variables"]:
        buffer = []
        num_vars = len(data["variables"])

        for variable in data["variables"]:
            values = [variable["name"].upper() if "name" in variable else None]

            if "options" in variable:
                v = variable["options"]

                if np.ndim(v) == 0:
                    values += [v]
                    buffer += write_record(values, fmt3, space_between_values)

                else:
                    if np.ndim(v[0]) == 0:
                        values += list(v)
                        buffer += write_record(values, fmt3, space_between_values)

                    else:
                        for vv in v:
                            values_in = values + list(vv)
                            buffer += write_record(
                                values_in, fmt3, space_between_values
                            )

            else:
                buffer += write_record(values, fmt3, space_between_values)

        out += write_record([str(num_vars)], fmt2, space_between_values)
        out += buffer

    return out


@block("ELEME", multi=True)
def _write_eleme(parameters, space_between_values):
    """Write ELEME block data."""
    from ._common import elements

    # Format
    label_length = len(max(parameters["elements"], key=len))
    fmt = block_to_format["ELEME"]
    fmt = str2format(fmt[label_length])

    out = []
    for k, v in parameters["elements"].items():
        data = deepcopy(elements)
        data.update(v)

        material = (
            f"{data['material']:>5}"
            if isinstance(data["material"], int)
            else data["material"]
        )
        center = data["center"] if data["center"] is not None else [None, None, None]
        values = [
            k,
            data["nseq"],
            data["nadd"],
            material,
            data["volume"],
            data["heat_exchange_area"],
            data["permeability_modifier"],
            center[0],
            center[1],
            center[2],
        ]
        out += write_record(values, fmt, space_between_values)

    return out


@block("COORD", multi=True)
def _write_coord(parameters, space_between_values):
    """Write COORD block data."""
    # Format
    fmt = block_to_format["COORD"]
    fmt = str2format(fmt)

    out = []
    for v in parameters["elements"].values():
        values = v["center"]
        out += write_record(values, fmt, space_between_values)

    return out


@block("CONNE", multi=True)
def _write_conne(parameters, space_between_values):
    """Write CONNE block data."""
    from ._common import connections

    # Return empty connection (i.e., single element)
    if "connections" not in parameters or not parameters["connections"]:
        return []

    # Format
    label_length = len(max(parameters["connections"], key=len)) // 2
    fmt = block_to_format["CONNE"]
    fmt = str2format(fmt[label_length])

    out = []
    for k, v in parameters["connections"].items():
        data = deepcopy(connections)
        data.update(v)

        values = [
            k,
            data["nseq"],
            data["nadd"][0] if data["nadd"] is not None else None,
            data["nadd"][1] if data["nadd"] is not None else None,
            data["permeability_direction"],
            data["nodal_distances"][0],
            data["nodal_distances"][1],
            data["interface_area"],
            data["gravity_cosine_angle"],
            data["radiant_emittance_factor"],
        ]
        out += write_record(values, fmt, space_between_values)

    return out


@block("INCON", multi=True)
def _write_incon(parameters, space_between_values, eos_=None, simulator="tough"):
    """Write INCON block data."""
    from ._common import initial_conditions

    # Format
    label_length = len(max(parameters["initial_conditions"], key=len))
    label_length = max(label_length, 5)
    fmt = block_to_format["INCON"]
    fmt1 = str2format(
        fmt[simulator][label_length]
        if simulator == "toughreact"
        else fmt[eos_][label_length] if eos_ in fmt else fmt["default"][label_length]
    )
    fmt2 = str2format(fmt[0])

    out = []
    for k, v in parameters["initial_conditions"].items():
        data = deepcopy(initial_conditions)
        data.update(v)

        # Record 1
        values = [
            k,
            None,
            None,
            data["porosity"],
        ]

        if simulator == "toughreact":
            per = data["permeability"]
            per = [per] * 3 if not np.ndim(per) else per

            if not (isinstance(per, (list, tuple, np.ndarray)) and len(per) == 3):
                raise TypeError()

            values += [k for k in per]

        elif eos_ in {"eco2m", "tmvoc"}:
            values += [data["phase_composition"]]

        else:
            values += list(data["userx"])

        out += write_record(values, fmt1, space_between_values)

        # Record 2
        out += write_record(data["values"], fmt2, space_between_values, multi=True)

    return out


@block("MESHM", multi=True)
def _write_meshm(parameters, space_between_values):
    """Write MESHM block data."""
    from ._common import meshmaker, rz2d, xyz

    out = []

    if parameters["meshmaker"]:
        data = deepcopy(meshmaker)
        data.update(parameters["meshmaker"])

        # Format
        fmt = block_to_format["MESHM"]
        fmt1 = str2format(fmt[1])

        # Mesh type
        mesh_type = data["type"].upper() if data["type"] else data["type"]
        out += write_record([mesh_type], fmt1, space_between_values)

        # XYZ
        if mesh_type == "XYZ":
            fmt = fmt["XYZ"]
            fmt1 = str2format(fmt[1])
            fmt2 = str2format(fmt[2])
            fmt3 = str2format(fmt[3])

            # Record 1
            out += write_record([data["angle"]], fmt1, space_between_values)

            # Record 2
            for parameter_ in data["parameters"]:
                parameter = deepcopy(xyz)
                parameter.update(parameter_)

                values = [parameter["type"].upper()]
                ndim = np.ndim(parameter["sizes"])

                if ndim == 0:
                    values += [
                        parameter["n_increment"],
                        parameter["sizes"],
                    ]
                    out += write_record(values, fmt2, space_between_values)

                elif ndim == 1:
                    values += [
                        (
                            parameter["n_increment"]
                            if parameter["n_increment"]
                            else len(parameter["sizes"])
                        )
                    ]
                    out += write_record(values, fmt2, space_between_values)
                    out += write_record(
                        parameter["sizes"], fmt3, space_between_values, multi=True
                    )

                else:
                    raise ValueError()

            # Blank record
            out += ["\n"]

        # RZ2D
        elif mesh_type in {"RZ2D", "RZ2DL"}:
            fmt = fmt["RZ2D"]
            fmt0 = str2format(fmt[1])

            for parameter_ in data["parameters"]:
                parameter = deepcopy(rz2d)
                parameter.update(parameter_)

                parameter_type = parameter["type"].upper()
                out += write_record([parameter_type], fmt0, space_between_values)

                if parameter_type == "RADII":
                    fmt1 = str2format(fmt["RADII"][1])
                    fmt2 = str2format(fmt["RADII"][2])

                    out += write_record(
                        [len(parameter["radii"])], fmt1, space_between_values
                    )
                    out += write_record(
                        parameter["radii"], fmt2, space_between_values, multi=True
                    )

                elif parameter_type == "EQUID":
                    fmt1 = str2format(fmt["EQUID"])

                    values = [
                        parameter["n_increment"],
                        None,
                        parameter["size"],
                    ]
                    out += write_record(values, fmt1, space_between_values)

                elif parameter_type == "LOGAR":
                    fmt1 = str2format(fmt["LOGAR"])

                    values = [
                        parameter["n_increment"],
                        None,
                        parameter["radius"],
                        parameter["radius_ref"],
                    ]
                    out += write_record(values, fmt1, space_between_values)

                elif parameter_type == "LAYER":
                    fmt1 = str2format(fmt["LAYER"][1])
                    fmt2 = str2format(fmt["LAYER"][2])

                    out += write_record(
                        [len(parameter["thicknesses"])], fmt1, space_between_values
                    )
                    out += write_record(
                        parameter["thicknesses"], fmt2, space_between_values, multi=True
                    )

    if parameters["minc"]:
        out += _write_minc(parameters, space_between_values)

    return out


def _write_minc(parameters, space_between_values):
    """Write MESHM block data (MINC)."""
    from ._common import minc

    data = deepcopy(minc)
    data.update(parameters["minc"])

    # Format
    fmt = block_to_format["MESHM"]
    fmt0 = str2format(fmt[1])
    fmt1 = str2format(fmt["MINC"][1])
    fmt2 = str2format(fmt["MINC"][2])
    fmt3 = str2format(fmt["MINC"][3])

    # Mesh type
    out = write_record(["MINC"], fmt0, space_between_values)

    # Record 1
    values = [
        "PART",
        data["type"].upper(),
        None,
        data["dual"].upper(),
    ]
    out += write_record(values, fmt1, space_between_values)

    # Record 2
    values = [
        data["n_minc"],
        len(data["volumes"]),
        f"{data['where'].upper():<4}",
        *data["parameters"],
    ]
    out += write_record(values, fmt2, space_between_values)

    # Record 3
    out += write_record(data["volumes"], fmt3, space_between_values, multi=True)

    return out


@block("POISE")
def _write_poise(parameters):
    """Write POISE block data."""
    poise = parameters["react"]["poiseuille"]
    for key in ["start", "end", "aperture"]:
        if key not in poise:
            raise ValueError()

        if key != "aperture" and len(poise[key]) != 2:
            raise ValueError()

    values = [x for x in poise["start"][:2]]
    values += [x for x in poise["end"][:2]]
    values += [poise["aperture"]]
    out = [f"{' '.join(str(x) for x in values)}\n"]

    return out


@block("NOVER")
def _write_nover():
    """Write NOVER block data."""
    return []


@block("ENDCY")
def _write_endcy():
    """Write ENDCY block data."""
    return []
