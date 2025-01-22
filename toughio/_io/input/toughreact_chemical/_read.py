from ...._common import open_file
from ....core import FileIterator, ReadError
from ..._common import to_float
from .._common import read_end_comments

__all__ = [
    "read",
]


def read(filename):
    """
    Read TOUGHREACT chemical input file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Input file name or buffer.

    """
    with open_file(filename, "r") as f:
        out = read_buffer(f)

    return out


def read_buffer(f):
    """Read TOUGHREACT chemical input file."""
    parameters = {}
    fiter = FileIterator(f)

    try:
        parameters.update(_read_title(fiter))
        parameters.update(_read_prim(fiter))
        parameters.update(_read_akin(fiter))
        parameters.update(_read_aque(fiter))
        parameters.update(_read_miner(fiter))
        parameters.update(_read_gas(fiter))
        parameters.update(_read_surx(fiter))
        parameters.update(_read_kdde(fiter))

        exch, nxsites = _read_exch(fiter)
        parameters.update(exch)

        parameters["zones"] = {}
        parameters["zones"].update(_read_water(fiter))
        parameters["zones"].update(_read_imin(fiter))
        parameters["zones"].update(_read_igas(fiter))
        parameters["zones"].update(_read_zppr(fiter))
        parameters["zones"].update(_read_zads(fiter))
        parameters["zones"].update(_read_zlkd(fiter))
        parameters["zones"].update(_read_zexc(fiter, nxsites))

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


def _read_prim(f):
    """Read primary species."""
    prim = {"primary_species": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.split()
        if len(data) < 2:
            raise ReadError()

        tmp = {"name": data[0], "transport": int(data[1])}
        if len(data) >= 3:
            tmp["mineral_name"] = data[2]
        if len(data) >= 4:
            tmp["sorption_density"] = to_float(data[3])
        if len(data) >= 5:
            tmp["adsorption_id"] = int(data[4])
        if len(data) >= 6:
            tmp["capacitance"] = to_float(data[5])
        prim["primary_species"].append(tmp)

        line = _nextline(f).strip()

    return prim


def _read_akin(f):
    """Read aqueous kinetics."""
    akin = {"aqueous_kinetics": None}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 2
    ntrx = int(line.strip())
    akin["aqueous_kinetics"] = []

    # Loop ntrx times
    for _ in range(ntrx):
        # Record 3
        line = _nextline(f).strip()
        # irx = int(line.strip())

        # Record 4
        line = _nextline(f).strip()
        data = line.split()
        ncp = int(data[0])
        if len(data) < 2 * ncp + 1:
            raise ReadError()

        names = data[2 : 2 * ncp + 2 : 2]
        coeffs = data[1 : 2 * ncp + 1 : 2]
        tmp = {
            "species": [
                {
                    "name": name,
                    "stoichiometric_coeff": to_float(coeff),
                }
                for name, coeff in zip(names, coeffs)
            ]
        }

        if len(data) > 2 * ncp + 1:
            tmp["reaction_affinity"] = {
                "id": int(data[2 * ncp + 1]),
                "cf": to_float(data[2 * ncp + 2]),
                "logK": to_float(data[2 * ncp + 3]),
            }

        # Record 5
        line = _nextline(f).strip()
        data = line.split()
        tmp["id"] = int(data[0])
        tmp["n_mechanism"] = int(data[1])

        # Record 6
        line = _nextline(f).strip()
        data = to_float(line.split()[0])

        if data != -1.0:
            tmp["rate"] = data

        else:
            # Record 6.1
            line = _nextline(f).strip()
            data = line.split()
            tmp["rate"] = {
                "k25": to_float(data[0]),
                "Ea": to_float(data[1]),
            }

        # Record 7
        line = _nextline(f).strip()
        data = line.split()
        ncp_rx1 = int(data[0])
        if len(data) < 3 * ncp_rx1 + 1:
            raise ReadError()

        tmp["product"] = [
            {
                "specie": data[3 * i + 1],
                "flag": int(data[3 * i + 2]),
                "power": to_float(data[3 * i + 3]),
            }
            for i in range(ncp_rx1)
        ]

        # Record 8
        line = _nextline(f).strip()
        data = line.split()
        ncp_rx2 = int(data[0])
        if len(data) < 3 * ncp_rx2 + 1:
            raise ReadError()

        tmp["monod"] = [
            {
                "specie": data[3 * i + 1],
                "flag": int(data[3 * i + 2]),
                "half_saturation": to_float(data[3 * i + 3]),
            }
            for i in range(ncp_rx2)
        ]

        # Record 9
        line = _nextline(f).strip()
        data = line.split()
        ncp_rx3 = int(data[0])
        if len(data) < 3 * ncp_rx3 + 1:
            raise ReadError()

        tmp["inhibition"] = [
            {
                "specie": data[3 * i + 1],
                "flag": int(data[3 * i + 2]),
                "constant": to_float(data[3 * i + 3]),
            }
            for i in range(ncp_rx3)
        ]

        akin["aqueous_kinetics"].append(tmp)

    # '*' is not used here to mark the end of the list. Skip it.
    _ = f.next()

    return akin


def _read_aque(f):
    """Read aqueous species."""
    aque = {"aqueous_species": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.strip()
        aque["aqueous_species"].append(data)

        line = _nextline(f)

    return aque


def _read_miner(f):
    """Read minerals."""
    miner = {"minerals": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Loop until reading character *
    while not line.startswith("*"):
        # Record 2
        data = _nextsplitline(line, 5)

        tmp = {
            "name": data[0],
            "type": int(data[1]),
            "kinetic_constraint": int(data[2]),
            "solid_solution": int(data[3]),
            "precipitation_dry": int(data[4]),
        }

        # Records 2.1 and 2.2
        # The 8 first entries of record 2.2 are common with record 2.1
        if tmp["type"] == 1:
            n = 2 if tmp["kinetic_constraint"] == 3 else 1
            for ir in range(n):
                if tmp["kinetic_constraint"] == 1:
                    key = "dissolution"

                elif tmp["kinetic_constraint"] == 2:
                    key = "precipitation"

                elif tmp["kinetic_constraint"] == 3:
                    key = "dissolution" if ir == 0 else "precipitation"

                data = (
                    _nextsplitline(f, 8)
                    if key == "dissolution"
                    else _nextsplitline(f, 10)
                )

                tmp[key] = {}
                tmp[key]["k25"] = to_float(data[0])
                tmp[key]["rate_ph_dependence"] = int(data[1])
                tmp[key]["eta"] = to_float(data[2])
                tmp[key]["theta"] = to_float(data[3])
                tmp[key]["activation_energy"] = to_float(data[4])
                tmp[key]["a"] = to_float(data[5])
                tmp[key]["b"] = to_float(data[6])
                tmp[key]["c"] = to_float(data[7])

                # Record 2.2
                if key == "precipitation":
                    tmp[key]["volume_fraction_ini"] = to_float(data[8])
                    tmp[key]["id"] = int(data[9])

                # Record 2.1.1
                if tmp[key]["rate_ph_dependence"] == 1:
                    data = _nextsplitline(f, 4)

                    tmp[key]["ph1"] = to_float(data[0])
                    tmp[key]["slope1"] = to_float(data[1])
                    tmp[key]["ph2"] = to_float(data[2])
                    tmp[key]["slope2"] = to_float(data[3])

                # Record 2.1.2
                elif tmp[key]["rate_ph_dependence"] == 2:
                    tmp[key]["extra_mechanisms"] = []

                    line = _nextline(f).strip()
                    ndis = int(line.strip())

                    # Record 2.1.2.1
                    for _ in range(ndis):
                        line = _nextline(f).strip()
                        data = line.split()
                        nspds = int(data[2])
                        if len(data) < 2 * nspds + 3:
                            raise ReadError()

                        tmp[key]["extra_mechanisms"].append(
                            {
                                "ki": to_float(data[0]),
                                "activation_energy": to_float(data[1]),
                                "species": [
                                    {
                                        "name": data[2 * i + 3],
                                        "power": to_float(data[2 * i + 4]),
                                    }
                                    for i in range(nspds)
                                ],
                            }
                        )

        # Record 3
        if tmp["kinetic_constraint"] != 1:
            data = _nextsplitline(f, 3)
            tmp["gap"] = to_float(data[0])
            tmp["temp1"] = to_float(data[1])
            tmp["temp2"] = to_float(data[2])

        miner["minerals"].append(tmp)
        line = _nextline(f)

    return miner


def _read_gas(f):
    """Read gaseous species."""
    gas = {"gaseous_species": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.split()
        if len(data) < 2:
            raise ReadError()

        tmp = {
            "name": data[0],
            "fugacity": int(data[1]),
        }
        gas["gaseous_species"].append(tmp)

        line = _nextline(f).strip()

    return gas


def _read_surx(f):
    """Read surface complexe."""
    surx = {"surface_complexes": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.strip()
        surx["surface_complexes"].append(data)

        line = _nextline(f)

    return surx


def _read_kdde(f):
    """Read Kd and decay."""
    kdde = {"kd_decay": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.split()
        if len(data) < 4:
            raise ReadError()

        tmp = {
            "name": data[0],
            "decay_constant": to_float(data[1]),
            "a": to_float(data[2]),
            "b": to_float(data[3]),
        }
        kdde["kd_decay"].append(tmp)

        line = _nextline(f).strip()

    return kdde


def _read_exch(f):
    """Read exchanged species."""
    exch = {"exchanged_species": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}, 0

    # Record 2
    data = line.split()
    if len(data) < 2:
        raise ReadError()

    nxsites = int(data[0])
    exch["exchange_sites_id"] = int(data[1])

    # Record 4
    line = _nextline(f, skip_empty=True, comments="#")

    # Loop until reading character *
    while not line.startswith("*"):
        data = line.split()
        if len(data) < nxsites + 3:
            raise ReadError()

        tmp = {
            "name": data[0],
            "reference": bool(int(data[1])),
            "type": int(data[2]),
            "site_coeffs": [to_float(x) for x in data[3:]],
        }
        exch["exchanged_species"].append(tmp)

        line = _nextline(f).strip()

    return exch, nxsites


def _read_water(f):
    """Read water zones."""
    zones = {"initial_waters": [], "injection_waters": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 2
    data = line.split()
    if len(data) < 2:
        raise ReadError()

    niwtype = int(data[0])
    nbwtype = int(data[1])

    # Loop niwtype + nbwtype times
    for i in range(niwtype + nbwtype):
        key = "initial_waters" if i < niwtype else "injection_waters"

        # Record 4
        line = _nextline(f, skip_empty=True, comments="#")
        data = line.split()
        iwtype = int(data[0])

        if iwtype >= 0 and len(data) < 3:
            raise ReadError()

        elif iwtype < 0 and len(data) < 4:
            raise ReadError()

        tmp = {
            "temperature": to_float(data[1]),
            "pressure": to_float(data[2]),
        }
        if iwtype < 0:
            tmp["rock"] = data[3]
        tmp["species"] = []

        # Record 6
        line = _nextline(f, remove_quotes=False, skip_empty=True, comments="#")
        while not line.replace("'", "").startswith("*"):
            data = _nextsplitline(line, 6)

            # split() doesn't work in this case as 'nameq' can be ' ' if not needed
            if data[4].startswith("'") and data[5].endswith("'"):
                nameq = (data[4] + data[5]).replace("'", "").strip()
                log_fugacity = data[6]

            else:
                nameq = data[4].replace("'", "")
                log_fugacity = data[5]

            tmp2 = {
                "name": data[0].replace("'", ""),
                "flag": int(data[1]),
                "guess": to_float(data[2]),
                "ctot": to_float(data[3]),
                "log_fugacity": to_float(log_fugacity),
            }
            if nameq:
                tmp2["nameq"] = nameq

            tmp["species"].append(tmp2)
            line = _nextline(f, remove_quotes=False).strip()

        zones[key].append(tmp)

    return zones


def _read_imin(f):
    """Read mineral zones."""
    zones = {"minerals": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    nmtype = int(line.strip())

    # Loop nmtype times
    for _ in range(nmtype):
        # Record 4
        line = _nextline(f, skip_empty=True, comments="#")
        data = line.split()
        imtype = int(data[0])

        if imtype >= 0 and len(data) < 1:
            raise ReadError()

        elif imtype < 0 and len(data) < 2:
            raise ReadError()

        tmp = {}
        if imtype < 0:
            tmp["rock"] = data[1]
        tmp["species"] = []

        # Record 6
        line = _nextline(f, skip_empty=True, comments="#")
        while not line.startswith("*"):
            data = _nextsplitline(line, 3)
            tmp2 = {
                "name": data[0],
                "volume_fraction_ini": to_float(data[1]),
                "flag": int(data[2]),
            }

            # Record 6.1
            if tmp2["flag"] == 1:
                data = _nextsplitline(f, 3)

                tmp2["radius"] = to_float(data[0])
                tmp2["area_ini"] = to_float(data[1])
                tmp2["area_unit"] = int(data[2])

                # TODO: investigate what is the next record when IMFLG2 < 0
                # This record is not documented in the user guide (p. 65)
                if tmp2["area_unit"] < 0:
                    data = _nextsplitline(f, 4)
                    tmp2["unknown"] = [
                        int(data[0]),
                        to_float(data[1]),
                        int(data[2]),
                        to_float(data[3]),
                    ]

            tmp["species"].append(tmp2)
            line = _nextline(f).strip()

        zones["minerals"].append(tmp)

    return zones


def _read_igas(f):
    """Read gas zones."""
    zones = {"initial_gases": [], "injection_gases": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    data = _nextsplitline(line, 2)
    nigtype = int(data[0])
    nbgtype = int(data[1])

    # Loop nigtype + nbgtype times
    for i in range(nigtype + nbgtype):
        key = "initial_gases" if i < nigtype else "injection_gases"

        # Record 4
        data = _nextsplitline(f, 1, skip_empty=True, comments="#")
        # igtype = int(data[0])

        tmp = []

        # Record 6
        line = _nextline(f, skip_empty=True, comments="#")
        while not line.replace("'", "").startswith("*"):
            data = _nextsplitline(line, 2)
            tmp2 = {"name": data[0]}
            if key == "initial_gases":
                tmp2["partial_pressure"] = to_float(data[1])

            else:
                tmp2["mole_fraction"] = to_float(data[1])

            tmp.append(tmp2)
            line = _nextline(f).strip()

        zones[key].append(tmp)

    return zones


def _read_zppr(f):
    """Read permeability-porosity law zones."""
    zones = {"permeability_porosity": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    nppzone = int(line.strip())

    # Loop nppzone times
    for _ in range(nppzone):
        # Record 4
        data = _nextsplitline(f, 1, skip_empty=True, comments="#")
        # ippzone = int(data[0])

        # Record 6
        # Here, '*' does not mark the end of the list
        data = _nextsplitline(f, 3, skip_empty=True, comments="#")
        tmp = {
            "id": int(data[0]),
            "a": to_float(data[1]),
            "b": to_float(data[2]),
        }

        zones["permeability_porosity"].append(tmp)

    # '*' is not used here to mark the end of the list. Skip it.
    _ = f.next()

    return zones


def _read_zads(f):
    """Read surface adsorption zones."""
    zones = {"adsorption": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    ndtype = int(line.strip())

    # Loop ndtype times
    for _ in range(ndtype):
        # Record 5
        data = _nextsplitline(f, 2, skip_empty=True, comments="#")
        # idzone = int(data[0])

        tmp = {
            "flag": int(data[1]),
            "species": [],
        }

        # Record 6
        # Not sure about the parsing, the manual states that '*' is only
        # written once in record 7 at the end of the zone list
        # If it's the case, it's unclear how the code knows how many
        # records to read per zone
        line = _nextline(f, skip_empty=True, comments="#")
        while not line.replace("'", "").startswith("*"):
            data = _nextsplitline(line, 3)
            tmp2 = {
                "name": data[0],
                "area_unit": int(data[1]),
                "area": to_float(data[2]),
            }

            tmp["species"].append(tmp2)
            line = _nextline(f).strip()

        zones["adsorption"].append(tmp)

    # '*' is not used here to mark the end of the list. Skip it.
    # _ = f.next()

    return zones


def _read_zlkd(f):
    """Read linear Kd zones."""
    zones = {"linear_kd": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    kdtype = int(line.strip())

    # Loop kdtype times
    for _ in range(kdtype):
        # Record 4
        data = _nextsplitline(f, 1, skip_empty=True, comments="#")
        # idtype = int(data[0])

        tmp = []

        # Record 6
        # Not sure about the parsing, the manual states that '*' is only
        # written once in record 7 at the end of the zone list
        # If it's the case, it's unclear how the code knows how many
        # records to read per zone
        line = _nextline(f, skip_empty=True, comments="#")
        while not line.replace("'", "").startswith("*"):
            data = _nextsplitline(line, 3)
            tmp2 = {
                "name": data[0],
                "solid_density": to_float(data[1]),
                "value": to_float(data[2]),
            }

            tmp.append(tmp2)
            line = _nextline(f).strip()

        zones["linear_kd"].append(tmp)

    # '*' is not used here to mark the end of the list. Skip it.
    # _ = f.next()

    return zones


def _read_zexc(f, nxsites):
    """Read cation exchange zones."""
    zones = {"cation_exchange": []}

    # Find next non skip record
    line = _nextline(f, skip_empty=True, comments="#")

    if line.startswith("*"):
        return {}

    # Record 3
    nxtype = int(line.strip())

    # Loop nxtype times
    for _ in range(nxtype):
        # Record 5
        data = _nextsplitline(f, nxsites + 1, skip_empty=True, comments="#")
        # ixtype = int(data[0])

        tmp = [to_float(x) for x in data[1:]]
        zones["cation_exchange"].append(tmp)

    # '*' is not used here to mark the end of the list. Skip it.
    _ = f.next()

    return zones


def _nextline(f, remove_quotes=True, **kwargs):
    """
    Go to next line.

    Note
    ----
    Remove trailing Fortran comments.

    """
    line = f.next(**kwargs)
    line = line.replace('"', "'")

    # Remove quotes
    if remove_quotes:
        line = line.replace("'", "")

    # Remove Fortran comments
    if "!" in line:
        line = line[: line.index("!")]

    return line


def _nextsplitline(f_or_line, n=0, **kwargs):
    """Go to next line and split it."""
    if not isinstance(f_or_line, str):
        line = _nextline(f_or_line, **kwargs).strip()
    else:
        line = f_or_line

    data = line.split()
    if len(data) < n:
        raise ReadError(f"expected at least {n} entries, got {len(data)}.")

    return data
