import numpy as np


def read_record(data, fmt):
    """Parse string to data given format."""
    token_to_type = {
        "s": str,
        "S": str,
        "d": int,
        "f": to_float,
        "e": to_float,
    }

    i = 0
    out = []
    for token in fmt.split(","):
        n = int(token[:-1].split(".")[0])
        tmp = data[i : i + n]
        tmp = tmp if token[-1] == "S" else tmp.strip()
        out.append(token_to_type[token[-1]](tmp) if tmp else None)
        i += n

    return out


def write_record(data, fmt, space_between_values=False, multi=False):
    """Return a list of record strings given format."""
    if not multi:
        data = [to_str(d, f, space_between_values) for d, f in zip(data, fmt)]
        out = [f"{''.join(data):80}\n"]

    else:
        n = len(data)
        ncol = len(fmt)
        data = [
            data[ncol * i : min(ncol * i + ncol, n)]
            for i in range(int(np.ceil(n / ncol)))
        ]

        out = []
        for d in data:
            d = [to_str(dd, f, space_between_values) for dd, f in zip(d, fmt)]
            out += [f"{''.join(d):80}\n"]

    return out


def to_float(s):
    """Convert variable string to float."""
    try:
        return float(s.replace("d", "e"))

    except ValueError:
        # It's probably something like "0.0001-001"
        significand, exponent = s[:-4], s[-4:]

        return float(f"{significand}e{exponent}")


def to_str(x, fmt, space_between_values=False):
    """Convert variable to string."""
    from .. import scientific_notation

    x = "" if x is None else x

    if not isinstance(x, str):
        if "f" in fmt:
            tmp = str(float(x))

            n = int(fmt[3:].split("f")[0])
            fmt = f"{{:>{n}}}"

            if space_between_values:
                n -= 1

            if len(tmp) > n or "e" in tmp:
                tmp = (
                    tmp[:n]
                    if 1.0 <= abs(x) < 10.0
                    else format(scientific_notation(x, n))
                )

            return fmt.format(tmp)

        else:
            return fmt.format(x)

    else:
        return fmt.replace("g", "").replace("f", "").format(x)
