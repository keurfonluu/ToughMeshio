import numpy as np

from .._io.output import read as read_output
from .._io.output import write as write_output

__all__ = [
    "extract",
]


format_to_ext = {
    "csv": ".csv",
    "tecplot": ".tec",
    "petrasim": ".csv",
}


def extract(argv=None):
    import os

    from .. import read_mesh

    parser = _get_parser()
    args = parser.parse_args(argv)

    # Check output file format
    if args.connection and args.file_format != "csv":
        raise ValueError("Connection data can only be exported to CSV.")

    # Check that TOUGH output and MESH file exist
    if not os.path.isfile(args.infile):
        raise ValueError(f"TOUGH output file '{args.infile}' not found.")
    if not os.path.isfile(args.mesh):
        raise ValueError(f"MESH file '{args.mesh}' not found.")

    # Read MESH and extract X, Y and Z
    parameters = read_mesh(
        args.mesh, file_format="tough", label_length=args.label_length
    )
    if "elements" not in parameters:
        raise ValueError(f"Invalid MESH file '{args.mesh}'.")

    # Read TOUGH output file
    output = read_output(args.infile, connection=args.connection)
    output = output if isinstance(output, list) else [output]

    try:
        if not args.connection:
            points = np.vstack(
                [parameters["elements"][label]["center"] for label in output[-1].labels]
            )
        else:
            points = np.array(
                [
                    np.mean(
                        [parameters["elements"][l]["center"] for l in label], axis=0
                    )
                    for label in output[-1].labels
                ]
            )
        points = {k: v for k, v in zip(["X", "Y", "Z"], points.T)}
        for out in output:
            out.data.update(points)

    except KeyError:
        raise ValueError(
            f"Elements in '{args.infile}' and '{args.mesh}' are not consistent."
        )

    # Write TOUGH3 element output file
    ext = format_to_ext[args.file_format]
    filename = (
        args.output_file
        if args.output_file is not None
        else f"OUTPUT_ELEME{ext}" if not args.connection else f"OUTPUT_CONNE{ext}"
    )
    if not args.split or len(output) == 1:
        write_output(filename, output, file_format=args.file_format)
    else:
        head, ext = os.path.splitext(filename)
        for i, out in enumerate(output):
            write_output(f"{head}_{i + 1}{ext}", out, file_format="csv")


def _get_parser():
    import argparse

    # Initialize parser
    parser = argparse.ArgumentParser(
        description=(
            "Extract results from TOUGH main output file and reformat as a TOUGH3 CSV output file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input file
    parser.add_argument(
        "infile",
        type=str,
        help="TOUGH output file",
    )

    # Mesh file
    parser.add_argument(
        "mesh",
        type=str,
        help="TOUGH MESH file (can be INFILE)",
    )

    # Output file
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=None,
        help="TOUGH3 element output file",
    )

    # File format
    parser.add_argument(
        "--file-format",
        "-f",
        type=str,
        choices=("csv", "tecplot", "petrasim"),
        default="csv",
        help="exported file format",
    )

    # Label length
    parser.add_argument(
        "--label-length",
        "-l",
        type=int,
        choices=(5, 6, 7, 8, 9),
        default=None,
        help="number of characters in cell labels",
    )

    # Split or not
    parser.add_argument(
        "--split",
        "-s",
        default=False,
        action="store_true",
        help="write one file per time step",
    )

    # Read connection data
    parser.add_argument(
        "--connection",
        "-c",
        default=False,
        action="store_true",
        help="extract data related to connections",
    )

    return parser
