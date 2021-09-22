import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from collections import namedtuple


# globals


Box = namedtuple("Box", ["x", "y", "w", "h", "conf"])
Box.__new__.__defaults__ = (0,)  # set defaults starting from rightmost arg (conf)
STAR_COL_X = "_rlnCoordinateX"
STAR_COL_Y = "_rlnCoordinateY"
STAR_COL_C = "_rlnAutopickFigureOfMerit"
STAR_COL_N = "_rlnMicrographName"
DF_COL_NAMES = ["x", "y", "w", "h", "conf", "name"]
STAR_HEADER_MAP = {
    "x": STAR_COL_X,
    "y": STAR_COL_Y,
    "w": None,
    "h": None,
    "conf": STAR_COL_C,
    "name": STAR_COL_N,
}
BOX_HEADER_MAP = {"x": 0, "y": 1, "w": 2, "h": 3, "conf": 4, "name": None}
TSV_HEADER_MAP = {"x": 0, "y": 1, "w": None, "h": None, "conf": 2, "name": None}

AUTO = "auto"


# utils


def _log(msg, lvl=0, quiet=False):
    """Format and print message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if quiet=True)
    1: warning (print and continue execution)
    2: error (print and exit with code 1)
    """

    if lvl == 0 and quiet:
        return

    prefix = ""
    if lvl == 0:
        prefix = "INFO: "
    elif lvl == 1:
        prefix = "WARN: "
    elif lvl == 2:
        prefix = "CRITICAL: "

    print(f"{prefix}{msg}")
    if lvl == 2:
        sys.exit(1)


def _is_int(x):
    try:
        int(x)
    except ValueError:
        return False
    return True


def _has_numbers(s):
    """Returns True if string s contains any number, otherwise False."""

    res = re.search("[0-9]", str(s)) is not None
    return res


def _make_parent_dir(path_str):
    par_dir = Path(path_str).parent.resolve()
    if not par_dir.is_dir():
        par_dir.mkdir(parents=True, exist_ok=True)


def _path_occupied(path_str):
    return Path(path_str).resolve().is_file()


# parsing


def star_to_df(path):
    """Convert any well formatted STAR file into a DataFrame with correct column
    headers.
    """

    header = {}
    header_line_count = 0  # file line index where data starts

    with open(path, mode="r") as f:
        for i, line in enumerate(f):
            ln = line.strip()
            if not ln:  # skip blank lines
                continue
            if line.startswith("_") and line.count("#") == 1:
                header_entry = "".join(ln).split("#")
                try:
                    header[int(header_entry[1]) - 1] = header_entry[0].strip()
                except ValueError:
                    _log("STAR file not properly formatted", 2)
            elif header and _has_numbers(line):
                header_line_count = i
                break  # we've reached coordinate data

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        skip_blank_lines=True,
        skiprows=header_line_count,
    )

    # rename columns according to STAR header
    df = df.rename(columns={df.columns[k]: v for k, v in header.items()})

    return df


def tsv_to_df(path):
    """Generate a dataframe from the TSV-like file at the specified path, skipping
    any non-numeric header rows.
    """

    header_line_count = 0  # file line index where data starts

    with open(path, mode="r") as f:
        for i, line in enumerate(f):
            if _has_numbers(line):
                header_line_count = i
                break

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        skip_blank_lines=True,
        skiprows=header_line_count,
    )

    return df


# writing


def df_to_star(df, out_path, do_force=False):
    """Write df generated from one of the *_to_df methods in this module out to file
    with appropriate STAR header prepended.
    """

    if do_force:
        _make_parent_dir(out_path)
    else:
        if _path_occupied(out_path):
            _log("re-run with the force flag to replace existing files", 2)

    df_cols = list(df.columns)
    star_loop = "data_\n\nloop_\n"
    for df_col, star_col in STAR_HEADER_MAP.items():
        if star_col is None:
            continue
        try:
            idx = df_cols.index(df_col)
            star_loop += f"{star_col} #{idx + 1}\n"
        except ValueError:
            pass

    with open(out_path, "w") as f:
        f.write(star_loop)

    df.to_csv(out_path, header=False, sep="\t", index=False, mode="a")


def df_to_tsv(df, col_order, out_path, include_header=False, do_force=False):
    """Write df generated from one of the *_to_df methods in this module out to file,
    optionally writing out [x, y, w, h, conf] labels as a header.
    """

    if do_force:
        _make_parent_dir(out_path)
    else:
        if _path_occupied(out_path):
            _log("re-run with the force flag to replace existing files", 2)

    out_cols = [c for c in col_order if c in df.columns]
    df[out_cols].to_csv(out_path, header=include_header, sep="\t", index=False)


# handler method


def process_conversion(
    paths,
    in_fmt,
    out_fmt,
    boxsize=None,
    out_dir=None,
    in_cols=("auto", "auto", "auto", "auto", "auto", "auto"),
    out_col_order=("x", "y", "w", "h", "conf", "name"),
    suffix="",
    include_header=False,
    single_out=False,
    multi_out=False,
    round_to=None,
    do_force=False,
    quiet=False,
):

    # set default columns as needed
    cols = {}
    for i, col in enumerate(DF_COL_NAMES):
        cols[col] = in_cols[i] if in_cols[i] != "none" else None

    # read input files into dataframes
    dfs = {}
    if in_fmt == "star":
        default_cols = STAR_HEADER_MAP
        dfs = {Path(p).stem: star_to_df(p) for p in paths}
    elif in_fmt == "box":
        default_cols = BOX_HEADER_MAP
        dfs = {Path(p).stem: tsv_to_df(p) for p in paths}
    elif in_fmt == "tsv":
        default_cols = TSV_HEADER_MAP
        dfs = {Path(p).stem: tsv_to_df(p) for p in paths}

    # apply any default cols needed
    for k, v in default_cols.items():
        cols[k] = v if cols[k] == AUTO else cols[k]

    _log(f"using the following input column mapping:\n  {cols}", 0, quiet=quiet)

    out_dfs = {}
    for name, df in dfs.items():
        # rename columns to make conversion logic easier
        rename_dict = {}
        for new_name, cur_name in cols.items():
            if cur_name is None:
                continue
            if _is_int(cur_name):
                cur_name = int(cur_name)
                if cur_name in range(len(df.columns)):
                    rename_dict[df.columns[cur_name]] = new_name
            else:
                if cur_name in df.columns:
                    rename_dict[cur_name] = new_name

        df = df.rename(columns=rename_dict)

        # modify coordinates for output format if needed
        try:
            if in_fmt in ("star", "tsv") and out_fmt in ("box",):
                assert boxsize is not None, "Expected integer boxsize but got None"
                df["w"] = boxsize
                df["h"] = boxsize
                df["x"] = df["x"] - df["w"].div(2)
                df["y"] = df["y"] - df["h"].div(2)
            elif in_fmt in ("box",) and out_fmt in ("star", "tsv"):
                df["x"] = df["x"] + df["w"].div(2)
                df["y"] = df["y"] + df["h"].div(2)

            if round_to is not None:
                df["x"] = df["x"].round(round_to)
                df["y"] = df["y"].round(round_to)
                if round_to == 0:
                    df["x"] = df["x"].astype(int)
                    df["y"] = df["y"].astype(int)

        except KeyError as e:
            _log(f"did not find column {e} in input columns ({list(df.columns)})", 2)
        except TypeError as e:
            _log(f"unexpected type in input column(s) ({list(df.columns)})", 2)

        if out_fmt in ("star", "tsv"):
            out_cols = ["x", "y", "conf", "name"]
        elif out_fmt == "box":
            out_cols = ["x", "y", "w", "h", "conf", "name"]

        out_dfs[name] = df[[x for x in out_cols if x in df.columns]]

    if single_out:
        out_dfs = {"all": pd.concat(out_dfs, ignore_index=True)}

    if multi_out:
        if all("name" in df.columns for df in out_dfs.values()):
            grouped_by_mrc = pd.concat(out_dfs, ignore_index=True).groupby("name")
            out_dfs = {k: df.drop("name", axis=1) for k, df in grouped_by_mrc}
        else:
            _log("cannot fulfill multi_out without micrograph name information", 1)

    if out_dir is None:
        return out_dfs

    for name, df in out_dfs.items():
        filename = f"{name}{suffix}.{out_fmt}"
        out_path = Path(out_dir) / filename
        if out_fmt == "star":
            df_to_star(df, out_path, do_force=do_force)
        elif out_fmt in ("box", "tsv"):
            _log(
                f"using the following output column order:\n  {out_col_order}",
                0,
                quiet=quiet,
            )
            df_to_tsv(
                df,
                out_col_order,
                out_path,
                include_header=include_header,
                do_force=do_force,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts particle coordinate file data between "
        "several different formats. The -f (input format) and -t (output format) "
        "parameters define the conversion. The -c argument can be used if more "
        "granular control over column indices is required."
    )
    parser.add_argument(
        "input", help="Path(s) to input particle coordinates", nargs="+"
    )
    parser.add_argument(
        "out_dir",
        help="Output directory in which to store generated coordinate files (will be "
        "created if it does not exist)",
    )
    parser.add_argument(
        "-f",
        choices=["star", "box", "tsv"],
        help="Format FROM which to convert the input",
    )
    parser.add_argument(
        "-t",
        choices=["star", "box", "tsv"],
        help="Format TO which to convert the input",
    )
    parser.add_argument(
        "-b",
        type=int,
        help="Specifies or overrides the box size to be used "
        "(required if input does not include a box size)",
    )
    parser.add_argument(
        "-c",
        nargs=6,
        metavar=("X_COL", "Y_COL", "W_COL", "H_COL", "CONF_COL", "NAME_COL"),
        default=("auto", "auto", "auto", "auto", "auto", "auto"),
        help="Manually specify input column names (STAR) or zero-based indices "
        "(BOX/TSV). This can be useful if input file does not follow default column "
        "indices of the specified input (-f) format. Expects six positional arguments, "
        "corresponding to: [x, y, w, g, conf, mrc_name]. Set a column to 'none' to "
        "exclude it from conversion and 'auto' to keep its default value.",
    )
    parser.add_argument(
        "-d",
        nargs=6,
        default=("x", "y", "w", "h", "conf", "name"),
        help="Manually specify the order of columns in output files (only applies to "
        "BOX/TSV output formats). Expects six positional arguments, which should be "
        "some ordering of the strings ['x', 'y', 'w', 'h', 'conf', 'name']. Use any "
        "other string (like 'none') at any of the six positions to exclude the "
        "missing column from the output.",
    )
    parser.add_argument(
        "-s",
        default="",
        type=str,
        help="Suffix to append to generated output (default: _converted)",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="If output format is BOX or TSV, include column headers (has no effect "
        "with STAR output)",
    )
    parser.add_argument(
        "--single_out",
        action="store_true",
        help="If possible, make output a single file, with column for micrograph name",
    )
    parser.add_argument(
        "--multi_out",
        action="store_true",
        help="If possible, split output into multiple files by micrograph name",
    )
    parser.add_argument(
        "--round",
        default=None,
        type=int,
        help="Round coordinates to the specified number of decimal places "
        "(don't round by default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow files in output directory to be overwritten and make output "
        "directory if it does not exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence info-level output",
    )

    a = parser.parse_args()

    # validation
    if a.f in ("star", "tsv") and a.b is None:
        _log(f"box size required for '{a.f}' input", 2)
    if a.single_out and a.multi_out:
        _log(f"cannot fulfill both single_out and multi_out flags", 2)

    process_conversion(
        paths=np.atleast_1d(a.input),
        in_fmt=a.f,
        out_fmt=a.t,
        boxsize=a.b,
        out_dir=a.out_dir,
        in_cols=a.c,
        out_col_order=a.d,
        suffix=a.s,
        include_header=a.header,
        single_out=a.single_out,
        multi_out=a.multi_out,
        round_to=a.round,
        do_force=a.force,
        quiet=a.quiet,
    )
