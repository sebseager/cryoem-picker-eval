import sys
import os
import logging
import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path
from collections import namedtuple


# globals


Box = namedtuple("Box", ["x", "y", "w", "h", "conf"])
Box.__new__.__defaults__ = (0,)  # set defaults starting from rightmost arg (conf)
STAR_COL_X = "_rlnCoordinateX"
STAR_COL_Y = "_rlnCoordinateY"
STAR_COL_C = "_rlnAutopickFigureOfMerit"
STAR_COL_N = "_rlnMicrographName"
_COLS = ["x", "y", "w", "h", "conf", "name"]


# utils


def _log(msg, lvl=0, verbose=False):
    """
    Format and print message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if verbose=False)
    1: warning (print and continue execution)
    2: error (print and exit with code 1)
    """

    if lvl == 0 and not verbose:
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
    """
    Returns True if string s contains any number, otherwise False.
    """

    res = re.search("[0-9]", str(s)) is not None
    return res


# parsing


def star_to_df(path):
    """
    Convert any well formatted STAR file into a DataFrame with correct column headers.
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
    """
    Generate a dataframe from the TSV-like file at the specified path, skipping
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


def df_to_star(df, col_names, out_dir, name, do_overwrite=False):
    out_path = Path(out_dir) / name

    if out_path.exists() and not do_overwrite:
        _log("re-run with the overwrite flag to replace existing files", 2)

    df_cols = list(df.columns)
    star_loop = "data_\n\nloop_\n"
    for col, col_name in col_names.items():
        if col_name is None:
            continue
        try:
            idx = df_cols.index(col)
            star_loop += f"{col_name} #{idx + 1}\n"
        except ValueError:
            # _log(f"could not find data for STAR column ({e})", 1)
            # import pdb
            # pdb.set_trace()
            pass

    with open(out_path, "w") as f:
        f.write(star_loop)

    df.to_csv(out_path, header=True, sep="\t", index=False, mode="a")


def df_to_tsv(df, out_dir, name, do_overwrite=False):
    out_path = Path(out_dir) / name

    if out_path.exists() and not do_overwrite:
        _log("re-run with the overwrite flag to replace existing files", 2)

    df.to_csv(out_path, header=True, sep="\t", index=False)


# handler method


def process_conversion(
    paths,
    out_dir,
    in_fmt,
    out_fmt,
    boxsize,
    cols,
    suffix,
    single_out,
    multi_out,
    do_overwrite,
):

    # read input files into dataframes
    dfs = {}
    AUTO = "auto"
    if in_fmt == "star":
        default_cols = {
            "x": STAR_COL_X,
            "y": STAR_COL_Y,
            "w": None,
            "h": None,
            "conf": STAR_COL_C,
            "name": STAR_COL_N,
        }
        dfs = {Path(p).stem: star_to_df(p) for p in paths}
    elif in_fmt == "box":
        default_cols = {"x": 0, "y": 1, "w": 2, "h": 3, "conf": 4, "name": None}
        dfs = {Path(p).stem: tsv_to_df(p) for p in paths}
    elif in_fmt == "tsv":
        default_cols = {"x": 0, "y": 1, "w": None, "h": None, "conf": 2, "name": None}
        dfs = {Path(p).stem: tsv_to_df(p) for p in paths}

    # apply any default cols needed
    for k, v in default_cols.items():
        cols[k] = v if cols[k] == AUTO else cols[k]

    _log(f"using the following input column mapping:\n  {cols}", 0, verbose=True)

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
                df["w"] = [boxsize] * len(df.index)
                df["h"] = [boxsize] * len(df.index)
                df["x"] = df["x"] - df["w"].div(2)
                df["y"] = df["y"] - df["h"].div(2)
            elif in_fmt in ("box",) and out_fmt in ("star", "tsv"):
                df["x"] = df["x"] + df["w"].div(2)
                df["y"] = df["y"] + df["h"].div(2)
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
        out_dfs = {"all_coords": pd.concat(out_dfs, ignore_index=True)}

    if multi_out:
        if all("name" in df.columns for df in out_dfs.values()):
            grouped_by_mrc = pd.concat(out_dfs, ignore_index=True).groupby("name")
            out_dfs = {k: df.drop("name", axis=1) for k, df in grouped_by_mrc}
        else:
            _log("cannot fulfill multi_out without micrograph name information", 1)

    for name, df in out_dfs.items():
        filename = f"{name}.{out_fmt}"
        if out_fmt == "star":
            df_to_star(df, cols, out_dir, filename)
        elif out_fmt in ("box", "tsv"):
            df_to_tsv(df, out_dir, filename)


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
        "-s",
        default="_converted",
        type=str,
        help="Suffix to append to generated output (default: _converted)",
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
        "--overwrite",
        action="store_true",
        help="Allow files in output directory to be overwritten",
    )

    a = parser.parse_args()

    # validation
    if a.f in ("star", "tsv") and a.b is None:
        _log(f"box size required for '{a.f}' input", 2)
    if a.single_out and a.multi_out:
        _log(f"cannot fulfill both single_out and multi_out flags", 2)

    cols = {}
    for i, col in enumerate(_COLS):
        cols[col] = a.c[i] if a.c[i] != "none" else None

    process_conversion(
        np.atleast_1d(a.input),
        a.out_dir,
        a.f,
        a.t,
        a.b,
        cols,
        a.s,
        a.single_out,
        a.multi_out,
        a.overwrite,
    )
