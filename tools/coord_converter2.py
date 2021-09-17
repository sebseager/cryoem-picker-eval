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


def _normalize_1d_list(x, new_min=0, new_max=1, default=None):
    """
    Normalize numerical list x by subtracting each element by the list minimum,
    dividing by the old range, multiplying by the new range, and adding the new
    minimum. If range of input list is 0, return list x if default=None, otherwise
    return list of length len(x) with all values equal to default.
    """

    old_min = float(min(x))
    old_max = float(max(x))
    old_range = old_max - old_min
    new_range = float(new_max - new_min)

    if old_range == 0:  # prevent division by 0
        if default is None:
            res = list(x)
        else:
            res = [default] * len(list(x))
    else:
        res = [new_range * (float(n) - old_min) / old_range + new_min for n in x]

    return res


def _normalize_confidences(boxes):
    """
    Normalize any confidences in input box list between 0 and 1.
    """

    confs = _normalize_1d_list([b.conf for b in boxes], default=0)
    norm_boxes = [b._replace(conf=confs[i]) for i, b in enumerate(boxes)]
    return norm_boxes


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


"""





























"""

# MARKER


# def handle_coord_parsing(cfg, path, picker_name):
#     """
#     Parse coordinate file at specified path, using column overrides from given picker.

#     Parameters
#     ----------
#     cfg : dict
#         Run configuration as returned by read_config.
#     path : str
#         Input coordinate file path.
#     picker_name : str or None
#         Name of picker; one of strings specified by user in cfg['picker_names'], 'gt' for ground truth, or None
#         to use defaults.

#     Returns
#     -------
#     norm_boxes : list
#         List of Box namedtuples, with normalized confidences.
#     """

#     ext = Path(path).suffix.lower()  # get file path extension
#     boxsize = cfg["eval_boxsize"]

#     if picker_name is None:
#         force_cols = None  # if picker_name was None, use defaults
#     elif picker_name == "gt":
#         force_cols = cfg[
#             "gt_column_overrides"
#         ]  # if picker_name was 'gt', pull from appropriate cfg entry
#         if cfg["gt_column_overrides"] is not None:
#             boxsize = float(cfg["gt_column_overrides"])
#         if cfg["gt_boxsize_override"]:  # if not None or empty string
#             boxsize = float(cfg["gt_boxsize_override"])
#     else:
#         picker_index = cfg["picker_names"].index(picker_name)
#         force_cols = np.atleast_1d(
#             cfg["picker_column_overrides"]
#         ).tolist()  # force_cols needs to be list for any()
#         force_cols = (
#             force_cols[picker_index] if any(force_cols) else None
#         )  # assign force_cols for box or tsv
#         force_boxsizes = np.atleast_1d(cfg["picker_boxsize_overrides"]).tolist()
#         if any(force_boxsizes) and force_boxsizes[picker_index] is not None:
#             boxsize = float(force_boxsizes[picker_index])

#     # switch by extension and get parsed coordinates from appropriate parser function
#     if ext in [".star"]:
#         boxes = _parse_star(path, boxsize, conf_col=cfg["star_conf_column_name"])

#     elif ext in [".cbox", ".box"]:
#         if force_cols is None:
#             boxes = _parse_tsv(path, cols=(0, 1, 2, 3, 4), xy_are_center=False)
#         else:
#             boxes = _parse_tsv(
#                 path, cols=force_cols, boxsize=boxsize, xy_are_center=False
#             )

#     else:
#         print("Assuming TSV-like format for file at %s" % path)
#         if force_cols is None:
#             boxes = _parse_tsv(
#                 path, cols=(0, 1, None, None, 2), boxsize=boxsize, xy_are_center=True
#             )
#         else:
#             boxes = _parse_tsv(
#                 path, cols=force_cols, boxsize=boxsize, xy_are_center=True
#             )

#     norm_boxes = _normalize_confidences(boxes)

#     return norm_boxes


# # Parsing Functions


# def _parse_star(
#     path,
#     boxsize,
#     conf_col="_rlnAutopickFigureOfMerit",
#     single_file=False,
#     mrc_col="_rlnMicrographName",
# ):
#     """
#     Generate a list of Box namedtuples from the STAR file at the specified path.

#     Parameters
#     ----------
#     path : str
#         Path of input STAR file.
#     boxsize : float
#         Default box size to use when creating Box objects.
#     conf_col : str
#         Name in STAR header of column representing pick confidence values.
#     single_file : bool
#         If True, and path is a multi-mrc STAR file, return dictionary with mrc names (or paths, depending on mrc_col)
#         as keys and box lists as values.
#     mrc_col : str
#         Name in STAR header of column representing micrograph names or paths to separate boxes by, if split_mrc.

#     Returns
#     -------
#     boxes : list or dict
#         Dictionary of micrograph names and box lists if split_mrc; otherwise, list of Box namedtuples.
#     """

#     df = _star_to_df(path)

#     if df.empty:
#         return []

#     w = (
#         h
#     ) = boxsize  # using square boxes here; can also pass both w and h as parameters, read from config, etc.
#     x_name = "_rlnCoordinateX"
#     y_name = "_rlnCoordinateY"

#     # convert x, y, and confidence columns (if they exist) to numerical values
#     for col in [x_name, y_name, conf_col]:
#         if col in df:
#             df[col] = pd.to_numeric(df[col])

#     # turn a dataframe with at least cols x_name and y_name (and optionally conf_col) into a box list
#     def make_boxes(star_df):
#         if conf_col in df:
#             zipped = zip(star_df[x_name], star_df[y_name], star_df[conf_col])
#             boxes = [Box(x - w / 2, y - h / 2, w, h, c) for x, y, c in zipped]
#         else:
#             zipped = zip(star_df[x_name], star_df[y_name])
#             boxes = [Box(x - w / 2, y - h / 2, w, h) for x, y in zipped]
#         return boxes

#     # if we can, return dictionary of box lists, keyed to respective micrograph name
#     if single_file:
#         if mrc_col in df:
#             boxes_dict = {}
#             mrc_names = df[mrc_col].unique()
#             for mrc in mrc_names:
#                 boxes_dict[mrc] = make_boxes(df[df[mrc_col] == mrc])
#             return boxes_dict
#         else:
#             print("mrc_col '%s' not found in STAR file at %s" % (mrc_col, path))

#     return make_boxes(df)


# def _parse_tsv(path, cols=(0, 1, 2, 3, 4), boxsize=None, xy_are_center=False):
#     """
#     Generate a list of Box namedtuples from the TSV-like file at the specified path, applying column index
#     overrides if necessary. Default parameters represent .box or .cbox conventions.

#     Parameters
#     ----------
#     path : str
#         Path of input .box file.
#     cols : tuple
#         Iterable of five indices representing zero-indexed positions of x, y, w, h, and confidence columns in .box file.
#     boxsize : float or None
#         Default box size to use when creating Box objects; must be specified if cols[2] or cols[3] are None,
#         but is ignored otherwise.
#     xy_are_center : bool
#         If True, shift coordinates in the -x and -y direction by half the boxsize, in order that they represent
#         the top-left corner of the detection box (and x + w, y + h represents the bottom-right corner).

#     Returns
#     -------
#     boxes : list
#         List of Box namedtuples.
#     """

#     header_line_count = 0  # file line index where data starts

#     with open(path, mode="r") as f:
#         for i, line in enumerate(f):
#             if ut.has_numbers(line):  # skip blank lines or header (text-only) lines
#                 header_line_count = i
#                 break

#     # build header, assuming confidence column exists at given index for now (we'll check before returning boxes)
#     header = dict(zip(Box._fields, cols))
#     header = {
#         k: v for k, v in header.items() if v is not None
#     }  # remove any columns with indices of None

#     df = pd.read_csv(
#         path,
#         delim_whitespace=True,
#         header=None,
#         skip_blank_lines=True,
#         skiprows=header_line_count,
#     )
#     df = df.rename(
#         columns={df.columns[v]: k for k, v in header.items() if v in df.columns}
#     )  # rename columns

#     if df.empty:
#         return []

#     # convert x, y, and confidence columns (if they exist) to numerical values
#     for col in Box._fields:
#         if col in df:
#             df[col] = pd.to_numeric(df[col])

#     # set height and width columns if needed
#     for dim in ["w", "h"]:
#         if (
#             boxsize is None and dim not in df
#         ):  # if w and h columns are not available, boxsize preset must be specified
#             print("Cannot parse file at %s without boxsize preset" % path)
#             sys.exit(1)

#         if dim not in df:
#             df[dim] = [boxsize] * len(
#                 df.index
#             )  # make new column for w and/or h if needed

#     # shift coordinates towards -x, -y by half the boxsize if needed
#     if xy_are_center:
#         df["x"] = df["x"] - df["w"].div(2)
#         df["y"] = df["y"] - df["h"].div(2)

#     # make list of boxes, with confidences if available
#     if "conf" in df:
#         boxes = [
#             Box(x, y, w, h, c)
#             for x, y, w, h, c in zip(*(df[col_name] for col_name in Box._fields))
#         ]
#     else:
#         boxes = [
#             Box(x, y, w, h)
#             for x, y, w, h in zip(*(df[col_name] for col_name in ["x", "y", "w", "h"]))
#         ]

#     return boxes


# Writing Functions


# def write_star(
#     path,
#     boxes,
#     single_file=False,
# ):
#     """
#     Write list of boxes to star file at given path.

#     Parameters
#     ----------
#     path : str
#         Path to write STAR file to; if boxes is a dict and single_file is False, should be a dir (into which to write
#         each micrograph's box file).
#     boxes : list or dict
#         Either a dict with micrograph names as keys and box lists as values, or a list of Box namedtuples,
#         depending on the value of single_file.
#     conf_col : str
#         Name of column used for confidence values.
#     single_file : bool
#         If True, write boxes dict into a single file, adding a column (mrc_col) for each box's associated
#         dict key. If False, boxes is expected to be either a list of Box namedtuples or a dict (which will
#         be flattened to a list).
#     mrc_col : str
#         Name of column used for micrograph name or path, used if single_file is True.

#     Returns
#     -------
#     None
#     """

#     if single_file and not isinstance(
#         boxes, dict
#     ):  # only dict can go to multiple output files
#         print("write_star requires that boxes be a dict to use single_file")
#         return

#     path = ut.expand_path(path)
#     star_loop = "data_\n\nloop_\n"
#     header = [
#         "_rlnCoordinateX #1",
#         "_rlnCoordinateY #2",
#         conf_col + " #3",
#         mrc_col + " #4",
#     ]

#     def df_to_tsv(box_df, tsv_path):
#         box_df["x"] = box_df["x"] + box_df["w"].div(2)
#         box_df["y"] = box_df["y"] + box_df["h"].div(2)
#         box_df = box_df.drop(columns=["w", "h"])
#         box_df.to_csv(tsv_path, mode="a", sep=TSV_SEP, header=False, index=False)

#     if single_file:  # write boxes dict or list to one file
#         if os.path.isdir(path):
#             print(
#                 "write_star requires that path NOT be a directory if writing coordinates to one file"
#             )
#             return

#         if isinstance(boxes, dict):
#             for k, v in boxes.items():
#                 boxes[k] = [box + tuple([k]) for box in v]
#             boxes = list(ut.flatten(list(boxes.values())))
#             header_str = (
#                 star_loop + "\n".join(header) + "\n"
#             )  # keep mrc_col if boxes is a dict
#             df = pd.DataFrame(boxes, columns=Box._fields + ["mrc"])
#         else:
#             header_str = (
#                 star_loop + "\n".join(header[:-1]) + "\n"
#             )  # don't need mrc_col if boxes is a list
#             df = pd.DataFrame(boxes, columns=Box._fields)

#         with open(path, mode="w") as f:
#             f.write(header_str)
#         df_to_tsv(df, path)

#     else:  # write boxes dict to multiple files, one per mrc
#         if not os.path.isdir(path):
#             print(
#                 "write_star requires path to be a directory if writing coordinates to multiple files"
#             )
#             return

#         for k, v in boxes.items():
#             new_file = os.path.join(path, ut.basename(k, mode="name") + ".star")
#             with open(new_file, mode="w") as f:
#                 f.write(
#                     star_loop + "\n".join(header[:-1]) + "\n"
#                 )  # don't need mrc_col if not single_file
#             df = pd.DataFrame(v, columns=Box._fields)
#             df_to_tsv(df, new_file)


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
