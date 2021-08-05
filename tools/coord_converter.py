import argparse
import os
import sys
import re
import glob
from datetime import datetime
import csv
import pandas as pd
import pickle
from collections import OrderedDict

# This script converts coordinate file data between several different formats
# The following file descriptions are supported (column labels in <angle brackets> are optional):
#
#  - tsv|coord: one tsv per mrc
#               optional non-numeric header (ignored)
#               ordered numeric columns [center_x, center_y, <confidence_score>]
#  - star:      one tsv per mrc
#               RELION loop header
#               unordered numeric columns [rlnCoordinateX, rlnCoordinateY, <FigureOfMerit>]
#  - box|cbox:  one tsv per mrc
#               optional non-numeric header (ignored)
#               ordered numeric columns [corner_x, corner_y, width, height, <confidence_score>]
#
# Single-file modes are available for each of these formats (in which the first/designated column
# contains the micrograph image name to which the coordinate on each row refers)


class Converter:
    def __init__(
        self,
        input_files,
        from_format=None,
        decode=None,
        output_path=None,
        to_format=None,
        boxsize=None,
        cols=None,
        single_in_file=False,
        single_out_file=False,
        output_suffix="",
        can_overwrite=False,
    ):
        self.from_format = from_format
        self.decode_format = decode
        self.to_format = to_format
        self.manual_boxsize = int(boxsize)
        self.col_overrides = cols
        self.use_single_file_input = single_in_file
        self.use_single_file_output = single_out_file
        self.output_suffix = output_suffix
        self.can_overwrite = can_overwrite

        self.time_str = str(datetime.strftime(datetime.now(), "%y%m%d_%H%M%S"))

        self.input_paths = self.expand_path(input_files, mode="input")
        self.output_dir = (
            None
            if output_path is None
            else self.expand_path(output_path, mode="output")
        )

        self.tmp_paths_to_delete = []
        self.input_data = []
        self.output_paths = []

        # start parsing
        self.detect_input_format()
        self.indices = self.get_input_indices()
        self.parse_input()

    # setup

    def get_input_indices(self):
        if self.use_single_file_input:
            mrc_name_idx = 0
            idx_shift = 1
        else:
            mrc_name_idx = None
            idx_shift = 0
        if self.from_format in ["star"]:
            indices = {"x": None, "y": None, "conf": None, "mrc_name": None}
        elif self.from_format in ["box", "cbox"]:
            indices = {
                "x": 0 + idx_shift,
                "y": 1 + idx_shift,
                "w": 2 + idx_shift,
                "h": 3 + idx_shift,
                "conf": 4 + idx_shift,
                "mrc_name": mrc_name_idx,
            }
        elif self.from_format in ["tsv", "coord"]:
            indices = {
                "x": 0 + idx_shift,
                "y": 1 + idx_shift,
                "conf": 2 + idx_shift,
                "mrc_name": mrc_name_idx,
            }
        else:
            print("ERROR: Invalid input format (-f): %s" % self.from_format)
            sys.exit(1)

        if self.col_overrides is None:
            return indices
        if not all([self.isint(n) or n == "auto" for n in self.col_overrides]):
            print(
                "ERROR: Input argument -c requires four parameters, each of which can either be positive/negative "
                "integers indices or 'auto'."
            )
            sys.exit(1)

        self.col_overrides = [str(n).lower() for n in self.col_overrides]
        user_indicies = {
            "x": self.col_overrides[0],
            "y": self.col_overrides[1],
            "w": self.col_overrides[2],
            "h": self.col_overrides[3],
            "conf": self.col_overrides[4],
            "mrc_name": self.col_overrides[5],
        }
        for key in indices:
            if user_indicies[key] != "auto":
                indices[key] = int(user_indicies[key])
        return indices

    def expand_path(self, path, mode):
        expanded_path = os.path.expandvars(os.path.expanduser(path))
        if mode == "input":
            expanded_path = glob.glob(expanded_path)
            if self.use_single_file_input and len(expanded_path) > 1:
                print("ERROR: Input should be one file if using single-file mode.")
                sys.exit(1)
        elif mode == "output":
            os.makedirs(expanded_path, exist_ok=True)
            # if not os.path.isdir(expanded_path):
            #     print("ERROR: Output path should be a directory.")
            #     sys.exit(1)
        return expanded_path

    # def read_input(self):
    #     input_data = []
    #     for file in self.input_paths:
    #         input_data.append(
    #             pd.read_csv(file, delim_whitespace=True, skipinitialspace=True, skip_blank_lines=True, header=None))
    #     if not input_data:
    #         print("ERROR: Invalid input path(s)")
    #         sys.exit(1)
    #     return input_data

    def detect_input_format(
        self,
    ):  # TODO: this happens ONCE at beginning of run; what if some files are of different from_format?
        if self.from_format is None:
            try:
                ext = os.path.splitext(self.input_paths[0])[1]
            except IndexError:
                print("Could not find any input paths. Exiting.")
                sys.exit(1)
            print("Detecting coordinate file format from extension: %s" % ext)
            self.from_format = str(ext).replace(".", "").lower()

    def parse_input(self):
        if self.decode_format in ["pickle"]:
            self.decode_pickle()

        if self.from_format in ["star"]:
            self.parse_star()
        elif self.from_format in ["box", "cbox"]:
            self.parse_box()
        elif self.from_format in ["tsv", "coord"]:
            self.parse_tsv()
        else:
            print("Please specify an input (from) file format (-f).")
            sys.exit(1)

        self.delete_tmp_files()

    def delete_tmp_files(self):
        for file in self.tmp_paths_to_delete:
            os.remove(file)

    # file type parsing
    # note: for consistency, self.input_data['x'] and ['y'] will always refer to the particle center after import

    def decode_pickle(self):
        # def flatten_to_2d(arr):
        #     if all(isinstance(n, list) for n in arr):
        #         if len(arr) > 1:
        #             return flatten_to_2d(arr[0]) + flatten_to_2d(arr[1:len(arr)])
        #         else:
        #             return flatten_to_2d(arr[0])
        #     else:
        #         return [arr]

        def unpack_innermost_lists(arr):
            if arr and not all(isinstance(n, list) for n in arr):
                yield list(arr)
            else:
                for x in arr:
                    yield from unpack_innermost_lists(x)

        tmp_input_paths = []

        for file in self.input_paths:
            with open(file, "rb") as f:
                raw = pickle.load(f, encoding="latin1")
            if not (len(raw) > 0 and len(raw[0]) > 0 and len(raw[0][0]) > 0):
                print("ERROR: Empty pickle file. Exiting.")
                sys.exit(1)
            raw_2d = unpack_innermost_lists(raw)
            tmp_file = "%s.%s.tmp" % (file, self.time_str)
            with open(tmp_file, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(raw_2d)
            tmp_input_paths.append(tmp_file)

            # raw_2dT = np.array(raw_2d).transpose()
            # if len(raw_2dT) >= 2 and str_is_dec(raw_2dT[0][0], raw_2dT[1][0]):
            #     if re.search('[A-Za-z/]', raw_2dT[-1][0]) is not None:
            #         if len(raw_2dT) >= 3 and str_is_dec(raw_2dT[2][0]):
            #             data = {'path': file, 'x': raw_2dT[0], 'y': raw_2dT[1], 'conf': raw_2dT[2], 'mrc_name': raw_2dT[-1]}
            #         else:
            #             data = {'path': file, 'x': raw_2dT[0], 'y': raw_2dT[1], 'mrc_name': raw_2dT[-1]}
            #     else:
            #         if len(raw_2dT[0]) >= 3 and str_is_dec(raw_2dT[0][2]):
            #             data = {'path': file, 'x': raw_2dT[0], 'y': raw_2dT[1], 'conf': raw_2dT[2]}
            #         else:
            #             data = {'path': file, 'x': raw_2dT[0], 'y': raw_2dT[1]}
            # else:
            #     print("ERROR: Cannot understand unpickled list format. Exiting.")
            #     sys.exit(1)
            #
            # self.input_data.append(data)
        self.input_paths = tmp_input_paths
        self.tmp_paths_to_delete = tmp_input_paths

    def parse_star(self):
        for file in self.input_paths:
            header = {}
            data = {"path": file, "x": [], "y": [], "conf": []}
            if self.use_single_file_input:
                data["mrc_name"] = []
            with open(file, mode="r") as f:
                for line in f:
                    ln = line.split()
                    if line.startswith("_") and line.count("#") == 1:
                        split_line = "".join(ln).split("#")
                        header[split_line[0]] = split_line[1]
                    elif (
                        self.hasnumbers(line)
                        and not line.isspace()
                        and not line.startswith("_")
                    ):
                        try:
                            x_index = (
                                int(header["_rlnCoordinateX"]) - 1
                                if self.indices["x"] is None
                                else self.indices["x"]
                            )
                            data["x"].append(float(ln[x_index]))
                            y_index = (
                                int(header["_rlnCoordinateY"]) - 1
                                if self.indices["y"] is None
                                else self.indices["y"]
                            )
                            data["y"].append(float(ln[y_index]))
                        except (KeyError, IndexError) as e:
                            print(
                                "WARN: Could not parse STAR header values (%s).\n%s"
                                % (e, header)
                            )
                            continue
                        try:
                            fom_index = (
                                int(header["_rlnAutopickFigureOfMerit"]) - 1
                                if self.indices["conf"] is None
                                else self.indices["conf"]
                            )
                            data["conf"].append(float(ln[fom_index]))
                        except (KeyError, IndexError):
                            pass
                        if self.use_single_file_input:
                            try:
                                name_index = (
                                    int(header["_rlnMicrographName"]) - 1
                                    if self.indices["mrc_name"] is None
                                    else self.indices["mrc_name"]
                                )
                                data["mrc_name"].append(str(ln[name_index]))
                            except (KeyError, IndexError):
                                print(
                                    "ERROR: Could not find micrograph name column for single-file input."
                                )
                                sys.exit(1)

            self.input_data.append(data)

    def parse_box(self):
        for file in self.input_paths:
            data = {"path": file, "x": [], "y": [], "w": [], "h": [], "conf": []}
            if self.use_single_file_input:
                data["mrc_name"] = []
            with open(file, mode="r") as f:
                for line in f:
                    ln = line.split()
                    if self.hasnumbers(line) and not line.isspace():
                        if self.use_single_file_input:
                            try:
                                data["mrc_name"].append(
                                    str(ln[self.indices["mrc_name"]])
                                )
                            except IndexError:
                                print(
                                    "ERROR: Could not find micrograph name column for single-file input."
                                )
                                sys.exit(1)
                        try:
                            w = float(ln[self.indices["w"]])
                            h = float(ln[self.indices["h"]])
                            data["x"].append(float(ln[self.indices["x"]]) + w / 2)
                            data["y"].append(float(ln[self.indices["y"]]) + h / 2)
                            data["w"].append(w)
                            data["h"].append(h)
                        except IndexError:
                            print("ERROR: Unexpected number of columns in %s." % file)
                            sys.exit(1)
                        try:
                            data["conf"].append(float(ln[self.indices["conf"]]))
                        except IndexError:
                            pass

            self.input_data.append(data)

    def parse_tsv(self):
        for file in self.input_paths:
            data = {"path": file, "x": [], "y": [], "conf": []}
            if self.use_single_file_input:
                data["mrc_name"] = []
            with open(file, mode="r") as f:
                for line in f:
                    ln = line.split()
                    if self.hasnumbers(line) and not line.isspace():
                        if self.use_single_file_input:
                            try:
                                data["mrc_name"].append(
                                    str(ln[self.indices["mrc_name"]])
                                )
                            except IndexError:
                                print(
                                    "ERROR: Could not find micrograph name column for single-file input."
                                )
                                sys.exit(1)
                        try:
                            data["x"].append(float(ln[self.indices["x"]]))
                            data["y"].append(float(ln[self.indices["y"]]))
                        except IndexError:
                            print("WARN: Unexpected number of columns in %s." % file)
                            continue
                        try:
                            data["conf"].append(float(ln[self.indices["conf"]]))
                        except IndexError:
                            pass
            self.input_data.append(data)

    # conversions

    def convert_input(self):
        if self.to_format is None or self.output_dir is None:
            print(
                "ERROR: Please specify an output (to) file format (-t) and an output directory."
            )
            sys.exit(1)

        # add manual boxsize if needed
        if self.to_format in ["box", "cbox"]:
            if self.manual_boxsize is None:
                if any(["w" not in d or "h" not in d for d in self.input_data]):
                    print(
                        "ERROR: Could not find a box size to use for conversion to '%s'. "
                        "Please specify one with -b" % self.to_format
                    )
                    sys.exit(1)
                else:
                    print("Using automatic box size from input")
            else:
                print("Using provided box size: %s" % str(self.manual_boxsize))
                for i, d in enumerate(self.input_data):
                    # essentially, [self.manual_boxsize] * n, where n is the length of the longest list in d
                    self.input_data[i]["w"] = self.input_data[i][
                        "h"
                    ] = self.pad_list_of_lists(
                        [[], d["x"], d["y"]], pad_with=self.manual_boxsize
                    )[
                        0
                    ]
                breakpoint()

        suffix = "%s.%s" % (self.output_suffix, self.to_format)
        # output_files are the filenames where generated coordinates will be written
        # output_names are the image names to use in the generated files if self.use_single_file_output was requested
        if self.use_single_file_output and self.use_single_file_input:
            output_files = ["%s%s" % (self.time_str, suffix)]
            output_names = [
                os.path.splitext(os.path.basename(x))[0] + suffix
                for x in self.input_data[0]["mrc_name"]
            ]
        elif self.use_single_file_output and not self.use_single_file_input:
            output_files = ["%s%s" % (self.time_str, suffix)]
            output_names = [
                os.path.splitext(os.path.basename(x["path"]))[0] + ".mrc"
                for x in self.input_data
            ]
            print(
                "WARN: Single-file output was requested without single-file input. Will attempt to guess "
                "micrograph image names from the input coordinate file names."
            )
        elif self.use_single_file_input and not self.use_single_file_output:
            output_files = [
                os.path.splitext(os.path.basename(x))[0] + suffix
                for x in self.input_data[0]["mrc_name"]
            ]
            output_names = []
        else:  # i.e. not self.use_single_file_input and not self.use_single_file_output
            output_files = [
                os.path.splitext(os.path.basename(x["path"]))[0] + suffix
                for x in self.input_data
            ]
            output_names = []
        self.output_paths = [os.path.join(self.output_dir, x) for x in output_files]
        if (
            any([os.path.isfile(x) for x in self.output_paths])
            and not self.can_overwrite
        ):
            print(
                "ERROR: Some files in the specified output directory will be overwritten. "
                "To confirm, run coord_converter.py again with the --overwrite flag."
            )
            sys.exit(1)

        # if self.use_single_file_input, we need a conversion layer from original large dict to list of one dict per mrc
        if self.use_single_file_input:
            # with use_single_file_input, there will be duplicate output_paths; remove them while preserving order
            self.output_paths = list(OrderedDict.fromkeys(self.output_paths))
            input_path = self.input_data[0]["path"]
            df = pd.DataFrame(*self.input_data)
            grouped = df.groupby("mrc_name")
            if "conf" in df:
                self.input_data = [
                    {
                        "path": input_path,
                        "x": group["x"].tolist(),
                        "y": group["y"].tolist(),
                        "conf": group["conf"].tolist(),
                    }
                    for name, group in grouped
                ]
            else:
                self.input_data = [
                    {
                        "path": input_path,
                        "x": group["x"].tolist(),
                        "y": group["y"].tolist(),
                    }
                    for name, group in grouped
                ]

        if self.to_format in ["star"]:
            self.convert_to_star(output_names)
        elif self.to_format in ["box", "cbox"]:
            self.convert_to_box(
                output_names, include_confidence=self.to_format in ["cbox"]
            )
        elif self.to_format in ["tsv", "coord"]:
            self.convert_to_tsv(output_names)

        print("Done. Output located at: %s" % self.output_dir)

    def convert_to_star(self, output_names):
        for i, file in enumerate(self.output_paths):
            try:
                d = self.input_data[i]
                rows = zip(*self.even_jagged_arr(d["x"], d["y"]))
            except KeyError as e:
                print(
                    "ERROR: Required fields not present in input (%s format).\n%s"
                    % (self.from_format, e)
                )
                sys.exit(1)
            if self.use_single_file_output:  # in this case output_names will not be []
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                header_str = "data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
                if self.use_single_file_output:
                    header_str = header_str + "_rlnMicrographName #3\n"
                f.write(header_str)
                writer = csv.writer(f, delimiter="\t")
                for row in rows:
                    writer.writerow(row)

    def convert_to_box(self, output_names, include_confidence):
        for i, file in enumerate(self.output_paths):
            try:
                d = self.input_data[i]
                x = d[
                    "x"
                ]  # we still need to subtract half the width here; do this after even_jagged_arr though
                y = d["y"]  # ditto for height
                w = d["w"]
                h = d["h"]
                rows = zip(*self.even_jagged_arr(x, y, w, h))
                rows = [
                    (r[0] - r[2] / 2, r[1] - r[3] / 2, r[2], r[3]) for r in rows
                ]  # indices follow x, y, w, h above
                if include_confidence:
                    conf = d["conf"]
                    rows = zip(*self.even_jagged_arr(*zip(*rows), conf))
            except KeyError as e:
                print(
                    "ERROR: Required fields not present in input (%s format).\n%s"
                    % (self.from_format, e)
                )
                sys.exit(1)
            if self.use_single_file_output:
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                for row in rows:
                    writer.writerow(row)

    def convert_to_tsv(self, output_names):
        for i, file in enumerate(self.output_paths):
            try:
                d = self.input_data[i]
                rows = zip(*self.even_jagged_arr(d["x"], d["y"]))
                if d["conf"]:
                    rows = zip(*self.even_jagged_arr(*zip(*rows), d["conf"]))
            except KeyError as e:
                print(
                    "ERROR: Required fields not present in input (%s format).\n%s"
                    % (self.from_format, e)
                )
                sys.exit(1)
            if self.use_single_file_output:
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                for row in rows:
                    writer.writerow(row)

    # utility

    def even_jagged_arr(self, *args):
        if len(set([len(x) for x in args])) > 1:
            print(
                "WARN: Found unequal columns in input data. Padding with zeroes as necessary."
            )
            args = self.pad_list_of_lists(args)
        return args

    @staticmethod
    def pad_list_of_lists(arr, pad_with=0):
        max_list_len = max(len(x) for x in arr)
        units_to_pad = [max_list_len - len(x) for x in arr]
        for i in range(len(arr)):
            arr[i].extend([pad_with] * units_to_pad[i])
        return arr

    @staticmethod
    def isint(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def isdec(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def hasnumbers(val):
        return re.search("[0-9]", val) is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts particle coordinate file data between several "
        "different formats. The -f (input format) and -t (output format) "
        "parameters define the conversion (e.g. whether coordinates refer "
        "to particle centers or box corners, whether file headers should be "
        "read/written, etc.). The -c argument can be used if more granular "
        "control over column indices is required."
    )
    parser.add_argument(
        "input",
        help="Path(s) to input particle coordinates; specify one or more files, not a "
        "directory (if using globbing patterns, be sure to enclose in quotes)",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory in which to store generated coordinate files (will be "
        "created if it does not exist)",
    )
    parser.add_argument(
        "-f",
        choices=["star", "box", "cbox", "tsv", "coord"],
        help="Format FROM which to convert the input",
    ),
    parser.add_argument(
        "-d",
        choices=["pickle"],
        help="Method to use to decode or deserialize input (i.e. if input is serialized rather than "
        "a tsv-like file). Default column indices for each file format are as follows."
        "TSV or COORD: [center_x, center_y, <confidence_score>], STAR: [rlnCoordinateX, "
        "rlnCoordinateY, <FigureOfMerit>], BOX or CBOX: [corner_x, corner_y, width, height, "
        "<confidence_score>]",
    ),
    parser.add_argument(
        "-t",
        choices=["star", "box", "cbox", "tsv", "coord"],
        help="Format TO which to convert the input",
    )
    parser.add_argument(
        "-b",
        type=int,
        help="If output format is BOX or CBOX, this argument either specifies "
        "or overrides the box size to be used (required if input does not "
        "include a box size)",
    )
    parser.add_argument(
        "-c",
        metavar=("X_COL", "Y_COL", "W_COL", "H_COL", "CONF_COL", "IMAGE_COL"),
        nargs=6,
        help="Manually set zero-based indices of x-coordinate, "
        "y-coordinate, width and height (if applicable), confidence, and image name columns. "
        "This can be useful if input file does not follow default column indices of the "
        "specified input (-f) format. Expects six values, either >= 0 (index from left), "
        "< 0 (index from right), or 'auto' for any unchanged or inapplicable parameters to "
        "keep default value.",
    )
    parser.add_argument(
        "-s",
        dest="suffix",
        default="_converted",
        type=str,
        help="Suffix to append to generated output, e.g. originalfilename_mysuffix.star (default "
        "if not specified: _converted)",
    )
    parser.add_argument(
        "--single_in",
        action="store_true",
        help="Input is of a 'single-file' format, where coordinates for multiple micrographs are "
        "located in the same tsv, distinguished by a micrograph name column",
    )
    parser.add_argument(
        "--single_out",
        action="store_true",
        help="Output should be a single file (see above)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow files in output directory to be overwritten",
    )

    a = parser.parse_args()
    converter = Converter(
        a.input,
        a.f,
        a.d,
        a.output_dir,
        a.t,
        a.b,
        a.c,
        a.single_in,
        a.single_out,
        a.suffix,
        a.overwrite,
    )
    converter.convert_input()
