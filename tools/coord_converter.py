import argparse
import os
import sys
import re
import glob
import csv
from datetime import datetime
import pandas as pd
from collections import OrderedDict

# This script converts coordinate file data between several different formats
# The following file descriptions are supported (column labels in <angle brackets> are optional):
#
#  - coord:     one tsv per mrc
#               optional non-numeric header (ignored)
#               ordered numeric columns [center_x, center_y, <confidence_score>]
#  - star:      one tsv per mrc
#               RELION loop header
#               unordered numeric columns [rlnCoordinateX, rlnCoordinateY, <FigureOfMerit>]
#  - box:       one tsv per mrc
#               optional non-numeric header (ignored)
#               ordered numeric columns [corner_x, corner_y, width, height, <confidence_score>]
#
# Single-file modes are available for each of these formats (in which the first/designated column
# contains the micrograph image name to which the coordinate on each row refers)

class Converter:
    def __init__(self, input_path, output_path, from_format, to_format, use_single_file_input, use_single_file_output,
                 output_suffix, can_overwrite):
        self.from_format = from_format
        self.to_format = to_format
        self.use_single_file_input = use_single_file_input
        self.use_single_file_output = use_single_file_output
        self.output_suffix = output_suffix
        self.can_overwrite = can_overwrite

        self.input_paths = self.expand_path(input_path, mode='input')
        self.output_dir = self.expand_path(output_path, mode='output')

        self.input_data = []
        self.output_paths = []

    def expand_path(self, path, mode):
        expanded_path = os.path.expandvars(os.path.expanduser(path))
        if mode == 'input':
            expanded_path = glob.glob(expanded_path)
            if self.use_single_file_input and len(expanded_path) > 1:
                print("ERROR: Input should be one file if using single-file mode.")
                sys.exit(1)
        elif mode == 'output':
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

    def parse_input(self):
        if self.from_format is None or self.to_format is None:
            print("Please specify both an input (-f) and output (-t) file format.")
            sys.exit(1)

        if self.from_format in ['star']:
            self.parse_star()
        elif self.from_format in ['box', 'cbox']:
            self.parse_box()
        elif self.from_format in ['coord']:
            self.parse_coord()
        else:
            print("ERROR: Input file format '%s' not recognized." % self.from_format)
            sys.exit(1)

    # file type parsing
    # note: for consistency, self.input_data['x'] and ['y'] will always refer to the particle center

    def parse_star(self):
        for file in self.input_paths:
            header = {}
            data = {'path': file, 'x': [], 'y': [], 'conf': []}
            if self.use_single_file_input:
                data['mrc_name'] = []
            with open(file, mode='r') as f:
                for line in f:
                    ln = line.split()
                    if line.startswith('_') and line.count('#') == 1:
                        split_line = ''.join(ln).split('#')
                        header[split_line[0]] = split_line[1]
                    elif re.search('[0-9.]', line) is not None and not line.isspace() and not line.startswith('_'):
                        try:
                            x_index = int(header['_rlnCoordinateX'])
                            data['x'].append(float(ln[x_index - 1]))
                            y_index = int(header['_rlnCoordinateY'])
                            data['y'].append(float(ln[y_index - 1]))
                        except (KeyError, IndexError) as e:
                            print("WARN: Could not parse STAR header values (%s).\n%s" % (e, header))
                            continue
                        try:
                            fom_index = int(header["_rlnAutopickFigureOfMerit"])
                            data['conf'].append(float(ln[fom_index - 1]))
                        except (KeyError, IndexError):
                            pass
                        if self.use_single_file_input:
                            try:
                                name_index = int(header["_rlnMicrographName"])
                                data['mrc_name'].append(str(ln[name_index - 1]))
                            except (KeyError, IndexError):
                                print("ERROR: Could not find micrograph name column for single-file input.")
                                sys.exit(1)

            self.input_data.append(data)

    def parse_box(self):
        for file in self.input_paths:
            data = {'path': file, 'x': [], 'y': [], 'w': [], 'h': [], 'conf': []}
            if self.use_single_file_input:
                data['mrc_name'] = []
            with open(file, mode='r') as f:
                for line in f:
                    ln = line.split()
                    if re.search('[0-9.]', line) is not None and not line.isspace():
                        index_shift = 0
                        if self.use_single_file_input:
                            try:
                                data['mrc_name'].append(str(ln[0]))
                                index_shift = 1
                            except IndexError:
                                print("ERROR: Could not find micrograph name column for single-file input.")
                                sys.exit(1)
                        try:
                            w = float(ln[2 + index_shift])
                            h = float(ln[3 + index_shift])
                            data['x'].append(float(ln[0 + index_shift]) - w / 2)
                            data['y'].append(float(ln[1 + index_shift]) - h / 2)
                            data['w'].append(w)
                            data['h'].append(h)
                        except IndexError:
                            print("ERROR: Unexpected number of columns in %s." % file)
                            sys.exit(1)
                        try:
                            data['conf'].append(float(ln[4 + index_shift]))
                        except IndexError:
                            pass

            self.input_data.append(data)

    def parse_coord(self):
        for file in self.input_paths:
            data = {'path': file, 'x': [], 'y': [], 'conf': []}
            if self.use_single_file_input:
                data['mrc_name'] = []
            with open(file, mode='r') as f:
                for line in f:
                    ln = line.split()
                    if re.search('[0-9.]', line) is not None and not line.isspace():
                        index_shift = 0
                        if self.use_single_file_input:
                            try:
                                data['mrc_name'].append(str(ln[0]))
                                index_shift = 1
                            except IndexError:
                                print("ERROR: Could not find micrograph name column for single-file input.")
                                sys.exit(1)
                        try:
                            data['x'].append(float(ln[0 + index_shift]))
                            data['y'].append(float(ln[1 + index_shift]))
                        except IndexError:
                            print("WARN: Unexpected number of columns in %s." % file)
                            continue
                        try:
                            data['conf'].append(float(ln[2 + index_shift]))
                        except IndexError:
                            pass
            self.input_data.append(data)

    # conversions

    def convert_input(self):
        suffix = '%s.%s' % (self.output_suffix, self.to_format)
        time_str = str(datetime.strftime(datetime.now(), '%y%m%d_%H%M%S'))
        # output_files are the filenames where generated coordinates will be written
        # output_names are the image names to use in the generated files if self.use_single_file_output was requested
        if self.use_single_file_output and self.use_single_file_input:
            output_files = ['%s%s' % (time_str, suffix)]
            output_names = [os.path.splitext(os.path.basename(x))[0] + suffix for x in self.input_data[0]['mrc_name']]
        elif self.use_single_file_output and not self.use_single_file_input:
            output_files = ['%s%s' % (time_str, suffix)]
            output_names = [os.path.splitext(os.path.basename(x['path']))[0] + '.mrc' for x in self.input_data]
            print("WARN: Single-file output was requested without single-file input. Will attempt to guess "
                  "micrograph image names from the input coordinate file names.")
        elif self.use_single_file_input and not self.use_single_file_output:
            output_files = [os.path.splitext(os.path.basename(x))[0] + suffix for x in self.input_data[0]['mrc_name']]
            output_names = []
        else:  # i.e. not self.use_single_file_input and not self.use_single_file_output
            output_files = [os.path.splitext(os.path.basename(x['path']))[0] + suffix for x in self.input_data]
            output_names = []
        self.output_paths = [os.path.join(self.output_dir, x) for x in output_files]
        if any([os.path.isfile(x) for x in self.output_paths]) and not self.can_overwrite:
            print("ERROR: Some files in the specified output directory will be overwritten. "
                  "To confirm, run coord_converter.py again with the --overwrite flag.")
            sys.exit(1)

        # if self.use_single_file_input, we need a conversion layer from original large dict to list of one dict per mrc
        if self.use_single_file_input:
            # with use_single_file_input, there will be duplicate output_paths; remove them while preserving order
            self.output_paths = list(OrderedDict.fromkeys(self.output_paths))
            input_path = self.input_data[0]['path']
            df = pd.DataFrame(*self.input_data)
            grouped = df.groupby('mrc_name')
            if 'conf' in df:
                self.input_data = [{'path': input_path, 'x': group['x'].tolist(), 'y': group['y'].tolist()}
                                   for name, group in grouped]
            else:
                self.input_data = [{'path': input_path, 'x': group['x'].tolist(), 'y': group['y'].tolist(),
                                    'conf': group['conf'].tolist()} for name, group in grouped]

        if self.to_format in ['star']:
            self.convert_to_star(output_names)
        elif self.to_format in ['box']:
            self.convert_to_box(output_names, include_confidence=False)
        elif self.to_format in ['cbox']:
            self.convert_to_box(output_names, include_confidence=True)
        elif self.to_format in ['coord']:
            self.convert_to_coord(output_names)

        print("Done. Output located at: %s" % self.output_dir)

    def convert_to_star(self, output_names):
        for i, file in enumerate(self.output_paths):
            try:
                x = self.input_data[i]['x']
                y = self.input_data[i]['y']
                rows = zip(*self.even_jagged_arr(x, y))
            except KeyError as e:
                print("ERROR: Required fields not present in input (%s format).\n%s" % (self.from_format, e))
                sys.exit(1)
            if self.use_single_file_output:  # in this case output_names will not be []
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                header_str = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n'
                if self.use_single_file_output:
                    header_str = header_str + '_rlnMicrographName #3\n'
                f.write(header_str)
                writer = csv.writer(f, delimiter='\t')
                for row in rows:
                    writer.writerow(row)

    def convert_to_box(self, output_names, include_confidence):
        for i, file in enumerate(self.output_paths):
            try:
                x = self.input_data[i]['x']
                y = self.input_data[i]['y']
                rows = zip(*self.even_jagged_arr(x, y))
                if include_confidence:
                    conf = self.input_data[i]['conf']
                    rows = zip(*self.even_jagged_arr(*zip(*rows), conf))
            except KeyError as e:
                print("ERROR: Required fields not present in input (%s format).\n%s" % (self.from_format, e))
                sys.exit(1)
            if self.use_single_file_output:
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                writer = csv.writer(f, delimiter='\t')
                for row in rows:
                    writer.writerow(row)

    def convert_to_coord(self, output_names):
        for i, file in enumerate(self.output_paths):
            try:
                x = self.input_data[i]['x']
                y = self.input_data[i]['y']
                rows = zip(*self.even_jagged_arr(x, y))
                if self.input_data[i]['conf']:
                    conf = self.input_data[i]['conf']
                    rows = zip(*self.even_jagged_arr(*zip(*rows), conf))
            except KeyError as e:
                print("ERROR: Required fields not present in input (%s format).\n%s" % (self.from_format, e))
                sys.exit(1)
            if self.use_single_file_output:
                rows = zip(*self.even_jagged_arr(output_names, *zip(*rows)))
            with open(file, "w") as f:
                writer = csv.writer(f, delimiter='\t')
                for row in rows:
                    writer.writerow(row)

    # utility

    def even_jagged_arr(self, *args):
        if len(set([len(x) for x in args])) > 1:
            print("WARN: Found unequal columns in input data. Padding with zeroes as necessary.")
            args = self.pad_list_of_lists(args)
        return args

    @staticmethod
    def pad_list_of_lists(arr, pad_with=0):
        max_list_len = max(len(x) for x in arr)
        units_to_pad = [max_list_len - len(x) for x in arr]
        for i in range(len(arr)):
            arr[i].extend([pad_with] * units_to_pad[i])
        return arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script converts particle coordinate file data between several "
                                                 "different formats.")
    parser.add_argument('input', help="Input particle coordinate file(s) (pass multiple files with globbing patterns; "
                                      "be sure to enclose path in quotes)")
    parser.add_argument('output_dir', help="Output directory in which to store generated coordinate files (will be "
                                           "created if it does not exist)")
    parser.add_argument('-f', choices=['star', 'box', 'cbox', 'coord'], help="Format FROM which to convert the input")
    parser.add_argument('-t', choices=['star', 'box', 'cbox', 'coord'], help="Format TO which to convert the input")
    parser.add_argument('-s', dest='suffix', default='converted', help="Suffix to append to generated output, "
                                                                       "e.g. originalfilename_mysuffix.star (default: "
                                                                       "converted)")
    parser.add_argument('--single_in', action='store_true', help="Input is of a 'single-file' format (coordinates "
                                                                 "for multiple micrographs are located in the same "
                                                                 "tsv, distinguished by a micrograph name column")
    parser.add_argument('--single_out', action='store_true', help="Output should be a single file (see above)")
    parser.add_argument('--overwrite', action='store_true', help="Allow files in output directory to be overwritten")

    a = parser.parse_args()

    converter = Converter(a.input, a.output_dir, a.f, a.t, a.single_in, a.single_out, a.suffix, a.overwrite)
    converter.parse_input()
    converter.convert_input()
