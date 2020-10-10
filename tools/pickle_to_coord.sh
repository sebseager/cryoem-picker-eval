#!/bin/sh

# DeepPicker outputs a pickle file of the format: x_coord, y_coord, score, image_name
# This script extracts the first three of these columns and sends them to a different file for each image_name

display_help() {
  echo "usage: $(basename "$0") [-h] <pickle_file>"
  echo "options:"
  echo "  -h   Show this message and exit."
  echo
  exit 1
}

# exit if we didn't get one argument
if [ $# -ne 1 ] || [ "$1" = "-h" ]; then
  display_help
fi

read -rp "This script will build *_unpickled.coord files in the directory containing the input pickle file. It is therefore recommended that this directory contain no files other than $1. Type 'y' to go ahead: " go_ahead

if [ "$go_ahead" != 'Y' ] || [ "$go_ahead" != 'y' ]; then
  echo "Exiting."
  exit 1
fi

pickle_file=$1 python - << END

import sys
import os
import csv
import pickle

print("NOTE: if you get a UnicodeDecodeError and have an active Anaconda environment, run conda deactivate and try again.")

def unpickle(filename):
    file = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data

pickle_path_full = os.path.expanduser(os.environ['pickle_file'])

all_picks = unpickle(pickle_path_full)
out_dict = {j[3]: [] for i in all_picks for j in i}

for i in all_picks:
    for j in i:
      out_dict[j[3]].append([j[0], j[1], j[2]])

pickle_dir = os.path.dirname(pickle_path_full)

for x in out_dict:
    out_file = os.path.join(pickle_dir, os.path.basename(os.path.splitext(x)[0]) + "_unpickled.coord")
    with open(out_file, 'w') as f:
        tsv_output = csv.writer(f, delimiter='\t')
        for ln in out_dict[x]:
            tsv_output.writerow(ln)

END
