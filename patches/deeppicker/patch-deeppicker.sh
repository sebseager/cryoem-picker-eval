#!/bin/bash

display_help() {
  echo "Syntax: patch-deeppicker.sh <path/to/DeepPicker_clone>"
  echo
  exit 1
}

# exit if we didn't get one argument
if [ $# -ne 1 ]; then
  display_help
fi

# exit if input isn't a directory
if [ ! -d "$1" ]; then
  echo "Not a directory: $1"
  display_help
fi

find "$1" -name '*.py' ! -type d -exec bash -c 'expand -t 4 "$0" > "$0".tmp && mv -v "$0".tmp "$0"' {} \;
