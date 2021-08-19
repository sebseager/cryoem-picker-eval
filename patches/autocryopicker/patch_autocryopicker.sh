#!/bin/bash

display_help() {
  echo "Syntax: patch_autocryopicker.sh <path/to/AutoCryoPicker_clone>"
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

src_dir="$1/Signle\ Particle\ Detection_Demo/"

# exit if src_dir isn't an existing directory
if [ ! -d "$src_dir" ]; then
  echo "Bad AutoCryoPicker clone. Could not find $src_dir"
  display_help
fi

self_dir=$(dirname "$0")
cp -v "$self_dir"/AutoPicker_Final_Demo.m "$src_dir"
