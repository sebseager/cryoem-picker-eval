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

self_dir=$(dirname "$0")
cp -v "$self_dir"/{autoPick.py,dataLoader.py,deepModel.py,starReader.py,train.py} "$1"
