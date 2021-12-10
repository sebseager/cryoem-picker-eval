#!/bin/bash

display_help() {
  echo "Syntax: patch-aspire.sh <path/to/ASPIRE_clone>"
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

apple_dir="$1/src/aspire/apple"

# exit if /src/aspire/apple isn't an existing directory
if [ ! -d "$apple_dir" ]; then
  echo "Bad ASPIRE-Python clone. Could not find $apple_dir"
  display_help
fi

self_dir=$(dirname "$0")
cp -v "$self_dir"/{apple.py,helper.py,picking.py} "$apple_dir"
