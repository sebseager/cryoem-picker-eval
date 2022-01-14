#!/bin/bash

display_help() {
  echo "Syntax: tabs-to-spaces.sh <files_to_convert>"
  echo
  exit 1
}

# exit if we didn't get arguments
if [ $# -eq 0 ]; then
  display_help
fi

for file in "$@"; do
  if [ -f "$file" ]; then
    expand -t 4 "$file" > "$file".tmp && mv -v "$file".tmp "$file"
  fi
done
