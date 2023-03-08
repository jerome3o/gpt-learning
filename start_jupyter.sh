#!/usr/bin/env bash
# Variables
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

(
  cd "$script_dir" || exit
  # activate venv
  source venv/bin/activate

  # run script
  jupyter lab
)

