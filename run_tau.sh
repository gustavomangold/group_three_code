#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for quarter in {0..12}; do
    printf -v tau '%d.%02d' "$((quarter / 4))" "$((quarter % 4 * 25))"
    echo "Running tau=$tau"
    MPLBACKEND=Agg python3 "$script_dir/two_disks.py" --PERSISTENCE_TIME "$tau" "$@" --L 4
done
