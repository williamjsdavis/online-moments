#!/usr/bin/env bash
# Run all validation drivers at smoke (CI) sizes. Generates PNGs in
# validation/figures/.
set -euo pipefail
cd "$(dirname "$0")"
python figure_1_ou.py --smoke
python figure_2_tristable.py --smoke
python figure_table_1_scaling.py --smoke
echo
echo "Done. Inspect:"
echo "  validation/figures/figure_1_python.png"
echo "  validation/figures/figure_2_python.png"
echo "  validation/figures/scaling.png"
