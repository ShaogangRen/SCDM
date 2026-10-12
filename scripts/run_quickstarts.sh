#!/usr/bin/env bash
# Run bundled quick-start examples end to end.
#
#   bash scripts/run_quickstarts.sh
#
# Optional environment variables:
#   DEVICE  -- auto (default), cpu, cuda, cuda:0
#   PYTHON  -- python interpreter to use (default: python)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"

echo "=============================================================="
echo "Example 1: NC8 quick start (8 variables)"
echo "=============================================================="
"$PYTHON" -m scdm.nonlatent.cli --config configs/nonlatent/nc8_quick.yaml --device "$DEVICE"

echo
echo "=============================================================="
echo "Example 2: high-dimensional quick start (64 variables)"
echo "=============================================================="
"$PYTHON" -m scdm.nonlatent.cli --config configs/nonlatent/highdim_quick.yaml --device "$DEVICE"

echo
echo "=============================================================="
echo "Example 3: latent Appendix C diagnostic"
echo "=============================================================="
"$PYTHON" -m scdm.latent.cli --config configs/latent/nc8_appendix_c_quick.yaml --device "$DEVICE"

echo
echo "=============================================================="
echo "All quick starts completed."
echo "  results/nc8_quickstart/summary.txt"
echo "  results/highdim_quickstart/summary.txt"
echo "  results/latent_nc8_appendix_c_quick/summary.txt"
echo "=============================================================="
