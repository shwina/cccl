#!/usr/bin/env bash
# Run the v1-vs-v2 suite at a realistic size against the prebuilt envs from
# setup_envs.sh, for both the fixed and unfixed v2, and generate a PDF report
# for each. v1 is the identical baseline in both runs.
#
# Env overrides (defaults shown):
#   ROOT     /tmp/cccl-bench         (where setup_envs.sh put v1/v2fix/v2nofix)
#   ELEMENTS 28                      (Elements axis = 2**ELEMENTS)
#   BENCH    ""                      (optional: one benchmark, e.g. transform/heavy)
set -euo pipefail

ROOT="${ROOT:-/tmp/cccl-bench}"
ELEMENTS="${ELEMENTS:-28}"
BENCH="${BENCH:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

V1="$ROOT/v1/bin/python"
V2FIX="$ROOT/v2fix/bin/python"
V2NOFIX="$ROOT/v2nofix/bin/python"
for p in "$V1" "$V2FIX" "$V2NOFIX"; do
  [[ -x "$p" ]] || { echo "ERROR: missing $p -- run setup_envs.sh first." >&2; exit 1; }
done

bench_arg=()
[[ -n "$BENCH" ]] && bench_arg=(-b "$BENCH")

run_one() {                          # label  v2python  outdir
  local label="$1" v2py="$2" outdir="$3"
  echo "======================================================================"
  echo " $label : v1 vs $(basename "$(dirname "$(dirname "$v2py")")")  (Elements=2^$ELEMENTS)"
  echo "======================================================================"
  cd "$HERE"
  "$V1" compare_v1_v2.py --v1-python "$V1" --v2-python "$v2py" \
    --elements "$ELEMENTS" "${bench_arg[@]}" --results-dir "$HERE/$outdir"
  "$V1" make_report.py --results "$HERE/$outdir" --out "$HERE/$outdir/report.pdf"
}

run_one "WITH FIX"    "$V2FIX"   results_28_fix
run_one "WITHOUT FIX" "$V2NOFIX" results_28_nofix

echo
echo "Done."
echo "  with fix:    $HERE/results_28_fix/report.pdf"
echo "  without fix: $HERE/results_28_nofix/report.pdf"
