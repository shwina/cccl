#!/usr/bin/env bash
# Measure the hostjit unroll/SROA fix at realistic size (2**28) by running the
# v1-vs-v2 suite twice: once with the fix (HEAD) and once without (HEAD~1's
# compiler.cpp). Only compiler.cpp is toggled, so ninja recompiles one TU --
# LLVM is NOT rebuilt. v1 is the (identical) baseline in both runs.
#
# Outputs results_28_fix/ and results_28_nofix/ (+ report.pdf in each).
#
# Env overrides (defaults shown):
#   REPO     /home/ashwin/workspace/cccl
#   BD       /tmp/cccl-bench/build-v2/cp313-cp313-linux_x86_64   (warm ninja dir)
#   NINJA    $REPO/.venv/bin/ninja
#   TARGET   _bindings_impl.cpython-313-x86_64-linux-gnu.so
#   V1PY     /tmp/cccl-bench/v1/bin/python
#   V2PY     /tmp/cccl-bench/v2/bin/python
#   ELEMENTS 28
set -euo pipefail

REPO="${REPO:-/home/ashwin/workspace/cccl}"
BD="${BD:-/tmp/cccl-bench/build-v2/cp313-cp313-linux_x86_64}"
NINJA="${NINJA:-$REPO/.venv/bin/ninja}"
TARGET="${TARGET:-_bindings_impl.cpython-313-x86_64-linux-gnu.so}"
V1PY="${V1PY:-/tmp/cccl-bench/v1/bin/python}"
V2PY="${V2PY:-/tmp/cccl-bench/v2/bin/python}"
ELEMENTS="${ELEMENTS:-28}"
CC="c/parallel.v2/src/hostjit/compiler.cpp"
HERE="$REPO/python/cuda_cccl/benchmarks/compute"

for p in "$NINJA" "$V1PY" "$V2PY"; do
  [[ -x "$p" ]] || { echo "ERROR: not executable: $p (override via env)" >&2; exit 1; }
done
[[ -f "$BD/build.ninja" ]] || { echo "ERROR: no build.ninja in $BD" >&2; exit 1; }

# Always restore the working tree to HEAD's compiler.cpp on exit.
restore() { git -C "$REPO" checkout -q HEAD -- "$CC" 2>/dev/null || true; }
trap 'restore; "$NINJA" -C "$BD" "$TARGET" >/dev/null 2>&1 || true' EXIT

run_case() {
  local label="$1" ref="$2" outdir="$3"
  echo "======================================================================"
  echo " $label : compiler.cpp @ $ref  ->  $outdir   (Elements=2^$ELEMENTS)"
  echo "======================================================================"
  git -C "$REPO" checkout -q "$ref" -- "$CC"
  "$NINJA" -C "$BD" "$TARGET"
  cd "$HERE"
  "$V2PY" compare_v1_v2.py --v1-python "$V1PY" --v2-python "$V2PY" \
    --elements "$ELEMENTS" --results-dir "$HERE/$outdir"
  "$V2PY" make_report.py --results "$HERE/$outdir" --out "$HERE/$outdir/report.pdf"
}

run_case "WITH FIX"    HEAD    results_28_fix
run_case "WITHOUT FIX" HEAD~1  results_28_nofix

echo
echo "Done."
echo "  with fix:    $HERE/results_28_fix/report.pdf"
echo "  without fix: $HERE/results_28_nofix/report.pdf"
echo "Compare the v2 columns; v1 should be ~identical across the two."
