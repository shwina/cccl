#!/usr/bin/env bash
# Build three fresh benchmark envs with uv (Python 3.13; cuda-bench/cupy have no
# 3.14 wheels). Each gets its own venv and its own build dir -- no LLVM reuse:
#   v1       : default backend (NVRTC/LTO; no hostjit/LLVM build)
#   v2fix    : CCCL_PYTHON_USE_V2=ON at HEAD            (the unroll/SROA fix)
#   v2nofix  : CCCL_PYTHON_USE_V2=ON at HEAD~1 compiler.cpp  (no fix)
#
# The two v2 builds each compile LLVM from scratch (slow) -- intentional, per
# "don't worry about caching llvm".
#
# Env overrides: REPO, ROOT (venv/build root), CU (cu13), PYV (3.13)
set -euo pipefail

REPO="${REPO:-/home/ashwin/workspace/cccl}"
PKG="$REPO/python/cuda_cccl"
ROOT="${ROOT:-/tmp/cccl-bench}"
CU="${CU:-cu13}"
PYV="${PYV:-3.13}"
CC="c/parallel.v2/src/hostjit/compiler.cpp"

command -v uv >/dev/null || { echo "ERROR: uv not on PATH" >&2; exit 1; }
mkdir -p "$ROOT"

build_env() {                 # name  use_v2(0|1)  build_subdir
  local name="$1" use_v2="$2" bdsub="$3"
  local venv="$ROOT/$name"
  echo "=========================================================="
  echo " building $name  (USING_V2=$use_v2)  -> $venv"
  echo "=========================================================="
  rm -rf "$venv"
  uv venv --python "$PYV" "$venv"
  # nvbench-compare runtime deps:
  uv pip install --python "$venv/bin/python" colorama jsondiff tabulate
  local cmake_env=()
  [[ "$use_v2" == "1" ]] && cmake_env=(CMAKE_ARGS="-DCCCL_PYTHON_USE_V2=ON")
  env "${cmake_env[@]}" uv pip install --python "$venv/bin/python" --verbose \
    --config-settings=build.verbose=true \
    --config-settings="build-dir=$ROOT/$bdsub/{wheel_tag}" \
    -e "$PKG[bench-$CU]"
  "$venv/bin/python" - <<'PY'
try:
    from cuda.compute._build_info import USING_V2
    print("  -> USING_V2 =", USING_V2)
except Exception:
    print("  -> _build_info absent (treated as v1)")
PY
}

# Always leave the working tree on HEAD's compiler.cpp.
trap 'git -C "$REPO" checkout -q HEAD -- "$CC" 2>/dev/null || true' EXIT

# v1: default backend, no hostjit/LLVM
git -C "$REPO" checkout -q HEAD -- "$CC"
build_env v1 0 bd-v1

# v2 WITH fix (current HEAD)
git -C "$REPO" checkout -q HEAD -- "$CC"
build_env v2fix 1 bd-v2fix

# v2 WITHOUT fix (revert just compiler.cpp to HEAD~1)
git -C "$REPO" checkout -q HEAD~1 -- "$CC"
build_env v2nofix 1 bd-v2nofix
git -C "$REPO" checkout -q HEAD -- "$CC"

echo
echo "Done. Envs: $ROOT/{v1,v2fix,v2nofix}"
echo "Next:"
echo "  cd $PKG/benchmarks/compute"
echo "  python compare_v1_v2.py --v1-python $ROOT/v1/bin/python --v2-python $ROOT/v2fix/bin/python   --elements 28 --results-dir results_28_fix   && python make_report.py --results results_28_fix   --out results_28_fix/report.pdf"
echo "  python compare_v1_v2.py --v1-python $ROOT/v1/bin/python --v2-python $ROOT/v2nofix/bin/python --elements 28 --results-dir results_28_nofix && python make_report.py --results results_28_nofix --out results_28_nofix/report.pdf"
