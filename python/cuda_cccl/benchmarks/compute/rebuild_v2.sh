#!/usr/bin/env bash
# Incrementally rebuild the v2 hostjit lib after a C++ change, by driving the
# ALREADY-CONFIGURED warm build dir directly with `cmake --build`.
#
# Why not pip: `pip install -e` reconfigures into build/{wheel_tag}; if that dir
# isn't the warm one, CPM re-clones + rebuilds LLVM (huge). `cmake --build` on the
# existing dir never reconfigures (no CMake files changed) -> CPM never runs ->
# only changed TUs (compiler.cpp) recompile.
#
# Usage:  ./rebuild_v2.sh [BUILD_DIR] [HOST_PY]
#   BUILD_DIR : the warm cmake build dir (has CMakeCache.txt + _deps/llvm_project-src)
#               default: <repo>/python/cuda_cccl/build/cp314-cp314-linux_x86_64
#   HOST_PY   : python of the env you run diag with (for the verify step)
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # -> python/cuda_cccl
BUILD_DIR="${1:-${BUILD_DIR:-${PKG_DIR}/build/cp314-cp314-linux_x86_64}}"
HOST_PY="${2:-${V2_PY:-}}"
JOBS="${JOBS:-4}"

command -v cmake >/dev/null || { echo "ERROR: cmake not on PATH." >&2; exit 1; }
[[ -f "${BUILD_DIR}/CMakeCache.txt" ]] || {
  echo "ERROR: ${BUILD_DIR} is not a configured cmake build dir (no CMakeCache.txt)." >&2
  echo "Pick the warm one. Candidates:" >&2
  for d in "${PKG_DIR}"/build/*/; do
    [[ -f "${d}CMakeCache.txt" ]] && echo "  $d" >&2
  done
  exit 1
}
[[ -d "${BUILD_DIR}/_deps/llvm_project-src" ]] \
  && echo "OK: warm LLVM present in ${BUILD_DIR}/_deps (no refetch)." \
  || echo "WARN: no _deps/llvm_project-src in ${BUILD_DIR}; a reconfigure could refetch."

# Belt-and-suspenders: if anything *does* reconfigure, cache the CPM clone so
# LLVM is never re-downloaded.
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-${HOME}/.cache/cpm}"
echo "CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}"

echo "Building (incremental, verbose) in ${BUILD_DIR} with ${JOBS} jobs ..."
cmake --build "${BUILD_DIR}" -j "${JOBS}" --verbose

if [[ -n "${HOST_PY}" && -x "${HOST_PY}" ]]; then
  echo
  echo "=== verify the new dump code is in the extension ${HOST_PY} actually loads ==="
  "${HOST_PY}" - <<'PY'
import cuda.compute._bindings_impl as m
print("loaded extension:", m.__file__)
data = open(m.__file__, "rb").read()
print("OK: dump code present" if b"hostjit] dumped" in data
      else "WARNING: dump string NOT in the loaded extension — rebuild didn't land here")
PY
else
  echo "(pass HOST_PY as arg 2 to auto-verify the dump string landed)"
fi
