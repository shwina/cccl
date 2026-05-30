#!/usr/bin/env bash
# Find out WHY ptxas spills on v2's PTX but not v1's.
# The array is register-promoted in PTX (no ld.local) yet SASS spills (LDL/STL),
# so the spill is ptxas register allocation under the kernel's launch-bound cap.
# This extracts the embedded PTX, compares launch-bound directives, and re-runs
# ptxas -v with and without a raised register cap.
#
# Usage: ./check_heavy_ptxas.sh [SM]      # SM defaults to 89 (RTX 6000 Ada)
set -uo pipefail

SM="${1:-89}"
command -v cuobjdump >/dev/null || { echo "ERROR: cuobjdump not on PATH." >&2; exit 1; }
HAVE_PTXAS=1; command -v ptxas >/dev/null || { echo "WARN: ptxas not on PATH; skipping ptxas runs." >&2; HAVE_PTXAS=0; }

for tag in v1 v2; do
  cubin="heavy_${tag}.cubin"
  [[ -f "$cubin" ]] || { echo "ERROR: $cubin not found." >&2; exit 1; }
  cuobjdump -ptx "$cubin" > "${tag}.ptx" 2>/dev/null || true
  if [[ -s "${tag}.ptx" ]] && grep -qE '\.entry|\.visible' "${tag}.ptx"; then
    echo "[$tag] extracted PTX -> ${tag}.ptx ($(wc -l < ${tag}.ptx) lines)"
  else
    echo "[$tag] NO embedded PTX in $cubin (cubin is SASS-only) — ptxas re-run not possible for $tag"
    rm -f "${tag}.ptx"
  fi
done

echo
echo "=== launch-bound / reg-cap directives in PTX ==="
for tag in v1 v2; do
  [[ -f "${tag}.ptx" ]] || continue
  echo "--- ${tag}.ptx ---"
  grep -nE '\.maxntid|\.minnctapersm|\.maxnreg|\.reqntid|\.entry' "${tag}.ptx" | head -20 || echo "  (none found)"
done

if [[ "$HAVE_PTXAS" -eq 1 && -f v2.ptx ]]; then
  echo
  echo "=== ptxas -v on v2.ptx (default: honors launch bounds) ==="
  ptxas -arch="sm_${SM}" -v v2.ptx -o /dev/null 2>&1 | grep -iE 'registers|spill|stack|used|Function|gmem|cmem' || true

  echo
  echo "=== ptxas -v on v2.ptx with --maxrregcount=255 (lift the cap) ==="
  echo "    (if spill bytes drop to 0 here, the spill is a launch-bound/occupancy cap, not raw pressure)"
  ptxas -arch="sm_${SM}" -v --maxrregcount=255 v2.ptx -o /dev/null 2>&1 | grep -iE 'registers|spill|stack|used|Function' || true

  if [[ -f v1.ptx ]]; then
    echo
    echo "=== ptxas -v on v1.ptx (baseline) ==="
    ptxas -arch="sm_${SM}" -v v1.ptx -o /dev/null 2>&1 | grep -iE 'registers|spill|stack|used|Function' || true
  fi
fi
