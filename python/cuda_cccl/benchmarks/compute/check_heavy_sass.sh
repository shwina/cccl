#!/usr/bin/env bash
# Confirm the transform/heavy v1-vs-v2 regression is a local-memory spill.
# Run from the dir containing heavy_v1.cubin / heavy_v2.cubin (produced by
# diag_transform_heavy.py). Requires cuobjdump on PATH.
set -euo pipefail

for f in heavy_v1.cubin heavy_v2.cubin; do
  [[ -f "$f" ]] || { echo "ERROR: $f not found. Run diag_transform_heavy.py in each env first." >&2; exit 1; }
done
command -v cuobjdump >/dev/null || { echo "ERROR: cuobjdump not on PATH." >&2; exit 1; }

ldst() { cuobjdump -sass "$1" | grep -cE '\bLDL\b|\bSTL\b' || true; }
locptx() { cuobjdump -ptx "$1" | grep -cE 'ld\.local|st\.local' || true; }

echo "=== SASS local-memory instructions (LDL/STL) ==="
printf "  v1: %s\n" "$(ldst heavy_v1.cubin)"
printf "  v2: %s\n" "$(ldst heavy_v2.cubin)"

echo
echo "=== PTX ld.local/st.local count ==="
printf "  v1: %s\n" "$(locptx heavy_v1.cubin)"
printf "  v2: %s\n" "$(locptx heavy_v2.cubin)"

echo
echo "=== v2 PTX: .local decl + rolled-loop signature (first 30 lines) ==="
cuobjdump -ptx heavy_v2.cubin | grep -nE '\.local|ld\.local|st\.local|bra ' | head -30 || true

echo
echo "=== res-usage recap (STACK = local frame bytes) ==="
for f in heavy_v1.cubin heavy_v2.cubin; do
  echo "--- $f ---"
  cuobjdump -res-usage "$f" | grep -E 'Function|REG:|STACK:' || true
done
