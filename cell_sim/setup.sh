#!/bin/bash
# setup.sh — clone upstream data and install dependencies
# Idempotent: safe to run multiple times.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== cell_sim setup ==="

# Step 1: Clone the Luthey-Schulten data if missing
DATA_DIR="data/Minimal_Cell_ComplexFormation"
if [ ! -d "$DATA_DIR" ]; then
    echo "Cloning Luthey-Schulten Syn3A data..."
    mkdir -p data
    cd data
    git clone --depth 1 https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation.git
    cd ..
else
    echo "Upstream data already present at $DATA_DIR"
fi

# Step 2: Check required files
echo "Checking required files..."
required=(
    "$DATA_DIR/input_data/syn3A.gb"
    "$DATA_DIR/input_data/Syn3A_updated.xml"
    "$DATA_DIR/input_data/initial_concentrations.xlsx"
    "$DATA_DIR/input_data/kinetic_params.xlsx"
    "$DATA_DIR/input_data/complex_formation.xlsx"
)
all_present=1
for f in "${required[@]}"; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ✓ $f ($size)"
    else
        echo "  ✗ MISSING: $f"
        all_present=0
    fi
done

if [ $all_present -eq 0 ]; then
    echo ""
    echo "ERROR: upstream data files missing. Check your network and re-run."
    exit 1
fi

# Step 3: Install Python deps
echo ""
echo "Installing Python dependencies..."
# Try regular pip, fall back to --break-system-packages if needed (PEP 668)
pip install -q -r requirements.txt 2>/dev/null || \
    pip install -q --break-system-packages -r requirements.txt 2>/dev/null || \
    pip3 install -q --user -r requirements.txt

# Step 4: Quick smoke test
echo ""
echo "Running smoke test (parse genome + load proteome)..."
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from layer0_genome.syn3a_real import build_real_syn3a_cellspec
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    spec, counts, complexes, kcats = build_real_syn3a_cellspec()
print(f"  Loaded: {len(spec.proteins)} proteins, {len(counts)} with counts, "
      f"{len(complexes)} complexes, {len(kcats)} reactions with k_cat")
PY

echo ""
echo "=== setup complete ==="
echo "Try: python tests/render_priority_15.py"
