#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${1:-data/receipts}"
OUTPUT_DIR="${2:-output}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Input directory not found: $INPUT_DIR"
  echo "Usage: bash scripts/verify_local.sh [input_dir] [output_dir]"
  exit 1
fi

echo "[1/5] Syntax checks"
python -m py_compile src/receipt_pipeline.py scripts/run_pipeline.py

echo "[2/5] Verify EasyOCR import"
python - <<'PY'
import easyocr
print('easyocr_ok', easyocr.__version__ if hasattr(easyocr, '__version__') else 'installed')
PY

echo "[3/5] Run pipeline"
python scripts/run_pipeline.py --input "$INPUT_DIR" --output "$OUTPUT_DIR"

echo "[4/5] Count input images vs output JSON"
IMG_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tiff' \) | wc -l)
JSON_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' ! -name 'expense_summary.json' | wc -l)
echo "images=$IMG_COUNT json_receipts=$JSON_COUNT"

echo "[5/5] Show summary"
python -m json.tool "$OUTPUT_DIR/expense_summary.json"

