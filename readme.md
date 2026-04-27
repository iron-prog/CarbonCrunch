# Receipt OCR Shortlisting Assignment Solution (EasyOCR)

This project extracts `store_name`, `date`, and `total_amount` from receipt images with confidence-aware outputs.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py --input data/receipts --output output
```

## One-command local verification

```bash
bash scripts/verify_local.sh data/receipts output
```

## Output JSON shape

```json
{
  "store_name": {"value": "BENS INDEPENDENT GROCER", "confidence": 0.92},
  "date": {"value": "2026-04-20", "confidence": 0.85},
  "total_amount": {"value": 123.45, "confidence": 0.95}
}
```

## Extraction improvements implemented

- Store name extraction from top lines with hard filtering (keyword/noise, numeric ratio, max length).
- Date extraction with regex + datetime validation and keyword prioritization.
- Total extraction with keyword-first pass, largest-number fallback.
- Field-level confidence:

```text
confidence = 0.5 * ocr_conf + 0.3 * keyword_match + 0.2 * regex_valid
```

- Store normalization for grouping consistency (`SDN BHD` variants collapsed).
