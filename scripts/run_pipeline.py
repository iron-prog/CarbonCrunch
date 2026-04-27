from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.receipt_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run receipt OCR and extraction pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Folder of receipt images")
    parser.add_argument("--output", type=Path, required=True, help="Folder for JSON outputs")
    args = parser.parse_args()

    receipts, summary = run_pipeline(args.input, args.output)
    print(f"Processed {len(receipts)} receipts")
    print(f"Summary written with total_spend={summary['total_spend']}")


if __name__ == "__main__":
    main()
