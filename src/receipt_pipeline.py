from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np

# ---------- Regex & keyword config ----------
DATE_PATTERNS = [
    re.compile(r"\b(\d{2}/\d{2}/\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\b(\d{2}-\d{2}-\d{4})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b"),
    re.compile(r"\b(\d{2}/\d{2}/\d{2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b"),
]
DATE_KEYWORDS = ["date", "invoice", "time"]
TOTAL_KEYWORDS = ["total", "grand total", "net total", "amount"]
STORE_INVALID_KEYWORDS = ["total", "amount", "qty", "item", "cash", "change", "tax", "gst", "invoice", "receipt"]
STORE_SUFFIX_NOISE = ["SDN BHD", "SDN. BHD", "BHD", "LTD", "LIMITED", "INC", "CO", "COMPANY"]
CURRENCY_NUMBER_RE = re.compile(r"(?:RM|MYR|USD|\$|₹)?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))")


@dataclass
class FieldConfidence:
    value: Any
    confidence: float


@dataclass
class OCRLine:
    text: str
    conf: float
    top: int


# ---------- Generic text cleaning ----------
def clean_text(line: str) -> str:
    """Remove non-ASCII noise and normalize spaces while preserving basic receipt symbols."""
    if not line:
        return ""
    line = line.encode("ascii", errors="ignore").decode("ascii")
    line = re.sub(r"[^A-Za-z0-9\s\-\.,:/&()$]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


# ---------- Confidence scoring ----------
def compute_confidence(value: Any, ocr_conf: float, keyword_match: int, regex_valid: int) -> float:
    """Field-level confidence: 0.5*ocr + 0.3*keyword + 0.2*pattern_valid."""
    if value in (None, "", []):
        return 0.0
    score = 0.5 * max(0.0, min(1.0, ocr_conf)) + 0.3 * int(bool(keyword_match)) + 0.2 * int(bool(regex_valid))
    return round(max(0.0, min(1.0, score)), 3)


# ---------- Store extraction ----------
def normalize_store_name(name: str) -> str:
    """Normalize noisy variants to stable grouping keys."""
    if not name:
        return "UNKNOWN"

    raw = clean_text(name).upper()
    raw = raw.translate(str.maketrans("", "", string.punctuation))
    raw = re.sub(r"\s+", " ", raw).strip()

    for suffix in STORE_SUFFIX_NOISE:
        raw = re.sub(rf"\b{re.escape(suffix)}\b", "", raw).strip()

    raw = re.sub(r"\s+", " ", raw).strip()
    return raw if raw else "UNKNOWN"


def extract_store_name(ocr_lines: list[dict[str, Any]]) -> str | None:
    """Pick the best store candidate from top receipt lines with hard filters."""
    top_lines = ocr_lines[:7]
    candidates: list[tuple[float, str]] = []

    for line in top_lines:
        text = clean_text(line.get("text", ""))
        if not text or len(text) > 50:
            continue

        low = text.lower()
        if any(k in low for k in STORE_INVALID_KEYWORDS):
            continue

        alpha_count = sum(ch.isalpha() for ch in text)
        digit_count = sum(ch.isdigit() for ch in text)
        digit_ratio = digit_count / max(1, len(text.replace(" ", "")))
        if digit_ratio > 0.4 or alpha_count == 0:
            continue

        uppercase_ratio = sum(ch.isupper() for ch in text if ch.isalpha()) / max(1, alpha_count)
        score = 0.6 * line.get("conf", 0.0) + 0.4 * uppercase_ratio
        candidates.append((score, text))

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x[0])[1]
    return normalize_store_name(best)


# ---------- Date extraction ----------
def _validate_and_standardize_date(token: str) -> str | None:
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            dt = datetime.strptime(token, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_date(text_lines: list[str]) -> str | None:
    """Extract first valid date, preferring lines with date-related keywords."""
    prioritized: list[tuple[int, str]] = []

    for idx, line in enumerate(text_lines):
        cleaned = clean_text(line)
        if not cleaned:
            continue

        for pattern in DATE_PATTERNS:
            for match in pattern.findall(cleaned):
                candidate = _validate_and_standardize_date(match)
                if not candidate:
                    continue

                low = cleaned.lower()
                keyword_hit = int(any(k in low for k in DATE_KEYWORDS))
                # lower sort key = better: keyword lines first, then earliest line index
                prioritized.append((0 if keyword_hit else 1, idx, candidate))

    if not prioritized:
        return None

    prioritized.sort(key=lambda x: (x[0], x[1]))
    return prioritized[0][2]


# ---------- Total extraction ----------
def _parse_amount(text: str) -> list[float]:
    values: list[float] = []
    for match in CURRENCY_NUMBER_RE.findall(text):
        normalized = match.replace(",", "").replace(" ", "")
        try:
            values.append(float(normalized))
        except ValueError:
            continue
    return values


def extract_total(text_lines: list[str]) -> float | None:
    """Pass-1 keyword-based total extraction, pass-2 fallback to largest amount."""
    # Pass 1: keyword-guided
    for line in text_lines:
        cleaned = clean_text(line)
        low = cleaned.lower()
        if any(k in low for k in TOTAL_KEYWORDS):
            amounts = _parse_amount(cleaned)
            if amounts:
                return max(amounts)

    # Pass 2: fallback to largest amount in receipt
    all_values: list[float] = []
    for line in text_lines:
        all_values.extend(_parse_amount(clean_text(line)))

    return max(all_values) if all_values else None


class ReceiptPipeline:
    """EasyOCR-based receipt extraction pipeline with rule/regex post-processing."""

    def __init__(self) -> None:
        # EasyOCR stays as OCR engine per requirement.
        self.reader = easyocr.Reader(["en"], gpu=False)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = 800 / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image < 255))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def run_ocr(self, image: np.ndarray) -> list[dict[str, Any]]:
        results = self.reader.readtext(image, detail=1, paragraph=False, batch_size=8)
        lines: list[dict[str, Any]] = []

        for bbox, text, conf in results:
            top = int(min(point[1] for point in bbox))
            cleaned = clean_text(text)
            if not cleaned:
                continue
            lines.append({"text": cleaned, "conf": float(conf), "top": top})

        return sorted(lines, key=lambda x: x["top"])

    def extract_fields(self, ocr_lines: list[dict[str, Any]]) -> dict[str, Any]:
        text_lines = [line["text"] for line in ocr_lines]

        # store_name
        store_value = extract_store_name(ocr_lines)
        store_line_conf = next((line["conf"] for line in ocr_lines[:7] if normalize_store_name(line["text"]) == store_value), 0.0)
        store_conf = compute_confidence(store_value, store_line_conf, keyword_match=1, regex_valid=int(store_value not in (None, "UNKNOWN")))

        # date
        date_value = extract_date(text_lines)
        date_line_conf = max((line["conf"] for line in ocr_lines if date_value and date_value in line["text"]), default=0.0)
        date_keyword_hit = int(any(any(k in line["text"].lower() for k in DATE_KEYWORDS) for line in ocr_lines))
        date_conf = compute_confidence(date_value, date_line_conf, keyword_match=date_keyword_hit, regex_valid=int(date_value is not None))

        # total
        total_value = extract_total(text_lines)
        total_line_confs = [line["conf"] for line in ocr_lines if any(k in line["text"].lower() for k in TOTAL_KEYWORDS)]
        total_ocr_conf = float(np.mean(total_line_confs)) if total_line_confs else (float(np.mean([l["conf"] for l in ocr_lines])) if ocr_lines else 0.0)
        total_keyword_hit = int(any(any(k in line["text"].lower() for k in TOTAL_KEYWORDS) for line in ocr_lines))
        total_conf = compute_confidence(total_value, total_ocr_conf, keyword_match=total_keyword_hit, regex_valid=int(total_value is not None))

        return {
            "store_name": asdict(FieldConfidence(value=store_value or "UNKNOWN", confidence=store_conf)),
            "date": asdict(FieldConfidence(value=date_value, confidence=date_conf)),
            "total_amount": asdict(FieldConfidence(value=total_value, confidence=total_conf)),
            "raw_text_lines": ocr_lines,
        }


def process_receipt_file(pipeline: ReceiptPipeline, path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        return {"file": path.name, "error": "Unable to read image"}

    processed = pipeline.preprocess(image)
    ocr_lines = pipeline.run_ocr(processed)
    payload = pipeline.extract_fields(ocr_lines)
    payload["file"] = path.name
    payload["ocr_word_count"] = len(ocr_lines)
    payload["ocr_avg_confidence"] = round(float(np.mean([line["conf"] for line in ocr_lines])) if ocr_lines else 0.0, 3)
    return payload


def summarize_expenses(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    total_spend = 0.0
    spend_per_store: dict[str, float] = {}
    transaction_count = 0

    for receipt in receipts:
        total_val = (receipt.get("total_amount") or {}).get("value")
        if total_val is None:
            continue
        try:
            amount = float(total_val)
        except (TypeError, ValueError):
            continue

        store_val = (receipt.get("store_name") or {}).get("value") or "UNKNOWN"
        store_key = normalize_store_name(store_val)

        transaction_count += 1
        total_spend += amount
        spend_per_store[store_key] = round(spend_per_store.get(store_key, 0.0) + amount, 2)

    return {
        "total_spend": round(total_spend, 2),
        "transaction_count": transaction_count,
        "spend_per_store": dict(sorted(spend_per_store.items())),
    }


def run_pipeline(input_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pipeline = ReceiptPipeline()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ]

    receipts: list[dict[str, Any]] = []
    for path in image_paths:
        result = process_receipt_file(pipeline, path)
        receipts.append(result)
        (output_dir / f"{path.stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = summarize_expenses(receipts)
    (output_dir / "expense_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return receipts, summary


if __name__ == "__main__":
    # Example usage (ready to integrate in existing project flow)
    in_dir = Path("data/receipts")
    out_dir = Path("output")
    if in_dir.exists():
        recs, summary_payload = run_pipeline(in_dir, out_dir)
        print(f"Processed receipts: {len(recs)}")
        print(json.dumps(summary_payload, indent=2))
    else:
        print("Example usage: create data/receipts then run `python -m src.receipt_pipeline`")

