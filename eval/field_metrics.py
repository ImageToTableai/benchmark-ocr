"""Field-level metrics for OCR benchmark evaluation."""

from __future__ import annotations

import re


def extract_rec_fields(text: str) -> dict[str, str]:
    """Extract receipt fields (company, date, total, address) from OCR text."""
    fields: dict[str, str] = {}
    # Keep the postprocessor bounded. Some document parsers emit long markdown or
    # HTML-like lines, and whole-text regexes can dominate evaluation time.
    lines = [line.strip()[:300] for line in text[:20000].splitlines() if line.strip()]

    # total: look for TOTAL line followed by a price with optional RM prefix
    # Patterns: "TOTAL: 193.00", "TOTAL(RM): 436.20", "Total Amount: 170.00", "TOTAL INCLUSIVE GST: 193.00"
    for index, line in enumerate(lines):
        if not re.search(r"(?i)\btotal\b", line):
            continue
        search_window = " ".join(lines[index : index + 2])
        total_match = re.search(r"(?i)\btotal\b.{0,80}?(?:rm\s*)?(\d+\.\d{2})", search_window)
        if total_match:
            fields["total"] = total_match.group(1)
            break

    # date: DD/MM/YYYY or DD/MM/YY after date label
    for line in lines:
        date_match = re.search(r"(?i)\bdate\b\s*[:#=-]?\s*(\d{1,2}/\d{1,2}/\d{2,4})", line)
        if date_match:
            fields["date"] = date_match.group(1)
            break

    # company: look for SDN BHD, BHD, ENTERPRISE, or the first all-caps name block
    for line in lines[:80]:
        if re.search(r"(?i)\b(?:SDN\.?\s*BHD|BHD|ENTERPRISE|SDN)\b", line):
            fields["company"] = line
            break
    if "company" not in fields:
        # fallback: first all-caps line that looks like a company name (2+ words, >= 10 chars)
        for line in lines[:80]:
            if len(line) >= 10 and line.upper() == line and len(line.split()) >= 2:
                if not re.search(r"(?i)^(?:tax|cash|invoice|date|total|no\.|tel|fax|email)", line):
                    fields["company"] = line
                    break

    # address: lines containing postal code pattern (5-digit Malaysian) or street indicators
    addr_lines = []
    in_addr = False
    for line in lines[:120]:
        if re.search(r"(?i)\b(jalan|street|road|lorong|lebuh|persiaran|no\.?\s*\d)", line) or re.search(r"\b\d{5}\b", line):
            in_addr = True
        if in_addr and line:
            addr_lines.append(line)
        if re.search(r"\b\d{5}\b", line) or (in_addr and re.search(r"(?i)\b(tel|fax|email|invoice|date|total)\b", line)):
            break
    if addr_lines:
        fields["address"] = " ".join(addr_lines)

    return fields


def extract_demo_fields(text: str) -> dict[str, str]:
    fields = extract_rec_fields(text)
    if not fields:
        total_match = re.search(r"\btotal\b\s*[:#-]?\s*([0-9]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
        date_match = re.search(r"\bdate\b\s*[:#-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text, re.IGNORECASE)
        if total_match:
            fields["total"] = total_match.group(1)
        if date_match:
            fields["date"] = date_match.group(1)
    return fields


def normalize_field_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def field_value_accuracy(predicted: dict[str, str], ground_truth: dict[str, str]) -> float:
    """Macro share of required GT fields whose values match exactly after whitespace normalization."""
    if not ground_truth:
        return 0.0
    correct = 0
    for key, gt_value in ground_truth.items():
        if normalize_field_value(predicted.get(key, "")) == normalize_field_value(gt_value):
            correct += 1
    return correct / len(ground_truth)


def document_fields_exact(predicted: dict[str, str], ground_truth: dict[str, str]) -> float:
    """Whether every required GT field is present and exactly matched for one document."""
    if not ground_truth:
        return 0.0
    return float(
        all(normalize_field_value(predicted.get(key, "")) == normalize_field_value(gt_value) for key, gt_value in ground_truth.items())
    )


def field_value_f1(predicted: dict[str, str], ground_truth: dict[str, str]) -> dict[str, float]:
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_positives = 0
    for key, gt_value in ground_truth.items():
        pred_value = predicted.get(key, "")
        if pred_value and normalize_field_value(pred_value) == normalize_field_value(gt_value):
            true_positives += 1
    total_predicted = sum(1 for v in predicted.values() if v and str(v).strip())
    total_gt = len(ground_truth)
    precision = true_positives / total_predicted if total_predicted > 0 else 0.0
    recall = true_positives / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_score}


# Kept for scripts that read historical metrics. New benchmark output uses the
# explicit names above so a per-field average is not mistaken for document exactness.
field_exact = field_value_accuracy
field_f1 = field_value_f1
