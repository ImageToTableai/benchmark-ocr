"""Programmatic CORD receipt field extraction from flattened OCR text.

CORD ground truth is a nested structure (menu rows, sub_total, total), so this
postprocessor recovers the same shape from plain OCR text using label and
line-item heuristics. Amount strings are emitted raw; eval/cord_metrics.py
normalizes thousands separators before comparison.

This is best-effort: flattened OCR loses the 2D layout, and menu item names
often carry spacing/punctuation differences versus ground truth, so name-level
matching is inherently limited. Amount/scalar recovery is the more reliable
signal here.
"""

from __future__ import annotations

import re

AMOUNT_RE = re.compile(r"-?[0-9][0-9,.]*")

# Label groups that should not be mistaken for menu line items.
_NON_ITEM_RE = re.compile(
    r"^(no\.?|qty|gt?y|total|sub\s?total|tax|ppn|disc|discount|service|"
    r"cash|change|kembali|credit|debit|edc|visa|master|card|receipt|invoice|"
    r"nota|struk|date|time|cashier|tel|fax|email|phone)\b",
    re.IGNORECASE,
)


def _last_amount(line: str) -> str | None:
    nums = AMOUNT_RE.findall(line)
    return nums[-1] if nums else None


def extract_cord_fields(text: str) -> dict:
    if not text:
        return {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    total: dict[str, str] = {}
    sub_total: dict[str, str] = {}
    menu: list[dict[str, str]] = []

    for line in lines:
        low = line.lower()

        # ---- total section ----
        if "total" in low:
            if "disc" in low or "discount" in low:
                value = _last_amount(line)
                if value is not None:
                    sub_total["discount_price"] = value
            else:
                value = _last_amount(line)
                if value is not None:
                    total["total_price"] = value
            quantity = re.search(r"gty\s*([0-9][0-9,.]*)", low)
            if quantity:
                total["menuqty_cnt"] = quantity.group(1)
            continue

        if "cash" in low:
            value = _last_amount(line)
            if value is not None:
                total["cashprice"] = value
            continue

        if "change" in low or "kembali" in low:
            value = _last_amount(line)
            if value is not None:
                total["changeprice"] = value
            continue

        if re.search(r"credit|edc|card|visa|master|debit", low):
            if re.search(r"\bno[.:]?\b", low):
                continue
            value = _last_amount(line)
            if value is not None:
                total["creditcardprice"] = value
            continue

        # ---- sub_total section ----
        if "subtotal" in low or "sub total" in low:
            value = _last_amount(line)
            if value is not None:
                sub_total["subtotal_price"] = value
            continue

        if re.search(r"\btax\b|ppn|pajak", low):
            value = _last_amount(line)
            if value is not None:
                sub_total["tax_price"] = value
            continue

        if "disc" in low or "discount" in low:
            value = _last_amount(line)
            if value is not None:
                sub_total["discount_price"] = value
            continue

        if "service" in low:
            value = _last_amount(line)
            if value is not None:
                sub_total["service_price"] = value
            continue

        # ---- menu line item: "name price" ----
        match = re.match(r"^(.+?)\s+(-?[0-9][0-9,.]*)$", line)
        if match:
            name = match.group(1).strip()
            price = match.group(2)
            if name and not _NON_ITEM_RE.match(name):
                menu.append({"nm": name, "price": price})

    fields: dict = {}
    if menu:
        fields["menu"] = menu
    if sub_total:
        fields["sub_total"] = sub_total
    if total:
        fields["total"] = total
    return fields
