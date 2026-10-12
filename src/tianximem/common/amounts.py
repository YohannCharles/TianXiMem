"""Reversible per-record unit-price annotation; never a cross-record total."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, localcontext

_ANNOTATION = re.compile(r" \(calculated sale total: [^)]+\)")


def annotate_unit_total(content: str) -> str:
    """Only structured sale evidence with explicit quantity and dollar unit price."""
    try:
        record = json.loads(content)
        if not isinstance(record, dict) or record.get("record_kind") != "reported sale":
            return content
        quote, quantity, price = (record[k] for k in ("user_quotation", "quantity", "unit_price"))
        if not all(isinstance(v, str) for v in (quote, quantity, price)):
            return content
        if (
            record.get("currency_symbol") != "$"
            or "reported_total" in record
            or _ANNOTATION.search(quote)
        ):
            return content
        if not all(re.fullmatch(r"\d+(?:\.\d+)?", v) for v in (quantity, price)):
            return content
        match = re.search(r"\bfor\s+\$" + re.escape(price) + r"\s+each\b", quote, re.I)
        if (
            match is None
            or re.search(r"\bsold\s+" + re.escape(quantity) + r"\s", quote, re.I) is None
        ):
            return content
        if not (0 <= Decimal(quantity) <= 1000000 and 0 <= Decimal(price) <= 1000000):
            return content
        with localcontext() as context:
            context.prec = len(quantity.replace(".", "")) + len(price.replace(".", ""))
            value = format(Decimal(quantity) * Decimal(price), "f")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        annotation = f" (calculated sale total: {quantity} × ${price} = ${value})"
        annotated = quote[: match.end()] + annotation + quote[match.end() :]
        assert _ANNOTATION.sub("", annotated) == quote
        return json.dumps(record | {"user_quotation": annotated}, ensure_ascii=False)
    except (ValueError, KeyError, InvalidOperation):
        return content
