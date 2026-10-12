"""Conservative literal fields for a user's explicitly completed market sale."""

from __future__ import annotations

import re
from decimal import Decimal

SALE_ASSERTION = (
    r"\bI(?:'ve| have)?\s+(?:(?:just|even|recently|already)\s+)?"
    r"(?:sold|earned|made|received)\b"
)
_SALE = re.compile(
    r"\bI(?:'ve| have)?\s+(?:(?:just|even|recently|already)\s+)?sold\s+"
    r"(?P<quantity>\d+(?:\.\d+)?)\s+(?P<product>[^\n.!?]{1,180}?)\s+at\s+"
    r"(?P<market>[^\n.!?]{1,180}?\bmarket\b)",
    re.I,
)
_TOTAL = re.compile(r"\bearning\s+(?:a total of\s+)?\$(\d+(?:\.\d+)?)\b", re.I)
_UNIT = re.compile(r"\bfor\s+\$(\d+(?:\.\d+)?)\s+each\b", re.I)
_UNSAFE = re.compile(
    r'["“”>]|\b(?:not|never|if|would|might|will|plan|hope|example|hypothetical|fictional|'
    r"quoted|said|refund(?:ed|s)?|discount(?:ed|s)?|returned|correct(?:ed|ion)?|"
    r"packs? of|boxes? of|bundles? of|dozen)\b",
    re.I,
)


def sale_fields(sentence: str) -> dict[str, str] | None:
    """No inferred dates, profits, currencies, sums or ambiguous unit conversions."""
    if _UNSAFE.search(sentence) or len(re.findall(SALE_ASSERTION, sentence, re.I)) != 1:
        return None
    sale = _SALE.search(sentence)
    totals, units = list(_TOTAL.finditer(sentence)), list(_UNIT.finditer(sentence))
    if sale is None or len(totals) + len(units) != 1:
        return None
    value = totals[0][1] if totals else units[0][1]
    if not (0 <= Decimal(sale["quantity"]) <= 1000000 and 0 <= Decimal(value) <= 1000000):
        return None
    fields = {
        "record_kind": "reported sale",
        "quantity": sale["quantity"],
        "product": sale["product"],
        "market": sale["market"],
    }
    fields["reported_total" if totals else "unit_price"] = value
    fields["currency_symbol"] = "$"
    return fields
