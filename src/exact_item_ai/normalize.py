from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

from .models import ReceiptItem

# Merchant aliases are for obvious canonicalization only, not merchant identity
# inference or product/category reasoning.
MERCHANT_ALIASES = {
    "covenant home school resource center": "covenant home school resource center",
    "barnes & noble booksellers #2211": "barnes and noble",
    "barnes and noble booksellers #2211": "barnes and noble",
    "goodwill of central & northern arizona": "goodwill",
    "goodwill of central and northern arizona": "goodwill",
    "fry's food stores": "frys food stores",
    "tpt": "teachers pay teachers",
}

# These are weak local-text signals only. They do not permanently block later
# photo, shared-photo, receipt-level, retrieval, or adjudication evidence.
WEAK_TITLE_PATTERNS = {
    "misc",
    "books",
    "crafts",
    "toys and hobbies",
    "used curriculum and books",
    "site merch",
}
CONDITION_SUFFIX_RE = re.compile(
    r"\s*-\s*(acceptable|good|very good|like new|new)\s+condition$",
    re.IGNORECASE,
)
PARENTHETICAL_RE = re.compile(r"\(([^)]*)\)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9&+]+")
MULTISPACE_RE = re.compile(r"\s+")
DISCOUNT_HINT_RE = re.compile(r"(@|\beach\b|%|\$\d+\.\d+\s*-\s*\$\d+\.\d+)")
METADATA_PARENTHETICAL_RE = re.compile(
    r"\b(pre-?sale|ship|ships|shipping|online only|pickup|delivery|fulfilled|fulfillment|discount|promo)\b",
    re.IGNORECASE,
)


def parse_price(raw_value: object) -> Optional[float]:
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    text = str(raw_value).strip().replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def canonicalize_text(text: str) -> str:
    lowered = text.lower().strip().replace("&", " and ")
    lowered = lowered.replace("™", " ").replace("®", " ")
    lowered = NON_ALNUM_RE.sub(" ", lowered)
    return MULTISPACE_RE.sub(" ", lowered).strip()


def normalize_merchant(merchant: str) -> str:
    normalized = canonicalize_text(merchant)
    return MERCHANT_ALIASES.get(normalized, normalized)


def _merchant_normalization_note(merchant: str, normalized_merchant: str) -> Optional[str]:
    normalized = canonicalize_text(merchant)
    if normalized != normalized_merchant:
        return f"normalized merchant alias: {normalized} -> {normalized_merchant}"
    return None


def normalize_item_id(item_id: Optional[str]) -> Optional[str]:
    if not item_id:
        return None
    cleaned = re.sub(r"\s+", "", str(item_id).strip())
    return cleaned or None


def classify_item_id(item_id: Optional[str]) -> Optional[str]:
    if not item_id:
        return None
    if re.fullmatch(r"97[89]\d{10}", item_id):
        return "isbn13"
    if re.fullmatch(r"\d{12,14}", item_id):
        return "barcode"
    if re.fullmatch(r"[A-Za-z]{1,4}-[A-Za-z0-9-]+", item_id):
        return "merchant_sku"
    if re.fullmatch(r"[A-Za-z0-9]{4,12}", item_id):
        return "short_sku"
    if re.fullmatch(r"\d+\.\d+", item_id):
        return "catalog_code"
    return "other"


def extract_condition_metadata(item_name: str) -> tuple[str, Optional[str]]:
    cleaned, condition, _notes = _extract_title_metadata(item_name)
    return cleaned, condition


def _extract_title_metadata(item_name: str) -> tuple[str, Optional[str], list[str]]:
    cleaned = item_name.strip()
    notes: list[str] = []
    condition_match = CONDITION_SUFFIX_RE.search(cleaned)
    condition = _format_condition(condition_match.group(1)) if condition_match else None
    if condition_match:
        cleaned = CONDITION_SUFFIX_RE.sub("", cleaned)
        notes.append(f"extracted condition: {condition}")
    cleaned, parenthetical_notes = strip_metadata_parentheticals(cleaned)
    notes.extend(parenthetical_notes)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip(" -/")
    return cleaned.strip(), condition, notes


def strip_metadata_parentheticals(item_name: str) -> tuple[str, list[str]]:
    notes: list[str] = []

    def replace_parenthetical(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if METADATA_PARENTHETICAL_RE.search(content):
            notes.append(f"stripped metadata parenthetical: {content}")
            return " "
        notes.append(f"preserved identity parenthetical: {content}")
        return match.group(0)

    cleaned = PARENTHETICAL_RE.sub(replace_parenthetical, item_name)
    return cleaned, notes


def clean_item_name(item_name: str) -> str:
    cleaned, _condition = extract_condition_metadata(item_name)
    return cleaned


def _format_condition(condition: str) -> str:
    return " ".join(part.capitalize() for part in condition.lower().split())


def detect_discount_line(item_name: str) -> bool:
    return bool(DISCOUNT_HINT_RE.search(item_name))


def is_generic_title(item_name: str) -> bool:
    normalized = canonicalize_text(item_name)
    if not normalized:
        return True
    if normalized in WEAK_TITLE_PATTERNS:
        return True
    if normalized.startswith("membership"):
        return False
    if normalized.count(" ") <= 1 and normalized in {"misc", "books", "crafts"}:
        return True
    return False


def compute_specificity_score(item_name: str) -> float:
    normalized = canonicalize_text(item_name)
    if not normalized:
        return 0.0
    tokens = [token for token in normalized.split() if len(token) > 2]
    if not tokens:
        return 0.0
    unique_tokens = {token for token in tokens}
    informative = [token for token in unique_tokens if token not in {"with", "from", "shop", "each"}]
    alpha_ratio = sum(token.isalpha() for token in informative) / max(len(informative), 1)
    count_score = min(len(informative) / 6.0, 1.0)
    return round((count_score * 0.7) + (alpha_ratio * 0.3), 3)


def normalize_receipt_item(item: ReceiptItem) -> ReceiptItem:
    cleaned_item_name, item_condition, normalization_notes = _extract_title_metadata(item.item_name)
    normalized_item_name = canonicalize_text(cleaned_item_name)
    normalized_merchant = normalize_merchant(item.merchant)
    merchant_note = _merchant_normalization_note(item.merchant, normalized_merchant)
    if merchant_note:
        normalization_notes.append(merchant_note)
    normalized_item_id = normalize_item_id(item.item_id)
    return replace(
        item,
        item_condition=item_condition,
        cleaned_item_name=cleaned_item_name,
        normalized_item_name=normalized_item_name,
        normalized_merchant=normalized_merchant,
        normalized_item_id=normalized_item_id,
        has_item_id=normalized_item_id is not None,
        has_reference_photo=bool(item.reference_photo_urls),
        is_generic_title=is_generic_title(cleaned_item_name),
        looks_like_discount_line=detect_discount_line(item.item_name),
        specificity_score=compute_specificity_score(cleaned_item_name),
        id_type=classify_item_id(normalized_item_id),
        normalization_notes=normalization_notes,
    )
