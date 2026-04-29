"""Confidence and evidence helpers shared by resolution stages."""

from __future__ import annotations

from typing import Any, Dict

from .models import ReceiptItem


def clamp_confidence(value: float) -> float:
    """Keep confidence scores inside the project-wide 0.00 to 0.99 range."""

    return round(max(0.0, min(0.99, value)), 2)


def build_evidence(
    *,
    route_reason: str,
    used_item_id: bool = False,
    used_title_match: bool = False,
    used_photo: bool = False,
    used_price: bool = False,
    used_merchant: bool = False,
    candidate_basis: str | None = None,
) -> Dict[str, Any]:
    """Build the compact evidence dictionary attached to every result."""

    return {
        "used_item_id": used_item_id,
        "used_title_match": used_title_match,
        "used_photo": used_photo,
        "used_price": used_price,
        "used_merchant": used_merchant,
        "candidate_basis": candidate_basis,
        "route_reason": route_reason,
    }


def deterministic_confidence(item: ReceiptItem, *, uses_item_id: bool) -> float:
    """Score a locally resolved item based on title specificity and ID strength."""

    score = 0.58
    if uses_item_id:
        score += 0.18
    if item.id_type == "isbn13":
        score += 0.1
    if item.id_type == "merchant_sku":
        score += 0.08
    if item.specificity_score >= 0.85:
        score += 0.1
    elif item.specificity_score >= 0.65:
        score += 0.05
    if item.has_reference_photo:
        score += 0.04
    return clamp_confidence(score)


def ambiguous_confidence(item: ReceiptItem) -> float:
    """Score items with some useful evidence but no confident exact match."""

    score = 0.34 + (item.specificity_score * 0.18)
    if item.has_item_id and item.id_type in {"short_sku", "catalog_code", "barcode"}:
        score += 0.05
    return clamp_confidence(score)


def insufficient_confidence(item: ReceiptItem) -> float:
    """Score abstentions where available evidence is too weak to identify an item."""

    score = 0.12
    if item.has_reference_photo:
        score += 0.06
    if item.has_item_id and item.is_generic_title:
        score += 0.03
    return clamp_confidence(score)


def apply_confidence_delta(base_confidence: float, delta: float) -> float:
    """Apply an evidence-driven confidence adjustment with normal clamping."""

    return clamp_confidence(base_confidence + delta)
