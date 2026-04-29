"""Route normalized receipt items into the safest resolution strategy."""

from __future__ import annotations

from .models import ReceiptItem, RouteDecision


def has_strong_identifier(item: ReceiptItem) -> bool:
    """Return whether the item ID format is strong enough to help local resolution."""

    if item.id_type in {"isbn13", "merchant_sku"}:
        return True
    if item.id_type == "barcode" and not item.is_generic_title:
        return True
    return False


def looks_like_abbreviated_title(item: ReceiptItem) -> bool:
    """Detect shorthand receipt titles that should avoid deterministic matches."""

    tokens = item.normalized_item_name.split()
    if not tokens:
        return False
    return any(len(token) == 1 for token in tokens)


def route_item(item: ReceiptItem) -> RouteDecision:
    """Choose the first pipeline bucket that fits the item's available evidence."""

    if item.looks_like_discount_line:
        return RouteDecision(
            bucket="insufficient_evidence",
            reason="Receipt line looks like a department or discount summary, not an exact item.",
        )

    if item.is_generic_title and not item.has_reference_photo:
        # Category-level text with no photo has too little evidence to identify
        # one exact item, even if it has weak metadata like price or merchant.
        return RouteDecision(
            bucket="insufficient_evidence",
            reason="Title is category-level and there is not enough metadata to identify one exact item.",
        )

    if item.is_generic_title and item.has_reference_photo:
        # A photo can turn a weak category line into an exact product, but the
        # resolver will still require later visual/adjudication evidence.
        return RouteDecision(
            bucket="photo_assisted",
            reason="Text is weak, but a reference photo may provide disambiguating evidence.",
        )

    if has_strong_identifier(item) and item.specificity_score >= 0.45 and not looks_like_abbreviated_title(item):
        # Strong IDs help only when the title has enough signal and does not look
        # like receipt shorthand.
        return RouteDecision(
            bucket="deterministic",
            reason="Specific title and strong identifier support a fast-path resolution.",
        )

    if item.specificity_score >= 0.78:
        return RouteDecision(
            bucket="deterministic",
            reason="Title appears specific enough to attempt a local deterministic resolution.",
        )

    if item.has_reference_photo:
        return RouteDecision(
            bucket="photo_assisted",
            reason="Photo is available and title may need additional grounding.",
        )

    return RouteDecision(
        bucket="retrieval_needed",
        reason="Text has some signal but not enough for a confident deterministic match.",
    )
