from __future__ import annotations

import base64
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence
from urllib.request import Request, urlopen

from .models import ReceiptItem, ResolutionResult
from .photo_assist import PhotoAIConfig, fetch_image_bytes
from .score import clamp_confidence


@dataclass(slots=True)
class SiblingContext:
    """Soft receipt-level context from other line items on the same receipt."""

    receipt_index: int
    merchant: str
    sibling_count: int
    resolved_sibling_titles: list[str] = field(default_factory=list)
    resolved_sibling_categories: list[str] = field(default_factory=list)
    sibling_item_name_patterns: list[str] = field(default_factory=list)
    sibling_item_id_patterns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VisiblePhotoCandidate:
    title: str | None
    description: str | None
    category: str | None
    author_or_brand: str | None
    confidence: float
    signals: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SharedPhotoEvidence:
    photo_url: str
    candidates: list[VisiblePhotoCandidate] = field(default_factory=list)
    success: bool = False
    notes: str | None = None
    analyzer_name: str = "noop"


@dataclass(slots=True)
class ReceiptLevelAssignment:
    item_index: int
    assigned_candidate_title: str | None
    assigned_description: str | None
    assigned_category: str | None
    assigned_confidence: float
    assignment_basis: str
    should_update_result: bool
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CatalogReceiptContext:
    """Receipt-level signal that a group behaves like a coherent catalog order."""

    merchant_catalog_confidence: float
    receipt_coherence_score: float
    catalog_like_receipt: bool
    sibling_named_title_ratio: float
    item_id_coverage: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SiblingAdjudicationResult:
    family_consistent: bool
    sibling_support_strength: str
    should_increase_confidence: bool
    confidence_delta: float
    can_promote_to_resolved: bool
    rationale: str | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = True
    adjudicator_name: str = "noop"


class SharedPhotoAnalyzer(Protocol):
    """Pluggable analyzer for photos that may contain multiple receipt items."""

    def analyze_shared_photo(
        self,
        image_source: str,
        receipt_context: Sequence[ReceiptItem],
    ) -> SharedPhotoEvidence:
        """Extract visually grounded candidate products from a shared image."""


class SiblingContextAdjudicator(Protocol):
    """Model-backed semantic adjudicator for same-receipt sibling support."""

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        sibling_context: SiblingContext,
    ) -> SiblingAdjudicationResult:
        """Decide whether resolved siblings support the current candidate."""


class NoopSiblingContextAdjudicator:
    """Safe fallback that disables model-based sibling promotion."""

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        sibling_context: SiblingContext,
    ) -> SiblingAdjudicationResult:
        return SiblingAdjudicationResult(
            family_consistent=False,
            sibling_support_strength="none",
            should_increase_confidence=False,
            confidence_delta=0.0,
            can_promote_to_resolved=False,
            rationale="No sibling-context adjudicator configured.",
            success=False,
            adjudicator_name="noop",
        )


class NoopSharedPhotoAnalyzer:
    """Safe fallback that disables shared-photo receipt-level reasoning."""

    def analyze_shared_photo(
        self,
        image_source: str,
        receipt_context: Sequence[ReceiptItem],
    ) -> SharedPhotoEvidence:
        return SharedPhotoEvidence(
            photo_url=image_source,
            candidates=[],
            success=False,
            notes="No shared-photo analyzer configured.",
            analyzer_name="noop",
        )


class AIBasedSiblingContextAdjudicator:
    """AI-backed semantic adjudicator for sibling-item context."""

    def __init__(self, config: PhotoAIConfig | None = None) -> None:
        self.config = config if config is not None else PhotoAIConfig.from_env()
        self.analysis_count = 0

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        sibling_context: SiblingContext,
    ) -> SiblingAdjudicationResult:
        if not self.config.enabled:
            return self._disabled("PHOTO_AI_ENABLED is not true.")
        if self.config.provider != "openai":
            return self._disabled(f"Unsupported PHOTO_AI_PROVIDER: {self.config.provider}.")
        if not self.config.api_key:
            return self._disabled("Missing PHOTO_AI_API_KEY or OPENAI_API_KEY.")
        if self.config.max_photo_analyses is not None and self.analysis_count >= self.config.max_photo_analyses:
            return self._disabled("PHOTO_AI_MAX_ANALYSES limit reached.")

        self.analysis_count += 1
        try:
            payload = self._call_openai_sibling_adjudicator(item_context, current_result, sibling_context)
            return self._result_from_payload(payload)
        except Exception as exc:
            return self._disabled(f"Sibling-context adjudication failed: {exc!s}")

    def _disabled(self, note: str) -> SiblingAdjudicationResult:
        return SiblingAdjudicationResult(
            family_consistent=False,
            sibling_support_strength="none",
            should_increase_confidence=False,
            confidence_delta=0.0,
            can_promote_to_resolved=False,
            rationale=note,
            success=False,
            adjudicator_name="ai_openai",
        )

    def _call_openai_sibling_adjudicator(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        sibling_context: SiblingContext,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": build_sibling_adjudication_prompt(item_context, current_result, sibling_context),
                }
            ],
            "temperature": 0,
            "max_tokens": 700,
        }
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "exact-item-ai/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
        return json.loads(content)

    def _result_from_payload(self, payload: dict[str, Any]) -> SiblingAdjudicationResult:
        strength = str(payload.get("sibling_support_strength") or "none").lower()
        if strength not in {"none", "weak", "moderate", "strong"}:
            strength = "none"
        return SiblingAdjudicationResult(
            family_consistent=bool(payload.get("family_consistent")),
            sibling_support_strength=strength,
            should_increase_confidence=bool(payload.get("should_increase_confidence")),
            confidence_delta=clamp_confidence(_coerce_float(payload.get("confidence_delta"), 0.0)),
            can_promote_to_resolved=bool(payload.get("can_promote_to_resolved")),
            rationale=_string_or_none(payload.get("rationale")),
            notes=_list_of_strings(payload.get("notes")),
            success=True,
            adjudicator_name="ai_openai",
        )


class AIBasedSharedPhotoAnalyzer:
    """AI-backed extractor for shared photos containing multiple visible items."""

    def __init__(self, config: PhotoAIConfig | None = None) -> None:
        self.config = config if config is not None else PhotoAIConfig.from_env()
        self.analysis_count = 0

    def analyze_shared_photo(
        self,
        image_source: str,
        receipt_context: Sequence[ReceiptItem],
    ) -> SharedPhotoEvidence:
        if not self.config.enabled:
            return self._disabled(image_source, "PHOTO_AI_ENABLED is not true.")
        if self.config.provider != "openai":
            return self._disabled(image_source, f"Unsupported PHOTO_AI_PROVIDER: {self.config.provider}.")
        if not self.config.api_key:
            return self._disabled(image_source, "Missing PHOTO_AI_API_KEY or OPENAI_API_KEY.")
        if self.config.max_photo_analyses is not None and self.analysis_count >= self.config.max_photo_analyses:
            return self._disabled(image_source, "PHOTO_AI_MAX_ANALYSES limit reached.")

        image = fetch_image_bytes(
            image_source,
            timeout_seconds=self.config.timeout_seconds,
            max_image_bytes=self.config.max_image_bytes,
        )
        if not image.success or image.content is None:
            return self._disabled(image_source, image.error or "Image fetch failed.")

        self.analysis_count += 1
        try:
            payload = self._call_openai_shared_photo(image.content, image.mime_type, receipt_context)
            return self._evidence_from_payload(image_source, payload)
        except Exception as exc:
            return self._disabled(image_source, f"Shared photo AI provider call failed: {exc!s}")

    def _disabled(self, image_source: str, note: str) -> SharedPhotoEvidence:
        return SharedPhotoEvidence(
            photo_url=image_source,
            candidates=[],
            success=False,
            notes=note,
            analyzer_name="ai_openai",
        )

    def _call_openai_shared_photo(
        self,
        image_bytes: bytes,
        mime_type: str,
        receipt_context: Sequence[ReceiptItem],
    ) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{image_b64}"
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_shared_photo_prompt(receipt_context)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 900,
        }
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "exact-item-ai/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
        return json.loads(content)

    def _evidence_from_payload(self, image_source: str, payload: dict[str, Any]) -> SharedPhotoEvidence:
        candidates = []
        for raw_candidate in payload.get("candidates", []):
            if not isinstance(raw_candidate, dict):
                continue
            title = _string_or_none(raw_candidate.get("title"))
            confidence = clamp_confidence(_coerce_float(raw_candidate.get("confidence"), 0.0))
            if not title or confidence < 0.55:
                continue
            candidates.append(
                VisiblePhotoCandidate(
                    title=title,
                    description=_string_or_none(raw_candidate.get("description")),
                    category=_string_or_none(raw_candidate.get("category")),
                    author_or_brand=_string_or_none(raw_candidate.get("author_or_brand")),
                    confidence=round(confidence, 2),
                    signals=_list_of_strings(raw_candidate.get("signals")),
                )
            )
        return SharedPhotoEvidence(
            photo_url=image_source,
            candidates=candidates,
            success=bool(candidates),
            notes=_string_or_none(payload.get("notes")),
            analyzer_name="ai_openai",
        )


def create_default_shared_photo_analyzer() -> SharedPhotoAnalyzer:
    config = PhotoAIConfig.from_env()
    if config.enabled:
        return AIBasedSharedPhotoAnalyzer(config)
    return NoopSharedPhotoAnalyzer()


def create_default_sibling_context_adjudicator() -> SiblingContextAdjudicator:
    config = PhotoAIConfig.from_env()
    if config.enabled:
        return AIBasedSiblingContextAdjudicator(config)
    return NoopSiblingContextAdjudicator()


def apply_receipt_level_context(
    items: Sequence[ReceiptItem],
    results: Sequence[ResolutionResult],
    *,
    shared_photo_analyzer: SharedPhotoAnalyzer | None = None,
    sibling_context_adjudicator: SiblingContextAdjudicator | None = None,
) -> list[ResolutionResult]:
    """Apply conservative receipt-level context after item-level resolution."""

    analyzer = shared_photo_analyzer if shared_photo_analyzer is not None else create_default_shared_photo_analyzer()
    sibling_adjudicator = (
        sibling_context_adjudicator
        if sibling_context_adjudicator is not None
        else create_default_sibling_context_adjudicator()
    )
    updated_results = list(results)
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped_indices[(item.dataset_name, item.receipt_index)].append(index)

    for indices in grouped_indices.values():
        receipt_items = [items[index] for index in indices]
        receipt_results = [updated_results[index] for index in indices]
        catalog_context = build_catalog_receipt_context(receipt_items)
        for local_index, result_index in enumerate(indices):
            updated_results[result_index] = apply_catalog_context_promotion(
                receipt_items[local_index],
                updated_results[result_index],
                catalog_context,
            )

        receipt_results = [updated_results[index] for index in indices]
        sibling_contexts = build_sibling_contexts(receipt_items, receipt_results)
        current_receipt_results = [updated_results[index] for index in indices]
        sibling_updates = apply_sibling_family_context(
            receipt_items,
            current_receipt_results,
            sibling_contexts,
            sibling_adjudicator,
        )
        for local_index, result_index in enumerate(indices):
            updated_results[result_index] = sibling_updates[local_index]

        shared_photo_urls = detect_shared_photo_urls(receipt_items)
        for photo_url in shared_photo_urls:
            evidence = analyzer.analyze_shared_photo(photo_url, receipt_items)
            if not evidence.success or not evidence.candidates:
                continue
            current_receipt_results = [updated_results[index] for index in indices]
            assignments = assign_visible_candidates_to_receipt_items(
                receipt_items,
                current_receipt_results,
                evidence,
            )
            for assignment in assignments:
                if not assignment.should_update_result:
                    continue
                for result_index in indices:
                    if updated_results[result_index].item_index == assignment.item_index:
                        updated_results[result_index] = apply_receipt_assignment(
                            updated_results[result_index],
                            assignment,
                            evidence,
                        )
                        break

    return updated_results


def build_catalog_receipt_context(items: Sequence[ReceiptItem]) -> CatalogReceiptContext:
    if not items:
        return CatalogReceiptContext(0.0, 0.0, False, 0.0, 0.0)

    item_count = len(items)
    item_id_coverage = _ratio(item.has_item_id for item in items)
    named_title_ratio = _ratio(_is_named_catalog_title(item) for item in items)
    education_pattern_ratio = _ratio(_has_catalog_education_signal(item) for item in items)
    id_family_ratio = _item_id_format_similarity(items)
    generic_ratio = _ratio(item.is_generic_title for item in items)

    merchant_catalog_confidence = clamp_confidence(
        round(
            (item_id_coverage * 0.3)
            + (named_title_ratio * 0.3)
            + (education_pattern_ratio * 0.2)
            + (id_family_ratio * 0.15)
            + ((1.0 if item_count >= 4 else 0.0) * 0.05),
            2,
        )
    )
    receipt_coherence_score = clamp_confidence(
        round(
            (named_title_ratio * 0.35)
            + (education_pattern_ratio * 0.3)
            + (id_family_ratio * 0.2)
            + ((1.0 - generic_ratio) * 0.15),
            2,
        )
    )
    notes = []
    if item_id_coverage >= 0.7:
        notes.append("Most receipt lines have item IDs.")
    if named_title_ratio >= 0.7:
        notes.append("Most receipt lines look like named catalog titles.")
    if education_pattern_ratio >= 0.25:
        notes.append("Sibling titles share curriculum/book/product structure.")
    if id_family_ratio >= 0.5:
        notes.append("Sibling item IDs use similar formats.")

    return CatalogReceiptContext(
        merchant_catalog_confidence=merchant_catalog_confidence,
        receipt_coherence_score=receipt_coherence_score,
        catalog_like_receipt=merchant_catalog_confidence >= 0.68 and receipt_coherence_score >= 0.62,
        sibling_named_title_ratio=round(named_title_ratio, 2),
        item_id_coverage=round(item_id_coverage, 2),
        notes=notes,
    )


def compute_merchant_catalog_confidence(items: Sequence[ReceiptItem], merchant: str) -> float:
    merchant_items = [item for item in items if item.merchant == merchant]
    return build_catalog_receipt_context(merchant_items).merchant_catalog_confidence


def compute_receipt_coherence(items: Sequence[ReceiptItem]) -> float:
    return build_catalog_receipt_context(items).receipt_coherence_score


def should_promote_to_catalog_deterministic(
    item: ReceiptItem,
    result: ResolutionResult,
    catalog_context: CatalogReceiptContext,
) -> bool:
    if not catalog_context.catalog_like_receipt:
        return False
    if result.route != "retrieval_needed":
        return False
    if item.is_generic_title or item.looks_like_discount_line:
        return False
    if not item.has_item_id:
        return False
    if not _is_named_catalog_title(item):
        return False
    return catalog_context.sibling_named_title_ratio >= 0.65 and catalog_context.item_id_coverage >= 0.65


def apply_catalog_context_promotion(
    item: ReceiptItem,
    result: ResolutionResult,
    catalog_context: CatalogReceiptContext,
) -> ResolutionResult:
    if not should_promote_to_catalog_deterministic(item, result, catalog_context):
        return result

    title = result.resolved_title or item.cleaned_item_name or item.item_name
    description = result.resolved_description or (
        "Likely catalog item resolved from named title, item ID, and coherent receipt context."
    )
    confidence = max(result.confidence, 0.78)
    if result.status == "resolved":
        confidence = max(result.confidence, 0.81)

    evidence = dict(result.evidence)
    evidence["candidate_basis"] = "catalog_context"
    evidence["used_item_id"] = True
    evidence["used_title_match"] = True
    evidence["used_receipt_context"] = True
    evidence["catalog_context"] = {
        "merchant_catalog_confidence": catalog_context.merchant_catalog_confidence,
        "receipt_coherence_score": catalog_context.receipt_coherence_score,
        "sibling_named_title_ratio": catalog_context.sibling_named_title_ratio,
        "item_id_coverage": catalog_context.item_id_coverage,
    }
    notes = list(result.receipt_level_notes)
    notes.append("Catalog-like receipt context supported local resolution for this named item.")
    notes.extend(catalog_context.notes)

    return replace(
        result,
        resolved_title=title,
        resolved_description=description,
        resolved_entity_type=result.resolved_entity_type or ("book" if _has_catalog_education_signal(item) else "product"),
        confidence=round(confidence, 2),
        status="resolved",
        route="deterministic",
        evidence=evidence,
        notes=None,
        receipt_context_used=True,
        receipt_level_assignment_used=True,
        receipt_level_assignment_confidence=round(confidence, 2),
        receipt_level_assignment_basis="catalog_context_promotion",
        receipt_level_notes=notes,
    )


def build_sibling_contexts(
    items: Sequence[ReceiptItem],
    results: Sequence[ResolutionResult],
) -> dict[int, SiblingContext]:
    contexts: dict[int, SiblingContext] = {}
    for item in items:
        siblings = [other for other in items if other.item_index != item.item_index]
        sibling_results = [result for result in results if result.item_index != item.item_index]
        resolved_titles = [
            result.resolved_title
            for result in sibling_results
            if result.status == "resolved" and result.resolved_title
        ]
        resolved_categories = [
            result.resolved_entity_type
            for result in sibling_results
            if result.status == "resolved" and result.resolved_entity_type
        ]
        name_patterns = sorted({_name_pattern(sibling.normalized_item_name or sibling.item_name) for sibling in siblings})
        id_signal = compute_item_id_family_signal(item, siblings)
        notes = []
        if resolved_titles:
            notes.append("Receipt has resolved sibling items.")
        if id_signal:
            notes.append("Sibling item IDs show weak family similarity.")
        contexts[item.item_index] = SiblingContext(
            receipt_index=item.receipt_index,
            merchant=item.merchant,
            sibling_count=len(siblings),
            resolved_sibling_titles=resolved_titles,
            resolved_sibling_categories=sorted(set(resolved_categories)),
            sibling_item_name_patterns=name_patterns,
            sibling_item_id_patterns=id_signal,
            notes=notes,
        )
    return contexts


def compute_item_id_family_signal(item: ReceiptItem, siblings: Sequence[ReceiptItem]) -> list[str]:
    """Return weak item-ID family indicators without inferring exact identity."""

    item_id = _digits(item.item_id)
    if not item_id:
        return []
    signals = set()
    for sibling in siblings:
        sibling_id = _digits(sibling.item_id)
        if not sibling_id or len(sibling_id) < 6 or len(item_id) < 6:
            continue
        if item_id[:6] == sibling_id[:6]:
            signals.add("same_prefix")
        try:
            if abs(int(item_id) - int(sibling_id)) <= 50:
                signals.add("close_numeric_range")
        except ValueError:
            pass
    if signals:
        signals.add("possibly_related_family")
    return sorted(signals)


def apply_sibling_family_context(
    items: Sequence[ReceiptItem],
    results: Sequence[ResolutionResult],
    sibling_contexts: dict[int, SiblingContext],
    adjudicator: SiblingContextAdjudicator,
) -> list[ResolutionResult]:
    updated_results = list(results)
    for index, (item, result) in enumerate(zip(items, results)):
        candidate_title, candidate_description, candidate_category = _plausible_candidate_from_result_or_item(item, result)
        if not candidate_title:
            continue
        sibling_context = sibling_contexts.get(item.item_index)
        if sibling_context is None or not sibling_context.resolved_sibling_titles:
            continue
        if not _should_invoke_sibling_adjudication(item, result):
            continue
        adjudication = adjudicator.adjudicate(item, result, sibling_context)
        if not can_promote_with_sibling_context(item, result, adjudication, candidate_title):
            if _can_increase_with_sibling_context(item, result, adjudication, candidate_title):
                updated_results[index] = _apply_sibling_confidence_update(result, adjudication)
            continue

        updated_results[index] = _apply_sibling_family_promotion(
            result,
            candidate_title=candidate_title,
            candidate_description=candidate_description,
            candidate_category=candidate_category,
            adjudication=adjudication,
            sibling_context=sibling_context,
        )
    return updated_results


def can_promote_with_sibling_context(
    item: ReceiptItem,
    current_result: ResolutionResult,
    adjudication: SiblingAdjudicationResult,
    candidate_title: str | None,
) -> bool:
    if current_result.status != "ambiguous":
        return False
    if not candidate_title:
        return False
    if item.is_generic_title or item.looks_like_discount_line:
        return False
    if current_result.adjudication_contradiction_detected and current_result.adjudication_contradiction_strength == "strong":
        return False
    return (
        adjudication.success
        and adjudication.family_consistent
        and adjudication.can_promote_to_resolved
        and adjudication.sibling_support_strength in {"moderate", "strong"}
    )


def _can_increase_with_sibling_context(
    item: ReceiptItem,
    current_result: ResolutionResult,
    adjudication: SiblingAdjudicationResult,
    candidate_title: str | None,
) -> bool:
    if current_result.status != "ambiguous" or not candidate_title:
        return False
    if item.is_generic_title or item.looks_like_discount_line:
        return False
    if current_result.adjudication_contradiction_detected and current_result.adjudication_contradiction_strength == "strong":
        return False
    return (
        adjudication.success
        and adjudication.family_consistent
        and adjudication.should_increase_confidence
        and adjudication.sibling_support_strength in {"weak", "moderate", "strong"}
    )


def fuse_with_sibling_context(
    current_confidence: float,
    confidence_delta: float,
    family_consistent: bool,
    has_plausible_candidate: bool,
) -> float:
    if not family_consistent or not has_plausible_candidate:
        return current_confidence
    boost = min(0.2, max(0.0, confidence_delta))
    return clamp_confidence(current_confidence + boost)


def _apply_sibling_confidence_update(
    result: ResolutionResult,
    adjudication: SiblingAdjudicationResult,
) -> ResolutionResult:
    updated_confidence = fuse_with_sibling_context(
        result.confidence,
        adjudication.confidence_delta,
        family_consistent=adjudication.family_consistent,
        has_plausible_candidate=True,
    )
    evidence = dict(result.evidence)
    evidence["used_receipt_context"] = True
    evidence["used_sibling_context"] = True
    evidence["sibling_context"] = _sibling_adjudication_evidence(adjudication)
    notes = list(result.receipt_level_notes)
    notes.append("Receipt-level sibling context increased confidence, but did not override direct evidence.")
    notes.extend(adjudication.notes)
    if adjudication.rationale:
        notes.append(adjudication.rationale)
    return replace(
        result,
        confidence=updated_confidence,
        evidence=evidence,
        receipt_context_used=True,
        sibling_context_used=True,
        sibling_similarity_score=_support_strength_score(adjudication.sibling_support_strength),
        family_consistent_with_siblings=adjudication.family_consistent,
        receipt_level_notes=notes,
    )


def _apply_sibling_family_promotion(
    result: ResolutionResult,
    *,
    candidate_title: str,
    candidate_description: str | None,
    candidate_category: str | None,
    adjudication: SiblingAdjudicationResult,
    sibling_context: SiblingContext,
) -> ResolutionResult:
    updated_confidence = max(
        fuse_with_sibling_context(
            result.confidence,
            adjudication.confidence_delta,
            family_consistent=adjudication.family_consistent,
            has_plausible_candidate=True,
        ),
        0.72,
    )
    evidence = dict(result.evidence)
    used_photo_candidate = result.photo_model_suggested_title == candidate_title
    evidence["used_receipt_context"] = True
    evidence["used_sibling_context"] = True
    if used_photo_candidate:
        evidence["used_photo"] = True
    evidence["candidate_basis"] = "sibling_family_context"
    evidence["sibling_context"] = _sibling_adjudication_evidence(adjudication)
    evidence["sibling_context"]["resolved_sibling_titles"] = sibling_context.resolved_sibling_titles[:5]
    notes = list(result.receipt_level_notes)
    notes.append("Sibling item on the same receipt strongly supports the same product family.")
    notes.extend(adjudication.notes)
    if adjudication.rationale:
        notes.append(adjudication.rationale)
    return replace(
        result,
        resolved_title=candidate_title,
        resolved_description=candidate_description
        or "Likely item resolved from direct title/photo evidence supported by same-receipt sibling context.",
        resolved_entity_type=candidate_category or result.resolved_entity_type or "product",
        confidence=updated_confidence,
        status="resolved",
        route="deterministic" if result.route == "retrieval_needed" else result.route,
        evidence=evidence,
        notes=None,
        photo_evidence_used=result.photo_evidence_used or used_photo_candidate,
        photo_result_changed=result.photo_result_changed or used_photo_candidate,
        receipt_context_used=True,
        sibling_context_used=True,
        sibling_similarity_score=_support_strength_score(adjudication.sibling_support_strength),
        family_consistent_with_siblings=adjudication.family_consistent,
        sibling_context_changed_status=result.status != "resolved",
        receipt_level_assignment_used=True,
        receipt_level_assignment_confidence=updated_confidence,
        receipt_level_assignment_basis="sibling_family_context",
        receipt_level_notes=notes,
    )


def detect_shared_photo_urls(items: Sequence[ReceiptItem]) -> list[str]:
    counts = Counter(
        photo_url
        for item in items
        for photo_url in item.reference_photo_urls
        if photo_url
    )
    return sorted(photo_url for photo_url, count in counts.items() if count > 1)


def assign_visible_candidates_to_receipt_items(
    items: Sequence[ReceiptItem],
    current_results: Sequence[ResolutionResult],
    shared_photo_evidence: SharedPhotoEvidence,
) -> list[ReceiptLevelAssignment]:
    assignments: list[ReceiptLevelAssignment] = []
    assigned_candidate_indices: set[int] = set()
    assigned_item_indices: set[int] = set()

    for item, result in zip(items, current_results):
        if item.is_generic_title:
            continue
        best_index, best_score = _best_candidate_for_item(item, shared_photo_evidence.candidates)
        if best_index is None or best_score < 0.5:
            continue
        assigned_candidate_indices.add(best_index)
        assigned_item_indices.add(item.item_index)
        candidate = shared_photo_evidence.candidates[best_index]
        assignments.append(
            ReceiptLevelAssignment(
                item_index=item.item_index,
                assigned_candidate_title=candidate.title,
                assigned_description=candidate.description,
                assigned_category=candidate.category,
                assigned_confidence=round(min(candidate.confidence, 0.86), 2),
                assignment_basis="shared_photo_text_alignment",
                should_update_result=result.status != "resolved" and candidate.confidence >= 0.72,
                notes=["Shared photo candidate aligned with receipt text."],
            )
        )

    leftover_candidates = [
        (index, candidate)
        for index, candidate in enumerate(shared_photo_evidence.candidates)
        if index not in assigned_candidate_indices and candidate.confidence >= 0.72 and candidate.title
    ]
    generic_items = [
        (item, result)
        for item, result in zip(items, current_results)
        if item.item_index not in assigned_item_indices
        and result.status != "resolved"
        and item.is_generic_title
    ]

    for item, result in generic_items:
        coherent = [
            (index, candidate)
            for index, candidate in leftover_candidates
            if _candidate_matches_generic_line(candidate, item)
        ]
        if len(coherent) != 1:
            continue
        candidate_index, candidate = coherent[0]
        leftover_candidates = [
            (index, leftover_candidate)
            for index, leftover_candidate in leftover_candidates
            if index != candidate_index
        ]
        assignments.append(
            ReceiptLevelAssignment(
                item_index=item.item_index,
                assigned_candidate_title=candidate.title,
                assigned_description=candidate.description,
                assigned_category=candidate.category,
                assigned_confidence=round(min(candidate.confidence, 0.78), 2),
                assignment_basis="shared_photo_leftover_generic_line",
                should_update_result=True,
                notes=["One coherent leftover visible candidate explained a generic receipt line."],
            )
        )

    return assignments


def apply_receipt_assignment(
    result: ResolutionResult,
    assignment: ReceiptLevelAssignment,
    shared_photo_evidence: SharedPhotoEvidence,
) -> ResolutionResult:
    if not assignment.should_update_result or not assignment.assigned_candidate_title:
        return result

    evidence = dict(result.evidence)
    evidence["used_receipt_context"] = True
    evidence["used_shared_photo"] = True
    evidence["shared_photo_analyzer"] = shared_photo_evidence.analyzer_name
    evidence["candidate_basis"] = assignment.assignment_basis
    evidence["initial_route"] = result.route
    evidence["effective_route_reason"] = "Resolved from shared reference photo evidence at the receipt level."
    notes = list(result.receipt_level_notes)
    notes.extend(assignment.notes)
    if shared_photo_evidence.notes:
        notes.append(shared_photo_evidence.notes)

    return replace(
        result,
        resolved_title=assignment.assigned_candidate_title,
        resolved_description=assignment.assigned_description,
        resolved_entity_type=assignment.assigned_category or "product",
        confidence=assignment.assigned_confidence,
        status="resolved",
        route="photo_assisted",
        evidence=evidence,
        notes=None,
        receipt_context_used=True,
        shared_photo_used=True,
        receipt_level_assignment_used=True,
        receipt_level_assignment_confidence=assignment.assigned_confidence,
        receipt_level_assignment_basis=assignment.assignment_basis,
        receipt_level_notes=notes,
    )


def build_shared_photo_prompt(receipt_context: Sequence[ReceiptItem]) -> str:
    lines = [
        "This image may contain multiple purchased items from one receipt.",
        "Extract all distinct visible products/items that are visually grounded.",
        "Only return candidates with visible evidence. Do not guess unseen items.",
        "Do not assume every visible item belongs to the receipt.",
        "Be conservative and prefer omissions over hallucinated products.",
        "",
        "Receipt context:",
    ]
    for item in receipt_context:
        lines.append(
            f"- item_index={item.item_index}; merchant={item.merchant}; "
            f"item_name={item.item_name}; item_id={item.item_id}; item_price={item.item_price}"
        )
    lines.extend(
        [
            "",
            "Return only a JSON object with this shape:",
            "{",
            '  "candidates": [',
            "    {",
            '      "title": string or null,',
            '      "description": string or null,',
            '      "category": string or null,',
            '      "author_or_brand": string or null,',
            '      "confidence": number between 0 and 1,',
            '      "signals": array of short strings',
            "    }",
            "  ],",
            '  "notes": string or null',
            "}",
        ]
    )
    return "\n".join(lines)


def build_sibling_adjudication_prompt(
    item_context: ReceiptItem,
    current_result: ResolutionResult,
    sibling_context: SiblingContext,
) -> str:
    lines = [
        "You are adjudicating same-receipt sibling context for exact-item resolution.",
        "Use semantic reasoning, not exact keyword overlap.",
        "Sibling items are supporting evidence only. Do not invent a candidate from sibling context.",
        "Only strengthen the current candidate if it is already plausible from text/photo/context.",
        "Preserve ambiguity if support is weak or if there is a strong contradiction from direct evidence.",
        "",
        "Current item:",
        json.dumps(
            {
                "merchant": item_context.merchant,
                "item_name": item_context.item_name,
                "cleaned_item_name": item_context.cleaned_item_name,
                "item_id": item_context.item_id,
                "item_price": item_context.item_price,
                "candidate_title": current_result.resolved_title or current_result.photo_model_suggested_title,
                "candidate_description": current_result.resolved_description
                or current_result.photo_model_suggested_description,
                "current_status": current_result.status,
                "current_confidence": current_result.confidence,
                "photo_evidence_summary": current_result.photo_evidence_summary,
                "adjudication_contradiction_strength": current_result.adjudication_contradiction_strength,
            },
            ensure_ascii=True,
        ),
        "",
        "Resolved sibling context:",
        json.dumps(
            {
                "receipt_index": sibling_context.receipt_index,
                "merchant": sibling_context.merchant,
                "sibling_count": sibling_context.sibling_count,
                "resolved_sibling_titles": sibling_context.resolved_sibling_titles,
                "resolved_sibling_categories": sibling_context.resolved_sibling_categories,
                "sibling_item_name_patterns": sibling_context.sibling_item_name_patterns,
                "sibling_item_id_patterns": sibling_context.sibling_item_id_patterns,
                "notes": sibling_context.notes,
            },
            ensure_ascii=True,
        ),
        "",
        "Return only a JSON object with this shape:",
        "{",
        '  "family_consistent": boolean,',
        '  "sibling_support_strength": "none" | "weak" | "moderate" | "strong",',
        '  "should_increase_confidence": boolean,',
        '  "confidence_delta": number between 0 and 0.2,',
        '  "can_promote_to_resolved": boolean,',
        '  "rationale": string,',
        '  "notes": array of short strings',
        "}",
    ]
    return "\n".join(lines)


def _best_candidate_for_item(
    item: ReceiptItem,
    candidates: Sequence[VisiblePhotoCandidate],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        candidate_text = " ".join(
            value
            for value in [candidate.title, candidate.description, candidate.category, candidate.author_or_brand]
            if value
        )
        overlap = _token_overlap(item.cleaned_item_name or item.item_name, candidate_text)
        score = round((overlap * 0.7) + (candidate.confidence * 0.3), 2)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def _plausible_candidate_from_result_or_item(
    item: ReceiptItem,
    result: ResolutionResult,
) -> tuple[str | None, str | None, str | None]:
    if result.resolved_title:
        return result.resolved_title, result.resolved_description, result.resolved_entity_type
    if result.photo_model_suggested_title and (result.photo_model_confidence or 0.0) >= 0.75:
        return result.photo_model_suggested_title, result.photo_model_suggested_description, "product"
    if result.evidence.get("used_title_match") and _is_named_catalog_title(item):
        return item.cleaned_item_name or item.item_name, None, "product"
    return None, None, None


def _should_invoke_sibling_adjudication(item: ReceiptItem, result: ResolutionResult) -> bool:
    if result.status != "ambiguous":
        return False
    if item.is_generic_title or item.looks_like_discount_line:
        return False
    if result.adjudication_contradiction_detected and result.adjudication_contradiction_strength == "strong":
        return False
    return bool(result.resolved_title or result.photo_model_suggested_title or result.evidence.get("used_title_match"))


def _sibling_adjudication_evidence(adjudication: SiblingAdjudicationResult) -> dict[str, Any]:
    return {
        "adjudicator": adjudication.adjudicator_name,
        "family_consistent": adjudication.family_consistent,
        "support_strength": adjudication.sibling_support_strength,
        "confidence_delta": adjudication.confidence_delta,
        "can_promote_to_resolved": adjudication.can_promote_to_resolved,
        "rationale": adjudication.rationale,
        "notes": adjudication.notes,
    }


def _support_strength_score(strength: str) -> float:
    return {
        "none": 0.0,
        "weak": 0.35,
        "moderate": 0.65,
        "strong": 0.9,
    }.get(strength, 0.0)


def _candidate_matches_generic_line(candidate: VisiblePhotoCandidate, item: ReceiptItem) -> bool:
    if not item.is_generic_title:
        return False
    # Generic leftover lines can use a single visually grounded leftover candidate,
    # but category semantics are intentionally left to the image model.
    return bool(candidate.title or candidate.description)


def _is_named_catalog_title(item: ReceiptItem) -> bool:
    if item.is_generic_title or item.looks_like_discount_line:
        return False
    tokens = _tokens(item.cleaned_item_name or item.item_name)
    if len(tokens) < 2:
        return False
    return item.specificity_score >= 0.28 or _has_catalog_education_signal(item)


def _has_catalog_education_signal(item: ReceiptItem) -> bool:
    title = item.normalized_item_name
    catalog_terms = {
        "answer",
        "key",
        "course",
        "book",
        "workbook",
        "guide",
        "reference",
        "level",
        "part",
        "math",
        "language",
        "arts",
        "reader",
        "readers",
        "curriculum",
        "handwriting",
    }
    return any(term in title.split() for term in catalog_terms)


def _item_id_format_similarity(items: Sequence[ReceiptItem]) -> float:
    ids = [item.normalized_item_id or item.item_id for item in items if item.has_item_id]
    if len(ids) < 2:
        return 0.0
    patterns = [_item_id_pattern(item_id) for item_id in ids if item_id]
    if not patterns:
        return 0.0
    most_common_count = Counter(patterns).most_common(1)[0][1]
    return most_common_count / len(patterns)


def _item_id_pattern(item_id: str) -> str:
    return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", item_id))


def _ratio(values: Sequence[bool] | Any) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(1 for value in value_list if value) / len(value_list)


def _name_pattern(value: str) -> str:
    tokens = _tokens(value)
    if not tokens:
        return "empty"
    return "-".join("short" if len(token) <= 3 else "word" for token in tokens[:4])


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1]


def _digits(value: str | None) -> str:
    return "".join(re.findall(r"\d+", value or ""))


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
