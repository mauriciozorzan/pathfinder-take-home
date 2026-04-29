from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional

from .adjudicate import AdjudicationResult, EvidenceAdjudicator, NoopEvidenceAdjudicator, create_default_adjudicator
from .models import ItemLatencyMetrics, ReceiptItem, ResolutionResult, RouteDecision
from .photo_assist import NoopPhotoAnalyzer, PhotoAnalyzer, PhotoEvidence, create_default_photo_analyzer
from .route import route_item
from .score import (
    ambiguous_confidence,
    build_evidence,
    clamp_confidence,
    deterministic_confidence,
    insufficient_confidence,
)


@dataclass(slots=True)
class Candidate:
    title: str
    description: str
    entity_type: str
    basis: str
    uses_item_id: bool = False


class ExactItemResolver:
    def __init__(
        self,
        *,
        photo_analyzer: PhotoAnalyzer | None = None,
        adjudicator: EvidenceAdjudicator | None = None,
        enable_photo_assist: bool = True,
    ) -> None:
        self.enable_photo_assist = enable_photo_assist
        self.photo_analyzer = photo_analyzer if photo_analyzer is not None else create_default_photo_analyzer()
        self.adjudicator = adjudicator if adjudicator is not None else create_default_adjudicator()

    def resolve_batch(self, items: Iterable[ReceiptItem]) -> List[ResolutionResult]:
        return [self.resolve_item(item) for item in items]

    def resolve_item(self, item: ReceiptItem) -> ResolutionResult:
        total_start = time.perf_counter()
        latency = ItemLatencyMetrics()
        routing_start = time.perf_counter()
        route = route_item(item)
        latency.routing_ms = round((time.perf_counter() - routing_start) * 1000, 2)
        local_start = time.perf_counter()
        local_candidate = self._candidate_from_text(item)
        latency.local_resolution_ms = round((time.perf_counter() - local_start) * 1000, 2)

        if route.bucket == "insufficient_evidence":
            result = self._insufficient_result(item, route.reason)
            result = self._attach_photo_metadata(result, item, attempted=False, photo_evidence=None)
            return self._finalize_latency(result, latency, total_start)

        if route.bucket == "photo_assisted":
            if local_candidate and item.specificity_score >= 0.72:
                result = self._resolved_result(item, route.bucket, route.reason, local_candidate)
            else:
                result = self._ambiguous_result(
                    item,
                    route.bucket,
                    route.reason,
                    "A photo is available, but text alone is not enough for a confident exact-item match.",
                )
            result = self._maybe_apply_photo_assistance(
                item,
                route_reason=route.reason,
                route_bucket=route.bucket,
                local_candidate=local_candidate,
                base_result=result,
                latency=latency,
            )
            return self._finalize_latency(result, latency, total_start)

        if local_candidate and self._should_resolve_without_retrieval(item, local_candidate, route.bucket):
            result = self._resolved_result(item, route.bucket, route.reason, local_candidate)
            result = self._relabel_local_resolution_if_needed(result, route, local_candidate)
            result = self._attach_photo_metadata(result, item, attempted=False, photo_evidence=None)
            return self._finalize_latency(result, latency, total_start)

        if route.bucket == "retrieval_needed":
            result = self._ambiguous_result(
                item,
                route.bucket,
                route.reason,
                self._retrieval_needed_note(item, route),
            )
            result = self._maybe_apply_photo_assistance(
                item,
                route_reason=route.reason,
                route_bucket=route.bucket,
                local_candidate=local_candidate,
                base_result=result,
                latency=latency,
            )
            return self._finalize_latency(result, latency, total_start)

        result = self._insufficient_result(
            item,
            "The item did not produce a stable local candidate.",
            route_bucket=route.bucket,
        )
        result = self._attach_photo_metadata(result, item, attempted=False, photo_evidence=None)
        return self._finalize_latency(result, latency, total_start)

    def _maybe_apply_photo_assistance(
        self,
        item: ReceiptItem,
        *,
        route_reason: str,
        route_bucket: str,
        local_candidate: Candidate | None,
        base_result: ResolutionResult,
        latency: ItemLatencyMetrics,
    ) -> ResolutionResult:
        should_attempt = self._should_attempt_photo_analysis(
            item,
            route_bucket=route_bucket,
            base_result=base_result,
        )
        if not should_attempt:
            return self._attach_photo_metadata(base_result, item, attempted=False, photo_evidence=None)

        photo_source = item.reference_photo_urls[0]
        photo_start = time.perf_counter()
        photo_evidence = self.photo_analyzer.analyze(photo_source, item)
        photo_elapsed = round((time.perf_counter() - photo_start) * 1000, 2)
        latency.photo_fetch_ms = round(float(getattr(self.photo_analyzer, "last_photo_fetch_ms", 0.0)), 2)
        latency.photo_model_ms = round(float(getattr(self.photo_analyzer, "last_photo_model_ms", photo_elapsed)), 2)
        if latency.photo_fetch_ms == 0.0 and latency.photo_model_ms == 0.0:
            latency.photo_model_ms = photo_elapsed
        return self._apply_photo_evidence(
            base_result,
            item=item,
            route_reason=route_reason,
            route_bucket=route_bucket,
            local_candidate=local_candidate,
            photo_evidence=photo_evidence,
            latency=latency,
        )

    def _should_attempt_photo_analysis(
        self,
        item: ReceiptItem,
        *,
        route_bucket: str,
        base_result: ResolutionResult,
    ) -> bool:
        if not self.enable_photo_assist:
            return False
        if not item.reference_photo_urls:
            return False
        if isinstance(self.photo_analyzer, NoopPhotoAnalyzer):
            return True
        if route_bucket == "photo_assisted":
            return base_result.status != "resolved" or base_result.confidence < 0.72
        if route_bucket == "retrieval_needed" and base_result.status != "resolved":
            return base_result.confidence < 0.55
        return False

    def _finalize_latency(
        self,
        result: ResolutionResult,
        latency: ItemLatencyMetrics,
        total_start: float,
    ) -> ResolutionResult:
        latency.total_item_ms = round((time.perf_counter() - total_start) * 1000, 2)
        return replace(
            result,
            latency_metrics=latency,
            used_photo_ai=result.photo_analysis_attempted and result.photo_analysis_success,
            used_adjudication=result.adjudication_attempted,
            used_sibling_context_flag=result.sibling_context_used,
            used_shared_photo_assignment=result.shared_photo_used,
            used_local_only=not (
                result.photo_analysis_attempted
                or result.adjudication_attempted
                or result.receipt_context_used
                or result.sibling_context_used
                or result.shared_photo_used
            ),
        )

    def _apply_photo_evidence(
        self,
        base_result: ResolutionResult,
        *,
        item: ReceiptItem,
        route_reason: str,
        route_bucket: str,
        local_candidate: Candidate | None,
        photo_evidence: PhotoEvidence,
        latency: ItemLatencyMetrics,
    ) -> ResolutionResult:
        result = self._attach_photo_metadata(
            base_result,
            item,
            attempted=True,
            photo_evidence=photo_evidence,
        )

        if not photo_evidence.success:
            return replace(
                result,
                photo_candidate_specific=False,
                photo_candidate_sent_to_adjudication=False,
                photo_adjudication_block_reason="Photo analysis did not return usable evidence.",
            )

        should_adjudicate, block_reason = self.should_adjudicate_photo_result(
            item,
            result,
            photo_evidence,
        )
        result = replace(
            result,
            photo_candidate_specific=self._is_specific_photo_candidate(photo_evidence),
            photo_candidate_sent_to_adjudication=should_adjudicate,
            photo_adjudication_block_reason=None if should_adjudicate else block_reason,
        )
        if not should_adjudicate:
            return result

        adjudication_start = time.perf_counter()
        adjudication = self.adjudicator.adjudicate(item, result, photo_evidence)
        latency.adjudication_ms = round((time.perf_counter() - adjudication_start) * 1000, 2)
        return self._apply_adjudication_result(
            result,
            route_reason=route_reason,
            photo_evidence=photo_evidence,
            adjudication=adjudication,
        )

    def _apply_adjudication_result(
        self,
        result: ResolutionResult,
        *,
        route_reason: str,
        photo_evidence: PhotoEvidence,
        adjudication: AdjudicationResult,
    ) -> ResolutionResult:
        updated_title = result.resolved_title
        updated_description = result.resolved_description
        updated_entity_type = result.resolved_entity_type
        updated_status = result.status
        updated_confidence = result.confidence
        used_photo = False
        changed = False

        strong_contradiction = (
            adjudication.contradiction_detected
            and adjudication.contradiction_strength == "strong"
        )

        if adjudication.success and not strong_contradiction:
            can_use_photo = adjudication.should_use_photo_result or adjudication.should_refine_existing_result
            if can_use_photo and adjudication.final_confidence >= 0.5:
                updated_title = adjudication.adjudicated_title or updated_title
                updated_description = adjudication.adjudicated_description or updated_description
                if updated_entity_type is None and updated_title:
                    updated_entity_type = "product"
                updated_confidence = clamp_confidence(adjudication.final_confidence)
                updated_status = adjudication.final_status
                used_photo = bool(adjudication.should_use_photo_result or adjudication.should_refine_existing_result)

            strong_photo_refinement = self._is_strong_photo_refinement(photo_evidence, adjudication)
            if strong_photo_refinement and updated_title:
                updated_status = "resolved"
                updated_confidence = max(
                    updated_confidence,
                    photo_evidence.model_confidence or 0.0,
                    0.82,
                )
                used_photo = True

            if updated_status == "resolved" and (
                not strong_photo_refinement
                and (
                    not (
                        adjudication.should_use_photo_result
                        or (
                            adjudication.should_refine_existing_result
                            and adjudication.plausible_refinement
                            and adjudication.image_as_high_weight_evidence
                        )
                    )
                    or adjudication.final_confidence < 0.72
                    or not updated_title
                )
            ):
                updated_status = "ambiguous"

        changed = (
            updated_title != result.resolved_title
            or updated_description != result.resolved_description
            or updated_entity_type != result.resolved_entity_type
            or updated_status != result.status
            or updated_confidence != result.confidence
        )

        evidence = dict(result.evidence)
        evidence["used_photo"] = used_photo
        evidence["route_reason"] = route_reason
        evidence["photo_analyzer"] = photo_evidence.analyzer_name
        evidence["photo_notes"] = photo_evidence.notes
        evidence["adjudicator"] = adjudication.adjudicator_name
        evidence["adjudication_decision"] = adjudication.decision
        if used_photo:
            evidence["candidate_basis"] = "photo_adjudicated"

        return ResolutionResult(
            dataset_name=result.dataset_name,
            receipt_index=result.receipt_index,
            item_index=result.item_index,
            merchant=result.merchant,
            input_item_name=result.input_item_name,
            input_item_id=result.input_item_id,
            input_item_price=result.input_item_price,
            resolved_title=updated_title,
            resolved_description=updated_description,
            resolved_entity_type=updated_entity_type,
            confidence=updated_confidence,
            status=updated_status,
            route=result.route,
            evidence=evidence,
            notes=result.notes if updated_status == result.status else None,
            condition=result.condition,
            normalization_notes=list(result.normalization_notes),
            photo_evidence_available=True,
            photo_analysis_attempted=True,
            photo_analysis_success=photo_evidence.success,
            photo_evidence_used=used_photo,
            photo_result_changed=changed,
            photo_evidence_summary=photo_evidence.summary,
            photo_confidence_delta=round(updated_confidence - result.confidence, 2),
            photo_extracted_signals=list(photo_evidence.extracted_signals),
            photo_model_confidence=photo_evidence.model_confidence,
            photo_model_suggested_title=photo_evidence.model_suggested_title,
            photo_model_suggested_description=photo_evidence.model_suggested_description,
            photo_model_notes=photo_evidence.model_notes,
            photo_candidate_specific=result.photo_candidate_specific,
            photo_candidate_sent_to_adjudication=True,
            photo_adjudication_block_reason=None,
            adjudication_attempted=True,
            adjudication_success=adjudication.success,
            adjudication_decision=adjudication.decision,
            adjudication_rationale=adjudication.rationale,
            adjudication_contradiction_detected=adjudication.contradiction_detected,
            adjudication_contradiction_strength=adjudication.contradiction_strength,
            adjudication_plausible_refinement=adjudication.plausible_refinement,
            adjudication_image_as_high_weight_evidence=adjudication.image_as_high_weight_evidence,
            adjudication_refinement_rationale=adjudication.refinement_rationale,
            adjudication_photo_refinement_strength=adjudication.photo_refinement_strength,
            adjudication_supports_exact_resolution=adjudication.supports_exact_resolution,
            adjudication_evidence_summary=list(adjudication.evidence_summary),
        )

    def should_adjudicate_photo_result(
        self,
        item: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> tuple[bool, str | None]:
        if current_result.adjudication_contradiction_detected and current_result.adjudication_contradiction_strength == "strong":
            return False, "A strong contradiction was already established before photo adjudication."
        if current_result.status == "resolved" and current_result.confidence >= 0.72:
            return False, "Current result is already resolved with sufficient confidence."
        if not photo_evidence.success:
            return False, "Photo analysis did not succeed."
        if photo_evidence.used:
            return True, None
        if self._is_specific_photo_candidate(photo_evidence):
            return True, None
        return False, "Photo result did not contain a sufficiently specific high-confidence candidate."

    def _is_specific_photo_candidate(self, photo_evidence: PhotoEvidence) -> bool:
        title = photo_evidence.model_suggested_title or photo_evidence.suggested_title
        if not title:
            return False
        model_confidence = photo_evidence.model_confidence or 0.0
        if model_confidence < 0.75:
            return False
        normalized_tokens = [token for token in title.replace("-", " ").replace("/", " ").split() if token]
        has_specific_shape = len(normalized_tokens) >= 3 or any(any(char.isdigit() for char in token) for token in normalized_tokens)
        has_visual_grounding = bool(photo_evidence.extracted_signals or photo_evidence.summary or photo_evidence.model_notes)
        return photo_evidence.is_sufficient_for_exact_identification or (
            model_confidence >= 0.75 and has_specific_shape and has_visual_grounding
        )

    def _is_strong_photo_refinement(
        self,
        photo_evidence: PhotoEvidence,
        adjudication: AdjudicationResult,
    ) -> bool:
        if adjudication.contradiction_detected and adjudication.contradiction_strength == "strong":
            return False
        if not adjudication.plausible_refinement or not adjudication.image_as_high_weight_evidence:
            return False
        if not (photo_evidence.model_suggested_title or photo_evidence.suggested_title):
            return False
        model_confidence = photo_evidence.model_confidence or 0.0
        explicit_support = (
            adjudication.supports_exact_resolution
            or adjudication.photo_refinement_strength == "strong"
        )
        return explicit_support or (
            model_confidence >= 0.85
            and photo_evidence.is_sufficient_for_exact_identification
            and adjudication.contradiction_strength in {"none", "weak"}
        )

    def _should_resolve_without_retrieval(
        self,
        item: ReceiptItem,
        candidate: Candidate,
        route_bucket: str,
    ) -> bool:
        if route_bucket == "deterministic":
            return True
        if route_bucket == "photo_assisted":
            return item.specificity_score >= 0.72
        if route_bucket != "retrieval_needed":
            return False
        if candidate.basis in {"isbn_and_title", "book_like_title"} and item.specificity_score >= 0.58:
            return True
        if candidate.basis == "book_like_title" and item.item_condition and item.cleaned_item_name:
            return True
        if candidate.basis in {"service_title", "subscription_title"} and (
            item.has_item_id or item.specificity_score >= 0.7
        ):
            return True
        if candidate.basis == "high_specificity_title":
            return True
        return candidate.uses_item_id and item.specificity_score >= 0.68

    def _relabel_local_resolution_if_needed(
        self,
        result: ResolutionResult,
        route: RouteDecision,
        candidate: Candidate,
    ) -> ResolutionResult:
        if route.bucket != "retrieval_needed":
            return result
        if result.status != "resolved":
            return result
        if result.photo_evidence_used or result.adjudication_attempted:
            return result
        if candidate.basis not in {
            "isbn_and_title",
            "book_like_title",
            "service_title",
            "subscription_title",
            "high_specificity_title",
            "specific_title",
        }:
            return result

        evidence = dict(result.evidence)
        evidence["initial_route"] = route.bucket
        evidence["effective_route_reason"] = (
            "Resolved locally from title, merchant, and structured item evidence without retrieval."
        )
        return replace(result, route="deterministic", evidence=evidence)

    def _retrieval_needed_note(self, item: ReceiptItem, route: RouteDecision) -> str:
        if item.has_reference_photo:
            return (
                "The available text suggests a plausible item, but additional external or "
                "stronger image-backed evidence would be needed for a confident exact-item identification."
            )
        if item.has_item_id:
            return (
                "The item may be identifiable with additional catalog or lookup evidence, but "
                "the current local evidence is not sufficient for a confident match."
            )
        if self._looks_abbreviated_or_truncated(item):
            return (
                "The receipt line appears abbreviated or truncated, so the current evidence is "
                "not specific enough to identify an exact item confidently."
            )
        if item.specificity_score >= 0.45 or "some signal" in route.reason.lower():
            return (
                "Available evidence suggests a plausible item family, but not enough specificity "
                "for a confident exact-item match."
            )
        return "The current evidence is not specific enough to support a confident exact-item identification."

    def _looks_abbreviated_or_truncated(self, item: ReceiptItem) -> bool:
        tokens = item.normalized_item_name.split()
        if not tokens:
            return False
        short_token_count = sum(1 for token in tokens if len(token) <= 3)
        return any(len(token) == 1 for token in tokens) or short_token_count >= max(2, len(tokens) - 1)

    def _candidate_from_text(self, item: ReceiptItem) -> Optional[Candidate]:
        title = item.cleaned_item_name or item.item_name
        if item.is_generic_title or item.looks_like_discount_line:
            return None

        if item.id_type == "isbn13":
            return Candidate(
                title=title,
                description="Likely book resolved from a specific title with ISBN-backed evidence.",
                entity_type="book",
                basis="isbn_and_title",
                uses_item_id=True,
            )

        if self._looks_like_book(item):
            return Candidate(
                title=title,
                description="Likely book or curriculum title resolved from merchant and title evidence.",
                entity_type="book",
                basis="book_like_title",
                uses_item_id=bool(item.has_item_id),
            )

        if self._looks_like_service(item):
            return Candidate(
                title=title,
                description="Likely service or educational program resolved from a specific service title.",
                entity_type="service",
                basis="service_title",
                uses_item_id=bool(item.has_item_id),
            )

        if self._looks_like_subscription(item):
            return Candidate(
                title=title,
                description="Likely subscription or membership product resolved from merchant and title evidence.",
                entity_type="subscription",
                basis="subscription_title",
                uses_item_id=bool(item.has_item_id),
            )

        if self._is_high_specificity_product_title(item):
            return Candidate(
                title=title,
                description=(
                    "Likely product resolved because the title is specific enough to function "
                    "as an exact item identity in this merchant context."
                ),
                entity_type="product",
                basis="high_specificity_title",
                uses_item_id=bool(item.has_item_id),
            )

        if item.specificity_score >= 0.62 or item.has_item_id:
            return Candidate(
                title=title,
                description="Likely product resolved from a specific receipt title and merchant context.",
                entity_type="product",
                basis="specific_title",
                uses_item_id=bool(item.has_item_id),
            )

        return None

    def _is_high_specificity_product_title(self, item: ReceiptItem) -> bool:
        if item.is_generic_title or item.looks_like_discount_line:
            return False
        title = item.cleaned_item_name or item.item_name
        tokens = item.normalized_item_name.split()
        if len(tokens) < 3:
            return False
        titlecase_tokens = [
            token.strip(":-/,()")
            for token in title.split()
            if token.strip(":-/,()")
        ]
        titlecase_ratio = sum(token[:1].isupper() for token in titlecase_tokens) / max(len(titlecase_tokens), 1)
        has_model_or_catalog_code = any(any(char.isdigit() for char in token) for token in tokens)
        has_structured_title = any(separator in title for separator in {":", " - ", "/", ","})
        starts_like_named_work = tokens[0] in {"the", "a", "an"} and len(tokens) >= 3

        return (
            (item.specificity_score >= 0.72 and titlecase_ratio >= 0.65)
            or (has_model_or_catalog_code and len(tokens) >= 4 and titlecase_ratio >= 0.5)
            or (has_structured_title and item.specificity_score >= 0.6 and len(tokens) >= 4)
            or starts_like_named_work
        )

    def _looks_like_book(self, item: ReceiptItem) -> bool:
        merchant = item.normalized_merchant
        title = item.normalized_item_name
        lowered_title = item.cleaned_item_name.lower()
        book_merchants = {
            "barnes and noble",
            "thriftbooks",
            "teachers pay teachers",
            "veritas press",
            "rainbow resource center",
            "covenant home school resource center",
        }
        return (
            merchant in book_merchants
            or (" by " in lowered_title and "created by " not in lowered_title)
            or "workbook" in title
            or "course book" in title
            or "answer key" in title
            or "guide" in title
            or "curriculum" in title
        )

    def _looks_like_service(self, item: ReceiptItem) -> bool:
        title = item.normalized_item_name
        return any(keyword in title for keyword in {"workshop", "class", "course", "program"})

    def _looks_like_subscription(self, item: ReceiptItem) -> bool:
        title = item.normalized_item_name
        return any(keyword in title for keyword in {"subscription", "membership", "annually", "monthly"})

    def _resolved_result(
        self,
        item: ReceiptItem,
        route_bucket: str,
        route_reason: str,
        candidate: Candidate,
    ) -> ResolutionResult:
        confidence = deterministic_confidence(item, uses_item_id=candidate.uses_item_id)
        return ResolutionResult(
            dataset_name=item.dataset_name,
            receipt_index=item.receipt_index,
            item_index=item.item_index,
            merchant=item.merchant,
            input_item_name=item.item_name,
            input_item_id=item.item_id,
            input_item_price=item.item_price,
            resolved_title=candidate.title,
            resolved_description=candidate.description,
            resolved_entity_type=candidate.entity_type,
            confidence=confidence,
            status="resolved",
            route=route_bucket,
            evidence=build_evidence(
                route_reason=route_reason,
                used_item_id=candidate.uses_item_id,
                used_title_match=True,
                used_photo=False,
                used_merchant=True,
                candidate_basis=candidate.basis,
            ),
            notes=None,
            condition=item.item_condition,
            normalization_notes=list(item.normalization_notes),
        )

    def _ambiguous_result(
        self,
        item: ReceiptItem,
        route_bucket: str,
        route_reason: str,
        note: str,
    ) -> ResolutionResult:
        return ResolutionResult(
            dataset_name=item.dataset_name,
            receipt_index=item.receipt_index,
            item_index=item.item_index,
            merchant=item.merchant,
            input_item_name=item.item_name,
            input_item_id=item.item_id,
            input_item_price=item.item_price,
            resolved_title=None,
            resolved_description=None,
            resolved_entity_type=None,
            confidence=ambiguous_confidence(item),
            status="ambiguous",
            route=route_bucket,
            evidence=build_evidence(
                route_reason=route_reason,
                used_item_id=item.has_item_id,
                used_title_match=item.specificity_score >= 0.5,
                used_photo=False,
                used_merchant=True,
                candidate_basis="needs_retrieval",
            ),
            notes=note,
            condition=item.item_condition,
            normalization_notes=list(item.normalization_notes),
        )

    def _insufficient_result(
        self,
        item: ReceiptItem,
        note: str,
        route_bucket: str = "insufficient_evidence",
    ) -> ResolutionResult:
        return ResolutionResult(
            dataset_name=item.dataset_name,
            receipt_index=item.receipt_index,
            item_index=item.item_index,
            merchant=item.merchant,
            input_item_name=item.item_name,
            input_item_id=item.item_id,
            input_item_price=item.item_price,
            resolved_title=None,
            resolved_description=None,
            resolved_entity_type=None,
            confidence=insufficient_confidence(item),
            status="insufficient_evidence",
            route=route_bucket,
            evidence=build_evidence(
                route_reason=note,
                used_item_id=False,
                used_title_match=False,
                used_photo=False,
                used_merchant=False,
                candidate_basis=None,
            ),
            notes=note,
            condition=item.item_condition,
            normalization_notes=list(item.normalization_notes),
        )

    def _attach_photo_metadata(
        self,
        result: ResolutionResult,
        item: ReceiptItem,
        *,
        attempted: bool,
        photo_evidence: PhotoEvidence | None,
    ) -> ResolutionResult:
        success = photo_evidence.success if photo_evidence is not None else False
        used = photo_evidence.used if photo_evidence is not None else False
        summary = photo_evidence.summary if photo_evidence is not None else None
        delta = photo_evidence.confidence_delta if photo_evidence is not None else 0.0
        signals = list(photo_evidence.extracted_signals) if photo_evidence is not None else []
        evidence = dict(result.evidence)
        if photo_evidence is not None:
            evidence["photo_analyzer"] = photo_evidence.analyzer_name
            evidence["photo_notes"] = photo_evidence.notes

        return ResolutionResult(
            dataset_name=result.dataset_name,
            receipt_index=result.receipt_index,
            item_index=result.item_index,
            merchant=result.merchant,
            input_item_name=result.input_item_name,
            input_item_id=result.input_item_id,
            input_item_price=result.input_item_price,
            resolved_title=result.resolved_title,
            resolved_description=result.resolved_description,
            resolved_entity_type=result.resolved_entity_type,
            confidence=result.confidence,
            status=result.status,
            route=result.route,
            evidence=evidence,
            notes=result.notes,
            condition=result.condition,
            normalization_notes=list(result.normalization_notes),
            photo_evidence_available=item.has_reference_photo,
            photo_analysis_attempted=attempted,
            photo_analysis_success=success,
            photo_evidence_used=used,
            photo_result_changed=False,
            photo_evidence_summary=summary,
            photo_confidence_delta=delta,
            photo_extracted_signals=signals,
            photo_model_confidence=photo_evidence.model_confidence if photo_evidence is not None else None,
            photo_model_suggested_title=photo_evidence.model_suggested_title if photo_evidence is not None else None,
            photo_model_suggested_description=photo_evidence.model_suggested_description if photo_evidence is not None else None,
            photo_model_notes=photo_evidence.model_notes if photo_evidence is not None else None,
            photo_candidate_specific=self._is_specific_photo_candidate(photo_evidence) if photo_evidence is not None else False,
            photo_candidate_sent_to_adjudication=False,
            photo_adjudication_block_reason=None,
            adjudication_attempted=False,
            adjudication_success=False,
            adjudication_decision=None,
            adjudication_rationale=None,
            adjudication_contradiction_detected=False,
            adjudication_contradiction_strength="none",
            adjudication_plausible_refinement=False,
            adjudication_image_as_high_weight_evidence=False,
            adjudication_refinement_rationale=None,
            adjudication_evidence_summary=[],
        )
