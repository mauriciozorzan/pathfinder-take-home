from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.request import Request, urlopen

from .models import ReceiptItem, ResolutionResult
from .photo_assist import PhotoAIConfig, PhotoEvidence
from .score import clamp_confidence

AdjudicatedStatus = Literal["resolved", "ambiguous", "insufficient_evidence"]
AdjudicationDecision = Literal["use_photo_result", "refine_existing_result", "remain_ambiguous", "contradiction"]
ContradictionStrength = Literal["none", "weak", "strong"]


@dataclass(slots=True)
class AdjudicationResult:
    """Structured decision for reconciling text-first and photo-derived evidence."""

    should_use_photo_result: bool
    should_refine_existing_result: bool
    final_status: AdjudicatedStatus
    final_confidence: float
    adjudicated_title: str | None
    adjudicated_description: str | None
    contradiction_detected: bool
    rationale: str
    evidence_summary: list[str] = field(default_factory=list)
    decision: AdjudicationDecision = "remain_ambiguous"
    plausible_refinement: bool = False
    contradiction_strength: ContradictionStrength = "none"
    image_as_high_weight_evidence: bool = False
    refinement_rationale: str | None = None
    photo_refinement_strength: Literal["none", "weak", "moderate", "strong"] = "none"
    supports_exact_resolution: bool = False
    success: bool = True
    notes: str | None = None
    adjudicator_name: str = "noop"


class EvidenceAdjudicator(Protocol):
    """Pluggable interface for model-based evidence reconciliation."""

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> AdjudicationResult:
        """Decide how text-first and photo-derived evidence should be reconciled."""


class NoopEvidenceAdjudicator:
    """Safe fallback that preserves the text-first result."""

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> AdjudicationResult:
        return AdjudicationResult(
            should_use_photo_result=False,
            should_refine_existing_result=False,
            final_status=current_result.status,
            final_confidence=current_result.confidence,
            adjudicated_title=current_result.resolved_title,
            adjudicated_description=current_result.resolved_description,
            contradiction_detected=False,
            rationale="No adjudicator configured; preserving text-first result.",
            evidence_summary=[],
            decision="remain_ambiguous",
            plausible_refinement=False,
            contradiction_strength="none",
            image_as_high_weight_evidence=False,
            refinement_rationale=None,
            photo_refinement_strength="none",
            supports_exact_resolution=False,
            success=False,
            notes="No adjudicator configured.",
            adjudicator_name="noop",
        )


class AIBasedEvidenceAdjudicator:
    """Model-backed adjudicator for receipt/photo evidence reconciliation.

    This class intentionally avoids product-specific matching logic. The model
    receives structured receipt context, current text-first result, and photo
    evidence, then returns a structured decision. Python code only validates the
    response and enforces broad fail-closed policy.
    """

    def __init__(self, config: PhotoAIConfig | None = None) -> None:
        self.config = config if config is not None else PhotoAIConfig.from_env()

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> AdjudicationResult:
        if not self.config.enabled:
            return self._disabled(current_result, "PHOTO_AI_ENABLED is not true.")
        if self.config.provider != "openai":
            return self._disabled(current_result, f"Unsupported PHOTO_AI_PROVIDER: {self.config.provider}.")
        if not self.config.api_key:
            return self._disabled(current_result, "Missing PHOTO_AI_API_KEY or OPENAI_API_KEY.")

        try:
            payload = self._call_openai_adjudicator(item_context, current_result, photo_evidence)
            return self._result_from_payload(payload, current_result)
        except Exception as exc:
            return self._disabled(current_result, f"Evidence adjudicator failed: {exc!s}")

    def _disabled(self, current_result: ResolutionResult, note: str) -> AdjudicationResult:
        return AdjudicationResult(
            should_use_photo_result=False,
            should_refine_existing_result=False,
            final_status=current_result.status,
            final_confidence=current_result.confidence,
            adjudicated_title=current_result.resolved_title,
            adjudicated_description=current_result.resolved_description,
            contradiction_detected=False,
            rationale=note,
            evidence_summary=[],
            decision="remain_ambiguous",
            plausible_refinement=False,
            contradiction_strength="none",
            image_as_high_weight_evidence=False,
            refinement_rationale=None,
            photo_refinement_strength="none",
            supports_exact_resolution=False,
            success=False,
            notes=note,
            adjudicator_name="ai_openai",
        )

    def _call_openai_adjudicator(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": build_adjudication_prompt(item_context, current_result, photo_evidence),
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

    def _result_from_payload(
        self,
        payload: dict[str, Any],
        current_result: ResolutionResult,
    ) -> AdjudicationResult:
        final_status = _status_or_default(payload.get("final_status"), current_result.status)
        decision = _decision_or_default(payload.get("decision"))
        final_confidence = clamp_confidence(_coerce_float(payload.get("final_confidence"), current_result.confidence))
        contradiction = bool(payload.get("contradiction_detected"))
        contradiction_strength = _contradiction_strength_or_default(payload.get("contradiction_strength"))
        should_use_photo = bool(payload.get("should_use_photo_result"))
        should_refine = bool(payload.get("should_refine_existing_result"))
        plausible_refinement = bool(payload.get("plausible_refinement"))
        image_as_high_weight_evidence = bool(payload.get("image_as_high_weight_evidence"))
        refinement_strength = _refinement_strength_or_default(payload.get("photo_refinement_strength"))
        supports_exact_resolution = bool(payload.get("supports_exact_resolution"))

        if contradiction and contradiction_strength == "strong":
            final_status = "ambiguous"
            should_use_photo = False
            should_refine = False
            decision = "contradiction"

        return AdjudicationResult(
            should_use_photo_result=should_use_photo,
            should_refine_existing_result=should_refine,
            final_status=final_status,
            final_confidence=final_confidence,
            adjudicated_title=_string_or_none(payload.get("adjudicated_title")),
            adjudicated_description=_string_or_none(payload.get("adjudicated_description")),
            contradiction_detected=contradiction,
            rationale=_string_or_none(payload.get("rationale")) or "No rationale returned.",
            evidence_summary=_list_of_strings(payload.get("evidence_summary")),
            decision=decision,
            plausible_refinement=plausible_refinement,
            contradiction_strength=contradiction_strength,
            image_as_high_weight_evidence=image_as_high_weight_evidence,
            refinement_rationale=_string_or_none(payload.get("refinement_rationale")),
            photo_refinement_strength=refinement_strength,
            supports_exact_resolution=supports_exact_resolution,
            success=True,
            notes=_string_or_none(payload.get("notes")),
            adjudicator_name="ai_openai",
        )


def create_default_adjudicator() -> EvidenceAdjudicator:
    config = PhotoAIConfig.from_env()
    if config.enabled:
        return AIBasedEvidenceAdjudicator(config)
    return NoopEvidenceAdjudicator()


def build_adjudication_prompt(
    item_context: ReceiptItem,
    current_result: ResolutionResult,
    photo_evidence: PhotoEvidence,
) -> str:
    return f"""You are adjudicating whether photo-derived product evidence plausibly resolves a parsed receipt line item.

The system is JSON-first and text-first. Photo evidence is optional secondary evidence.
For anchored product/reference images, treat clear packaging, visible branding, visible title words, covers, or retail product imagery as high-weight evidence of the actual purchased item's identity.
Treat receipt text as a potentially abbreviated, truncated, or partial representation of the purchased item, not as strict category truth.
Prefer resolving to the more specific image-derived title when it plausibly explains the receipt shorthand.
Do not require exact string equality between the receipt and photo.
Do not reject a photo-derived exact item merely because the image reveals a more specific retail product category than the receipt text.
Do not invent an exact product if evidence is weak.
Prefer ambiguity over fabricated precision.
Only mark contradiction when receipt evidence and image evidence are genuinely incompatible, not when the image is simply more specific.

Receipt item context:
- merchant: {item_context.merchant}
- item_name: {item_context.item_name}
- item_id: {item_context.item_id}
- item_price: {item_context.item_price}

Current text-first result:
- status: {current_result.status}
- confidence: {current_result.confidence}
- resolved_title: {current_result.resolved_title}
- resolved_description: {current_result.resolved_description}
- resolved_entity_type: {current_result.resolved_entity_type}

Photo evidence:
- success: {photo_evidence.success}
- suggested_title: {photo_evidence.model_suggested_title or photo_evidence.suggested_title}
- suggested_description: {photo_evidence.model_suggested_description or photo_evidence.suggested_description}
- model_confidence: {photo_evidence.model_confidence}
- sufficient_for_exact_identification: {photo_evidence.is_sufficient_for_exact_identification}
- summary: {photo_evidence.summary}
- signals: {photo_evidence.extracted_signals}
- notes: {photo_evidence.model_notes or photo_evidence.notes}

Answer these questions conservatively:
1. Do the receipt context and photo evidence likely refer to the same exact item?
2. Is the photo-derived identity a plausible expansion/refinement of the receipt text?
3. Is any mismatch a weak category/wording mismatch, or a strong incompatibility?
4. Should the system resolve now, refine but remain ambiguous, remain ambiguous, or mark contradiction?

Guidance:
- If the photo-derived title contains important receipt tokens or plausibly explains a shorthand receipt label, treat it as a plausible refinement.
- A word like "cake", "paper", "toy", or "SW" on a receipt may be part of a compressed title, not a strict product ontology.
- Do not mark contradiction just because a receipt token sounds category-like while the image shows a packaged retail product.
- Use contradiction_strength="strong" only when the evidence is truly incompatible.
- Use photo_refinement_strength="strong" and supports_exact_resolution=true when the image shows visually grounded branding/title/packaging and the photo-derived title plausibly explains the receipt shorthand.
- Use photo_refinement_strength="weak" when the image is only a helpful clue or has weak alignment to the receipt line.

Return only a JSON object with this shape:
{{
  "decision": "use_photo_result" | "refine_existing_result" | "remain_ambiguous" | "contradiction",
  "should_use_photo_result": boolean,
  "should_refine_existing_result": boolean,
  "final_status": "resolved" | "ambiguous" | "insufficient_evidence",
  "final_confidence": number between 0 and 1,
  "adjudicated_title": string or null,
  "adjudicated_description": string or null,
  "contradiction_detected": boolean,
  "contradiction_strength": "none" | "weak" | "strong",
  "plausible_refinement": boolean,
  "image_as_high_weight_evidence": boolean,
  "refinement_rationale": string or null,
  "photo_refinement_strength": "none" | "weak" | "moderate" | "strong",
  "supports_exact_resolution": boolean,
  "rationale": string,
  "evidence_summary": array of short strings,
  "notes": string or null
}}"""


def _status_or_default(value: Any, default: str) -> AdjudicatedStatus:
    if value in {"resolved", "ambiguous", "insufficient_evidence"}:
        return value
    if default in {"resolved", "ambiguous", "insufficient_evidence"}:
        return default  # type: ignore[return-value]
    return "ambiguous"


def _decision_or_default(value: Any) -> AdjudicationDecision:
    if value in {"use_photo_result", "refine_existing_result", "remain_ambiguous", "contradiction"}:
        return value
    return "remain_ambiguous"


def _contradiction_strength_or_default(value: Any) -> ContradictionStrength:
    if value in {"none", "weak", "strong"}:
        return value
    return "none"


def _refinement_strength_or_default(value: Any) -> Literal["none", "weak", "moderate", "strong"]:
    if value in {"none", "weak", "moderate", "strong"}:
        return value
    return "none"


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
