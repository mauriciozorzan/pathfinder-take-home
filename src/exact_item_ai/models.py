from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

RouteBucket = Literal[
    "deterministic",
    "retrieval_needed",
    "photo_assisted",
    "insufficient_evidence",
]
ResolutionStatus = Literal["resolved", "ambiguous", "insufficient_evidence"]


@dataclass(slots=True)
class ItemLatencyMetrics:
    normalization_ms: float = 0.0
    routing_ms: float = 0.0
    local_resolution_ms: float = 0.0
    photo_fetch_ms: float = 0.0
    photo_model_ms: float = 0.0
    adjudication_ms: float = 0.0
    sibling_context_ms: float = 0.0
    receipt_context_ms: float = 0.0
    total_item_ms: float = 0.0


@dataclass(slots=True)
class ReceiptLatencyMetrics:
    dataset_name: str
    receipt_index: int
    total_receipt_ms: float
    item_count: int
    items_resolved: int
    items_ambiguous: int
    items_insufficient_evidence: int
    items_with_photo_ai: int
    items_with_adjudication: int
    items_with_sibling_context: int
    external_api_call_count: int
    cached_photo_hits: int
    cached_receipt_context_hits: int


@dataclass(slots=True)
class ReceiptItem:
    dataset_name: str
    receipt_index: int
    item_index: int
    merchant: str
    item_name: str
    item_id: Optional[str]
    item_price: Optional[float]
    receipt_urls: List[str] = field(default_factory=list)
    reference_photo_urls: List[str] = field(default_factory=list)
    item_condition: Optional[str] = None
    normalized_merchant: str = ""
    normalized_item_name: str = ""
    cleaned_item_name: str = ""
    normalized_item_id: Optional[str] = None
    has_item_id: bool = False
    has_reference_photo: bool = False
    is_generic_title: bool = False
    looks_like_discount_line: bool = False
    specificity_score: float = 0.0
    id_type: Optional[str] = None
    normalization_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    bucket: RouteBucket
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResolutionResult:
    dataset_name: str
    receipt_index: int
    item_index: int
    merchant: str
    input_item_name: str
    input_item_id: Optional[str]
    input_item_price: Optional[float]
    resolved_title: Optional[str]
    resolved_description: Optional[str]
    resolved_entity_type: Optional[str]
    confidence: float
    status: ResolutionStatus
    route: RouteBucket
    evidence: Dict[str, Any]
    notes: Optional[str] = None
    condition: Optional[str] = None
    normalization_notes: List[str] = field(default_factory=list)
    photo_evidence_available: bool = False
    photo_analysis_attempted: bool = False
    photo_analysis_success: bool = False
    photo_evidence_used: bool = False
    photo_result_changed: bool = False
    photo_evidence_summary: Optional[str] = None
    photo_confidence_delta: float = 0.0
    photo_extracted_signals: List[str] = field(default_factory=list)
    photo_model_confidence: Optional[float] = None
    photo_model_suggested_title: Optional[str] = None
    photo_model_suggested_description: Optional[str] = None
    photo_model_notes: Optional[str] = None
    photo_candidate_specific: bool = False
    photo_candidate_sent_to_adjudication: bool = False
    photo_adjudication_block_reason: Optional[str] = None
    adjudication_attempted: bool = False
    adjudication_success: bool = False
    adjudication_decision: Optional[str] = None
    adjudication_rationale: Optional[str] = None
    adjudication_contradiction_detected: bool = False
    adjudication_contradiction_strength: str = "none"
    adjudication_plausible_refinement: bool = False
    adjudication_image_as_high_weight_evidence: bool = False
    adjudication_refinement_rationale: Optional[str] = None
    adjudication_photo_refinement_strength: str = "none"
    adjudication_supports_exact_resolution: bool = False
    adjudication_evidence_summary: List[str] = field(default_factory=list)
    receipt_context_used: bool = False
    sibling_context_used: bool = False
    sibling_similarity_score: Optional[float] = None
    family_consistent_with_siblings: bool = False
    sibling_context_changed_status: bool = False
    shared_photo_used: bool = False
    receipt_level_assignment_used: bool = False
    receipt_level_assignment_confidence: Optional[float] = None
    receipt_level_assignment_basis: Optional[str] = None
    receipt_level_notes: List[str] = field(default_factory=list)
    latency_metrics: ItemLatencyMetrics = field(default_factory=ItemLatencyMetrics)
    used_local_only: bool = False
    used_photo_ai: bool = False
    used_adjudication: bool = False
    used_sibling_context_flag: bool = False
    used_shared_photo_assignment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("resolved_entity_type", None)
        return payload
