"""Latency measurement, caching wrappers, and latency-report generation."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .adjudicate import AdjudicationResult, EvidenceAdjudicator
from .models import ReceiptItem, ReceiptLatencyMetrics, ResolutionResult
from .photo_assist import PhotoAnalyzer, PhotoEvidence
from .receipt_context import SharedPhotoAnalyzer, SharedPhotoEvidence


def elapsed_ms(start: float) -> float:
    """Return elapsed wall-clock milliseconds since a perf_counter start time."""

    return round((time.perf_counter() - start) * 1000, 2)


def percentile(values: Sequence[float], percent: float) -> float:
    """Compute a simple nearest-rank percentile for report-friendly timings."""

    clean = sorted(value for value in values if value is not None)
    if not clean:
        return 0.0
    index = min(len(clean) - 1, max(0, round((len(clean) - 1) * percent)))
    return round(clean[index], 2)


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Return average, median, p95, and max timing statistics."""

    clean = [value for value in values if value is not None]
    if not clean:
        return {"avg_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(sum(clean) / len(clean), 2),
        "median_ms": round(statistics.median(clean), 2),
        "p95_ms": percentile(clean, 0.95),
        "max_ms": round(max(clean), 2),
    }


class CachedPhotoAnalyzer:
    """In-memory cache for item-level photo analysis keyed by image source."""

    def __init__(self, delegate: PhotoAnalyzer) -> None:
        """Wrap a photo analyzer and expose cache/call counters for reports."""

        self.delegate = delegate
        self.cache: dict[str, PhotoEvidence] = {}
        self.cache_hits = 0
        self.external_api_call_count = 0
        self.last_cache_hit = False
        self.last_photo_fetch_ms = 0.0
        self.last_photo_model_ms = 0.0

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        """Return cached photo evidence or delegate to the wrapped analyzer."""

        if image_source in self.cache:
            self.cache_hits += 1
            self.last_cache_hit = True
            self.last_photo_fetch_ms = 0.0
            self.last_photo_model_ms = 0.0
            return self.cache[image_source]
        start = time.perf_counter()
        evidence = self.delegate.analyze(image_source, item_context)
        elapsed = elapsed_ms(start)
        self.last_cache_hit = False
        self.last_photo_fetch_ms = float(getattr(self.delegate, "last_photo_fetch_ms", 0.0))
        self.last_photo_model_ms = float(getattr(self.delegate, "last_photo_model_ms", elapsed))
        self.external_api_call_count += int(getattr(self.delegate, "last_external_api_call_count", 1 if evidence.success else 0))
        self.cache[image_source] = evidence
        return evidence


class CachedEvidenceAdjudicator:
    """In-memory cache for repeated adjudication inputs."""

    def __init__(self, delegate: EvidenceAdjudicator) -> None:
        """Wrap an adjudicator and expose cache/call counters for reports."""

        self.delegate = delegate
        self.cache: dict[str, AdjudicationResult] = {}
        self.cache_hits = 0
        self.external_api_call_count = 0
        self.last_cache_hit = False

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: ResolutionResult,
        photo_evidence: PhotoEvidence,
    ) -> AdjudicationResult:
        """Return cached adjudication output or run the wrapped adjudicator."""

        key = json.dumps(
            # The cache key includes the stable item identity plus current result
            # and photo candidate fields that can affect adjudication.
            {
                "dataset": item_context.dataset_name,
                "receipt": item_context.receipt_index,
                "item": item_context.item_index,
                "name": item_context.item_name,
                "id": item_context.item_id,
                "photo_title": photo_evidence.model_suggested_title or photo_evidence.suggested_title,
                "photo_confidence": photo_evidence.model_confidence,
                "status": current_result.status,
                "confidence": current_result.confidence,
            },
            sort_keys=True,
        )
        if key in self.cache:
            self.cache_hits += 1
            self.last_cache_hit = True
            return self.cache[key]
        result = self.delegate.adjudicate(item_context, current_result, photo_evidence)
        self.last_cache_hit = False
        self.external_api_call_count += 1 if result.adjudicator_name != "noop" else 0
        self.cache[key] = result
        return result


class CachedSharedPhotoAnalyzer:
    """In-memory cache for receipt-level shared-photo analysis."""

    def __init__(self, delegate: SharedPhotoAnalyzer) -> None:
        """Wrap a shared-photo analyzer and expose cache/call counters."""

        self.delegate = delegate
        self.cache: dict[str, SharedPhotoEvidence] = {}
        self.cache_hits = 0
        self.external_api_call_count = 0
        self.last_cache_hit = False

    def analyze_shared_photo(
        self,
        image_source: str,
        receipt_context: Sequence[ReceiptItem],
    ) -> SharedPhotoEvidence:
        """Return cached shared-photo evidence or delegate to the analyzer."""

        if image_source in self.cache:
            self.cache_hits += 1
            self.last_cache_hit = True
            return self.cache[image_source]
        evidence = self.delegate.analyze_shared_photo(image_source, receipt_context)
        self.last_cache_hit = False
        self.external_api_call_count += 1 if evidence.analyzer_name != "noop" and evidence.success else 0
        self.cache[image_source] = evidence
        return evidence


def build_latency_report(
    all_results: dict[str, Sequence[ResolutionResult]],
    receipt_metrics: Sequence[ReceiptLatencyMetrics],
    *,
    pipeline_mode: str,
    cache_stats: dict[str, int],
) -> dict[str, Any]:
    """Build the structured JSON latency report consumed by summaries and review."""

    all_items = [result for results in all_results.values() for result in results]
    item_totals = [result.latency_metrics.total_item_ms for result in all_items]
    receipt_totals = [metric.total_receipt_ms for metric in receipt_metrics]
    by_route = {
        route: latency_stats([result.latency_metrics.total_item_ms for result in all_items if result.route == route])
        for route in sorted({result.route for result in all_items})
    }
    expensive_groups = {
        "local_only": [result for result in all_items if result.used_local_only],
        "photo_ai": [result for result in all_items if result.used_photo_ai],
        "adjudication": [result for result in all_items if result.used_adjudication],
        "receipt_context": [result for result in all_items if result.receipt_context_used],
    }
    return {
        "pipeline_mode": pipeline_mode,
        "aggregate_item_latency": latency_stats(item_totals),
        "aggregate_receipt_latency": latency_stats(receipt_totals),
        "by_route": by_route,
        "by_expensive_path": {
            name: {
                "count": len(results),
                **latency_stats([result.latency_metrics.total_item_ms for result in results]),
            }
            for name, results in expensive_groups.items()
        },
        "cache_stats": cache_stats,
        "external_api_call_count": sum(cache_stats.get(name, 0) for name in cache_stats if name.endswith("_external_calls")),
        "items": [
            {
                "dataset_name": result.dataset_name,
                "receipt_index": result.receipt_index,
                "item_index": result.item_index,
                "route": result.route,
                "status": result.status,
                "latency_metrics": asdict(result.latency_metrics),
                "used_local_only": result.used_local_only,
                "used_photo_ai": result.used_photo_ai,
                "used_adjudication": result.used_adjudication,
                "used_sibling_context": result.used_sibling_context_flag,
                "used_shared_photo_assignment": result.used_shared_photo_assignment,
            }
            for result in all_items
        ],
        "receipts": [asdict(metric) for metric in receipt_metrics],
    }


def write_latency_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write the latency report JSON to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return output_path
