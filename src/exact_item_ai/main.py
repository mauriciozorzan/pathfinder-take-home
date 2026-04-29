"""Command-line entry point and orchestration for the exact-item pipeline.

The heavy lifting lives in smaller modules. This file wires those stages
together: load datasets, normalize items, resolve them, optionally apply
receipt-level context, record latency, and write output artifacts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .adjudicate import EvidenceAdjudicator, NoopEvidenceAdjudicator, create_default_adjudicator
from .io_utils import load_receipt_items, write_results, write_text
from .latency import (
    CachedEvidenceAdjudicator,
    CachedPhotoAnalyzer,
    CachedSharedPhotoAnalyzer,
    build_latency_report,
    elapsed_ms,
    latency_stats,
    write_latency_report,
)
from .models import ItemLatencyMetrics, ReceiptItem, ReceiptLatencyMetrics, ResolutionResult
from .photo_assist import PhotoAnalyzer, create_default_photo_analyzer
from .receipt_context import (
    NoopSiblingContextAdjudicator,
    SharedPhotoAnalyzer,
    SiblingContextAdjudicator,
    apply_receipt_level_context,
    create_default_shared_photo_analyzer,
)
from .normalize import normalize_receipt_item
from .resolve import ExactItemResolver
from .ui import write_html_report


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CLI flags used to run the pipeline from the terminal."""

    parser = argparse.ArgumentParser(description="Run the exact item resolver pipeline.")
    parser.add_argument("--photo-anchored", required=True, help="Path to the photo-anchored dataset.")
    parser.add_argument("--unanchored", required=True, help="Path to the unanchored dataset.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where predictions and the summary will be written.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["local_only", "photo_ai", "full"],
        default=os.getenv("PIPELINE_MODE", "full"),
        help="Latency/cost mode: local_only, photo_ai, or full.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable in-memory caching for expensive model/photo calls.",
    )
    return parser


def load_and_normalize(
    dataset_path: str,
    dataset_name: str,
    normalization_timings: dict[tuple[str, int, int], float] | None = None,
) -> List[ReceiptItem]:
    """Load one dataset and attach normalized signals to every receipt item."""

    raw_items = load_receipt_items(dataset_path, dataset_name)
    normalized = []
    for item in raw_items:
        start = time.perf_counter()
        normalized_item = normalize_receipt_item(item)
        if normalization_timings is not None:
            normalization_timings[(dataset_name, item.receipt_index, item.item_index)] = elapsed_ms(start)
        normalized.append(normalized_item)
    return normalized


def summarize_results(dataset_name: str, items: Sequence[ReceiptItem], results: Sequence[ResolutionResult]) -> str:
    """Build the per-dataset section of the Markdown summary report."""

    route_counts = Counter(result.route for result in results)
    status_counts = Counter(result.status for result in results)
    photo_available_count = sum(result.photo_evidence_available for result in results)
    photo_attempted_count = sum(result.photo_analysis_attempted for result in results)
    photo_changed_count = sum(result.photo_result_changed for result in results)
    photo_success_count = sum(result.photo_analysis_success for result in results)
    photo_confidence_changed_count = sum(result.photo_confidence_delta != 0 for result in results)
    photo_failure_count = sum(result.photo_analysis_attempted and not result.photo_analysis_success for result in results)
    adjudication_attempted_count = sum(result.adjudication_attempted for result in results)
    adjudication_success_count = sum(result.adjudication_success for result in results)
    receipt_context_used_count = sum(result.receipt_context_used for result in results)
    sibling_context_used_count = sum(result.sibling_context_used for result in results)
    sibling_confidence_changed_count = sum(
        result.sibling_context_used and not result.sibling_context_changed_status
        for result in results
    )
    sibling_status_changed_count = sum(result.sibling_context_changed_status for result in results)
    shared_photo_used_count = sum(result.shared_photo_used for result in results)
    receipt_assignment_used_count = sum(result.receipt_level_assignment_used for result in results)

    representative_success = next((result for result in results if result.status == "resolved"), None)
    # Representative examples make the summary easier to skim than raw counts
    # alone, especially when comparing abstentions and photo-assisted changes.
    representative_abstain = next(
        (
            result
            for result in results
            if result.status in {"ambiguous", "insufficient_evidence"}
        ),
        None,
    )
    representative_photo_help = next(
        (result for result in results if result.photo_result_changed and result.photo_analysis_success),
        None,
    )
    representative_photo_inconclusive = next(
        (result for result in results if result.photo_analysis_attempted and not result.photo_result_changed),
        None,
    )

    lines = [
        f"### {dataset_name}",
        f"- total_items: {len(items)}",
        f"- routes: {dict(sorted(route_counts.items()))}",
        f"- statuses: {dict(sorted(status_counts.items()))}",
        f"- photo_evidence_available: {photo_available_count}",
        f"- photo_analysis_attempted: {photo_attempted_count}",
        f"- photo_analysis_success: {photo_success_count}",
        f"- photo_result_changed: {photo_changed_count}",
        f"- photo_confidence_changed: {photo_confidence_changed_count}",
        f"- photo_failures_or_inconclusive: {photo_failure_count}",
        f"- adjudication_attempted: {adjudication_attempted_count}",
        f"- adjudication_success: {adjudication_success_count}",
        f"- receipt_context_used: {receipt_context_used_count}",
        f"- sibling_context_used: {sibling_context_used_count}",
        f"- sibling_confidence_changed: {sibling_confidence_changed_count}",
        f"- sibling_status_changed: {sibling_status_changed_count}",
        f"- shared_photo_used: {shared_photo_used_count}",
        f"- receipt_level_assignment_used: {receipt_assignment_used_count}",
    ]

    if representative_success:
        lines.append(
            "- representative_resolved: "
            f"{representative_success.input_item_name} -> {representative_success.resolved_title}"
        )
    if representative_abstain:
        lines.append(
            "- representative_abstain: "
            f"{representative_abstain.input_item_name} ({representative_abstain.status})"
        )
    if representative_photo_help:
        lines.append(
            "- representative_photo_helped: "
            f"{representative_photo_help.input_item_name} -> {representative_photo_help.resolved_title}"
        )
    if representative_photo_inconclusive:
        lines.append(
            "- representative_photo_inconclusive: "
            f"{representative_photo_inconclusive.input_item_name} ({representative_photo_inconclusive.status})"
        )

    return "\n".join(lines)


def build_summary(
    all_results: Dict[str, Sequence[ResolutionResult]],
    all_items: Dict[str, Sequence[ReceiptItem]],
    latency_report: dict | None = None,
) -> str:
    """Build the human-readable Markdown report across all datasets."""

    status_totals = Counter()
    route_totals = Counter()
    photo_available_total = 0
    photo_attempted_total = 0
    photo_changed_total = 0
    photo_success_total = 0
    photo_confidence_changed_total = 0
    photo_failure_total = 0
    adjudication_attempted_total = 0
    adjudication_success_total = 0
    receipt_context_used_total = 0
    sibling_context_used_total = 0
    sibling_confidence_changed_total = 0
    sibling_status_changed_total = 0
    shared_photo_used_total = 0
    receipt_assignment_used_total = 0

    for dataset_name, results in all_results.items():
        # Aggregate usage flags across datasets so the summary explains both
        # quality decisions and the cost/latency shape of the run.
        status_totals.update(result.status for result in results)
        route_totals.update(result.route for result in results)
        photo_available_total += sum(result.photo_evidence_available for result in results)
        photo_attempted_total += sum(result.photo_analysis_attempted for result in results)
        photo_changed_total += sum(result.photo_result_changed for result in results)
        photo_success_total += sum(result.photo_analysis_success for result in results)
        photo_confidence_changed_total += sum(result.photo_confidence_delta != 0 for result in results)
        photo_failure_total += sum(result.photo_analysis_attempted and not result.photo_analysis_success for result in results)
        adjudication_attempted_total += sum(result.adjudication_attempted for result in results)
        adjudication_success_total += sum(result.adjudication_success for result in results)
        receipt_context_used_total += sum(result.receipt_context_used for result in results)
        sibling_context_used_total += sum(result.sibling_context_used for result in results)
        sibling_confidence_changed_total += sum(
            result.sibling_context_used and not result.sibling_context_changed_status
            for result in results
        )
        sibling_status_changed_total += sum(result.sibling_context_changed_status for result in results)
        shared_photo_used_total += sum(result.shared_photo_used for result in results)
        receipt_assignment_used_total += sum(result.receipt_level_assignment_used for result in results)

    sections = [
        "# Exact Item AI Summary",
        "",
        "## Overall",
        f"- total_items: {sum(len(results) for results in all_results.values())}",
        f"- route_distribution: {dict(sorted(route_totals.items()))}",
        f"- status_distribution: {dict(sorted(status_totals.items()))}",
        f"- photo_evidence_available: {photo_available_total}",
        f"- photo_analysis_attempted: {photo_attempted_total}",
        f"- photo_analysis_success: {photo_success_total}",
        f"- photo_result_changed: {photo_changed_total}",
        f"- photo_confidence_changed: {photo_confidence_changed_total}",
        f"- photo_failures_or_inconclusive: {photo_failure_total}",
        f"- adjudication_attempted: {adjudication_attempted_total}",
        f"- adjudication_success: {adjudication_success_total}",
        f"- receipt_context_used: {receipt_context_used_total}",
        f"- sibling_context_used: {sibling_context_used_total}",
        f"- sibling_confidence_changed: {sibling_confidence_changed_total}",
        f"- sibling_status_changed: {sibling_status_changed_total}",
        f"- shared_photo_used: {shared_photo_used_total}",
        f"- receipt_level_assignment_used: {receipt_assignment_used_total}",
    ]
    if latency_report:
        total_items = sum(len(results) for results in all_results.values())
        item_stats = latency_report["aggregate_item_latency"]
        receipt_stats = latency_report["aggregate_receipt_latency"]
        local_only_count = latency_report["by_expensive_path"]["local_only"]["count"]
        local_only_pct = round((local_only_count / total_items) * 100, 1) if total_items else 0.0
        sections.extend(
            [
                f"- pipeline_mode: {latency_report['pipeline_mode']}",
                f"- local_fast_path_items: {local_only_count} ({local_only_pct}%)",
                f"- external_api_call_count: {latency_report['external_api_call_count']}",
                f"- cached_photo_hits: {latency_report['cache_stats'].get('photo_cache_hits', 0)}",
                f"- cached_shared_photo_hits: {latency_report['cache_stats'].get('shared_photo_cache_hits', 0)}",
                "",
                "## Latency",
                (
                    "- item_latency_ms: "
                    f"median={item_stats['median_ms']}, p95={item_stats['p95_ms']}, "
                    f"avg={item_stats['avg_ms']}, max={item_stats['max_ms']}"
                ),
                (
                    "- receipt_latency_ms: "
                    f"median={receipt_stats['median_ms']}, p95={receipt_stats['p95_ms']}, "
                    f"avg={receipt_stats['avg_ms']}, max={receipt_stats['max_ms']}"
                ),
                "- latency_report_json: output/latency_report.json",
            ]
        )
    sections.extend(["", "## Dataset Breakdown"])

    for dataset_name in sorted(all_results):
        sections.append(summarize_results(dataset_name, all_items[dataset_name], all_results[dataset_name]))
        sections.append("")

    sections.extend(
        [
            "## Observations",
            "- Generic thrift and category-level lines abstain instead of forcing exact products.",
            "- Detailed catalog or title-rich lines resolve locally without web or LLM dependencies.",
            "- Photo AI is gated behind low-confidence anchored cases, with model-based adjudication deciding whether photo evidence should affect the result.",
            "- Receipt-level context is a post-resolution stage that uses sibling signals and shared-photo assignments as soft evidence only.",
        ]
    )

    return "\n".join(sections).strip() + "\n"


def run_pipeline(
    *,
    photo_anchored_path: str,
    unanchored_path: str,
    output_dir: str | Path = "output",
    photo_analyzer: PhotoAnalyzer | None = None,
    adjudicator: EvidenceAdjudicator | None = None,
    shared_photo_analyzer: SharedPhotoAnalyzer | None = None,
    sibling_context_adjudicator: SiblingContextAdjudicator | None = None,
    enable_photo_assist: bool = True,
    enable_receipt_context: bool = True,
    pipeline_mode: str = "full",
    enable_cache: bool = True,
) -> Dict[str, List[ResolutionResult]]:
    """Run the full exact-item resolver and write all output artifacts."""

    if pipeline_mode == "local_only":
        # Local-only mode disables all expensive/model-backed stages so latency
        # can be compared against the richer modes.
        enable_photo_assist = False
        enable_receipt_context = False
        adjudicator = NoopEvidenceAdjudicator()
        sibling_context_adjudicator = NoopSiblingContextAdjudicator()
    elif pipeline_mode == "photo_ai":
        # Photo AI mode exercises item-level image evidence while leaving the
        # broader receipt-context pass disabled.
        enable_photo_assist = True
        enable_receipt_context = False
        sibling_context_adjudicator = NoopSiblingContextAdjudicator()

    normalization_timings: dict[tuple[str, int, int], float] = {}
    datasets = {
        "photo_anchored": load_and_normalize(photo_anchored_path, "photo_anchored", normalization_timings),
        "unanchored": load_and_normalize(unanchored_path, "unanchored", normalization_timings),
    }
    photo_cache = None
    adjudication_cache = None
    shared_photo_cache = None
    if photo_analyzer is None:
        # Dependencies are injectable for tests; production defaults come from
        # environment-backed factories.
        photo_analyzer = create_default_photo_analyzer()
    if adjudicator is None:
        adjudicator = create_default_adjudicator()
    if shared_photo_analyzer is None and enable_receipt_context:
        shared_photo_analyzer = create_default_shared_photo_analyzer()
    if enable_cache and photo_analyzer is not None:
        # Cache wrappers avoid repeating photo/adjudication work for repeated
        # image URLs or identical adjudication inputs during a single run.
        photo_cache = CachedPhotoAnalyzer(photo_analyzer)
        photo_analyzer = photo_cache
    if enable_cache and adjudicator is not None:
        adjudication_cache = CachedEvidenceAdjudicator(adjudicator)
        adjudicator = adjudication_cache
    if enable_cache and shared_photo_analyzer is not None:
        shared_photo_cache = CachedSharedPhotoAnalyzer(shared_photo_analyzer)
        shared_photo_analyzer = shared_photo_cache

    resolver = ExactItemResolver(
        photo_analyzer=photo_analyzer,
        adjudicator=adjudicator,
        enable_photo_assist=enable_photo_assist,
    )
    results = {name: resolver.resolve_batch(items) for name, items in datasets.items()}
    # Normalization happens before the resolver, so add that timing back onto
    # each item result before generating reports.
    results = {
        dataset_name: [
            _with_normalization_latency(result, normalization_timings)
            for result in dataset_results
        ]
        for dataset_name, dataset_results in results.items()
    }
    receipt_metrics: list[ReceiptLatencyMetrics] = []
    if enable_receipt_context:
        # Receipt context is applied one receipt at a time so sibling and shared
        # photo evidence cannot leak between unrelated purchases.
        for dataset_name in list(results):
            updated_dataset_results: list[ResolutionResult] = []
            for receipt_index in sorted({item.receipt_index for item in datasets[dataset_name]}):
                receipt_items = [item for item in datasets[dataset_name] if item.receipt_index == receipt_index]
                receipt_results = [result for result in results[dataset_name] if result.receipt_index == receipt_index]
                receipt_start = time.perf_counter()
                updated_receipt_results = apply_receipt_level_context(
                    receipt_items,
                    receipt_results,
                    shared_photo_analyzer=shared_photo_analyzer,
                    sibling_context_adjudicator=sibling_context_adjudicator,
                )
                receipt_context_ms = elapsed_ms(receipt_start)
                per_item_receipt_context_ms = receipt_context_ms / max(len(updated_receipt_results), 1)
                updated_receipt_results = [
                    _with_receipt_context_latency(result, per_item_receipt_context_ms)
                    for result in updated_receipt_results
                ]
                updated_dataset_results.extend(updated_receipt_results)
                receipt_metrics.append(
                    _build_receipt_latency_metric(
                        dataset_name,
                        receipt_index,
                        updated_receipt_results,
                        receipt_context_ms,
                        photo_cache=photo_cache,
                        adjudication_cache=adjudication_cache,
                        shared_photo_cache=shared_photo_cache,
                    )
                )
            results[dataset_name] = sorted(
                updated_dataset_results,
                key=lambda result: (result.receipt_index, result.item_index),
            )
    else:
        # When receipt context is disabled, still emit receipt-level latency
        # metrics by summing the item timings within each receipt.
        for dataset_name, dataset_results in results.items():
            for receipt_index in sorted({result.receipt_index for result in dataset_results}):
                receipt_results = [result for result in dataset_results if result.receipt_index == receipt_index]
                receipt_metrics.append(
                    _build_receipt_latency_metric(
                        dataset_name,
                        receipt_index,
                        receipt_results,
                        sum(result.latency_metrics.total_item_ms for result in receipt_results),
                        photo_cache=photo_cache,
                        adjudication_cache=adjudication_cache,
                        shared_photo_cache=shared_photo_cache,
                    )
                )

    results = {
        dataset_name: [_with_usage_flags(result) for result in dataset_results]
        for dataset_name, dataset_results in results.items()
    }
    cache_stats = {
        # These counters are intentionally centralized here because the summary,
        # JSON latency report, and HTML view all consume the same pipeline result.
        "photo_cache_hits": getattr(photo_cache, "cache_hits", 0),
        "photo_external_calls": getattr(photo_cache, "external_api_call_count", 0),
        "adjudication_cache_hits": getattr(adjudication_cache, "cache_hits", 0),
        "adjudication_external_calls": getattr(adjudication_cache, "external_api_call_count", 0),
        "shared_photo_cache_hits": getattr(shared_photo_cache, "cache_hits", 0),
        "shared_photo_external_calls": getattr(shared_photo_cache, "external_api_call_count", 0),
        "receipt_context_cache_hits": 0,
    }
    latency_report = build_latency_report(
        results,
        receipt_metrics,
        pipeline_mode=pipeline_mode,
        cache_stats=cache_stats,
    )

    output_root = Path(output_dir)
    write_results(output_root / "photo_anchored_predictions.json", results["photo_anchored"])
    write_results(output_root / "unanchored_predictions.json", results["unanchored"])
    write_text(output_root / "summary.md", build_summary(results, datasets, latency_report))
    write_latency_report(output_root / "latency_report.json", latency_report)
    write_html_report(output_root / "index.html", results)
    return results


def _with_normalization_latency(
    result: ResolutionResult,
    normalization_timings: dict[tuple[str, int, int], float],
) -> ResolutionResult:
    """Return a copy of a result with normalization timing included."""

    latency = replace(result.latency_metrics)
    latency.normalization_ms = normalization_timings.get(
        (result.dataset_name, result.receipt_index, result.item_index),
        0.0,
    )
    latency.total_item_ms = round(latency.total_item_ms + latency.normalization_ms, 2)
    return replace(result, latency_metrics=latency)


def _with_receipt_context_latency(result: ResolutionResult, receipt_context_ms: float) -> ResolutionResult:
    """Return a copy of a result with its share of receipt-context timing."""

    latency = replace(result.latency_metrics)
    latency.receipt_context_ms = round(receipt_context_ms, 2)
    if result.sibling_context_used:
        latency.sibling_context_ms = round(receipt_context_ms, 2)
    latency.total_item_ms = round(latency.total_item_ms + receipt_context_ms, 2)
    return replace(result, latency_metrics=latency)


def _with_usage_flags(result: ResolutionResult) -> ResolutionResult:
    """Set convenience booleans that describe which expensive paths were used."""

    used_photo_ai = result.photo_analysis_attempted and result.photo_analysis_success
    used_adjudication = result.adjudication_attempted
    used_shared_photo = result.shared_photo_used
    used_sibling = result.sibling_context_used
    used_local_only = not (
        used_photo_ai
        or used_adjudication
        or result.receipt_context_used
        or used_sibling
        or used_shared_photo
    )
    return replace(
        result,
        used_local_only=used_local_only,
        used_photo_ai=used_photo_ai,
        used_adjudication=used_adjudication,
        used_sibling_context_flag=used_sibling,
        used_shared_photo_assignment=used_shared_photo,
    )


def _build_receipt_latency_metric(
    dataset_name: str,
    receipt_index: int,
    receipt_results: Sequence[ResolutionResult],
    total_receipt_ms: float,
    *,
    photo_cache: CachedPhotoAnalyzer | None,
    adjudication_cache: CachedEvidenceAdjudicator | None,
    shared_photo_cache: CachedSharedPhotoAnalyzer | None,
) -> ReceiptLatencyMetrics:
    """Summarize item outcomes and expensive-path counts for one receipt."""

    return ReceiptLatencyMetrics(
        dataset_name=dataset_name,
        receipt_index=receipt_index,
        total_receipt_ms=round(total_receipt_ms, 2),
        item_count=len(receipt_results),
        items_resolved=sum(result.status == "resolved" for result in receipt_results),
        items_ambiguous=sum(result.status == "ambiguous" for result in receipt_results),
        items_insufficient_evidence=sum(result.status == "insufficient_evidence" for result in receipt_results),
        items_with_photo_ai=sum(result.photo_analysis_attempted and result.photo_analysis_success for result in receipt_results),
        items_with_adjudication=sum(result.adjudication_attempted for result in receipt_results),
        items_with_sibling_context=sum(result.sibling_context_used for result in receipt_results),
        external_api_call_count=(
            sum(result.photo_analysis_attempted and result.photo_analysis_success for result in receipt_results)
            + sum(result.adjudication_attempted for result in receipt_results)
            + sum(result.shared_photo_used for result in receipt_results)
        ),
        cached_photo_hits=getattr(photo_cache, "cache_hits", 0),
        cached_receipt_context_hits=0,
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Parse CLI arguments, run the pipeline, and print a short completion note."""

    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = run_pipeline(
        photo_anchored_path=args.photo_anchored,
        unanchored_path=args.unanchored,
        output_dir=args.output_dir,
        pipeline_mode=args.pipeline_mode,
        enable_cache=not args.disable_cache,
    )

    status_totals = Counter(
        result.status for dataset_results in results.values() for result in dataset_results
    )
    print("Pipeline complete.")
    print(f"Status totals: {dict(sorted(status_totals.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
