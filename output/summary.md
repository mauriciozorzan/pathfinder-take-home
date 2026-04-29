# Exact Item AI Summary

## Overall
- total_items: 92
- route_distribution: {'deterministic': 53, 'insufficient_evidence': 31, 'photo_assisted': 7, 'retrieval_needed': 1}
- status_distribution: {'ambiguous': 1, 'insufficient_evidence': 31, 'resolved': 60}
- photo_evidence_available: 13
- photo_analysis_attempted: 6
- photo_analysis_success: 6
- photo_result_changed: 6
- photo_confidence_changed: 6
- photo_failures_or_inconclusive: 0
- adjudication_attempted: 6
- adjudication_success: 6
- receipt_context_used: 3
- sibling_context_used: 0
- sibling_confidence_changed: 0
- sibling_status_changed: 0
- shared_photo_used: 1
- receipt_level_assignment_used: 3
- pipeline_mode: full
- local_fast_path_items: 83 (90.2%)
- external_api_call_count: 13
- cached_photo_hits: 0
- cached_shared_photo_hits: 0

## Latency
- item_latency_ms: median=0.17, p95=8064.77, avg=754.52, max=13766.34
- receipt_latency_ms: median=0.24, p95=1941.91, avg=268.64, max=5833.15
- latency_report_json: output/latency_report.json

## Dataset Breakdown
### photo_anchored
- total_items: 45
- routes: {'deterministic': 7, 'insufficient_evidence': 31, 'photo_assisted': 7}
- statuses: {'insufficient_evidence': 31, 'resolved': 14}
- photo_evidence_available: 13
- photo_analysis_attempted: 6
- photo_analysis_success: 6
- photo_result_changed: 6
- photo_confidence_changed: 6
- photo_failures_or_inconclusive: 0
- adjudication_attempted: 6
- adjudication_success: 6
- receipt_context_used: 1
- sibling_context_used: 0
- sibling_confidence_changed: 0
- sibling_status_changed: 0
- shared_photo_used: 1
- receipt_level_assignment_used: 1
- representative_resolved: Tara Toys Minecraft Pixel Art -> Tara Toys Minecraft Pixel Art
- representative_abstain: Used Curriculum and Books (insufficient_evidence)
- representative_photo_helped: LG UNICORN CAKE -> LEGO Friends Unicorn Cake Delivery Car

### unanchored
- total_items: 47
- routes: {'deterministic': 46, 'retrieval_needed': 1}
- statuses: {'ambiguous': 1, 'resolved': 46}
- photo_evidence_available: 0
- photo_analysis_attempted: 0
- photo_analysis_success: 0
- photo_result_changed: 0
- photo_confidence_changed: 0
- photo_failures_or_inconclusive: 0
- adjudication_attempted: 0
- adjudication_success: 0
- receipt_context_used: 2
- sibling_context_used: 0
- sibling_confidence_changed: 0
- sibling_status_changed: 0
- shared_photo_used: 0
- receipt_level_assignment_used: 2
- representative_resolved: Pastel Smarkers - Washable Patented Gourmet Scented Pastel Markers, Assorted Colors, Standard Point Felt Tip, 16 Count -> Pastel Smarkers - Washable Patented Gourmet Scented Pastel Markers, Assorted Colors, Standard Point Felt Tip, 16 Count
- representative_abstain: Eight Ball Helmet (ambiguous)

## Observations
- Generic thrift and category-level lines abstain instead of forcing exact products.
- Detailed catalog or title-rich lines resolve locally without web or LLM dependencies.
- Photo AI is gated behind low-confidence anchored cases, with model-based adjudication deciding whether photo evidence should affect the result.
- Receipt-level context is a post-resolution stage that uses sibling signals and shared-photo assignments as soft evidence only.
