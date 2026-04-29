# Exact Item AI

This repository contains a local-first prototype.

## What it does

- Loads the provided receipt JSON datasets
- Flattens receipts into item-level examples
- Normalizes merchant, title, ID, price, and resale condition signals
- Routes each item into deterministic, retrieval-needed, photo-assisted, or
  insufficient-evidence buckets
- Resolves high-confidence items conservatively and abstains on weak evidence
- Applies optional AI-based photo analysis selectively for weak anchored items
- Uses model-based adjudication to reconcile receipt context and photo evidence
- Applies a conservative receipt-level context stage for catalog coherence,
  sibling signals, and repeated shared-photo evidence
- Writes structured predictions and a short evaluation summary

## Project layout

```text
src/exact_item_ai/
  models.py
  io_utils.py
  normalize.py
  route.py
  resolve.py
  receipt_context.py
  score.py
  main.py
tests/
output/
```

## Run the pipeline

```bash
PYTHONPATH=src python3 -m exact_item_ai.main \
  --photo-anchored "/Users/mauricio/Downloads/exact_item_dataset_photo_anchored_takehome.json" \
  --unanchored "/Users/mauricio/Downloads/exact_item_dataset_unanchored_takehome.json"
```

Outputs are written to `output/`.

By default, the pipeline stays local-first and does not require an API key. To
enable real AI-based photo analysis for eligible anchored low-confidence items,
set.

When enabled, the pipeline keeps separate stages for hard anchored cases:

1. Text-first resolution runs first.
2. Photo evidence extraction fetches the actual reference image and asks a
   vision-capable model what product evidence is visible.
3. Evidence adjudication asks a model to compare the receipt context, current
   text result, and photo evidence before any photo-derived identity is used.
4. Receipt-level context runs after item-level resolution. It can recognize
   coherent catalog-like receipts, use resolved sibling items as weak support,
   and analyze repeated shared photos as multi-item evidence.

Python code only gates the expensive path and enforces broad fail-closed policy.
It does not use hardcoded shorthand expansion, product mappings, or
category-specific matching rules to reconcile receipt text with photo output.
The adjudicator treats receipt text as shorthand, anchored images as high-weight
evidence, and only blocks image-led refinement on strong contradictions. If
config is missing, image fetch fails, or the model/adjudicator is inconclusive,
the pipeline keeps the text-first result.
Strong photo refinements can resolve when the image provides visually grounded
branding/title/packaging, the adjudicator says the photo-derived title plausibly
explains the receipt shorthand, and there is no strong contradiction. Weak photo
clues remain ambiguous.

Receipt-level context is also fail-closed. Sibling context is adjudicated by a
model-backed semantic step rather than handwritten product-family maps. It can
promote an already plausible text/photo candidate when resolved siblings make the
candidate more plausible, but it cannot invent a candidate by itself. Shared-photo
assignment only updates a result when a repeated reference image produces
visually grounded candidates and the remaining receipt lines/candidates are
coherent enough for a conservative one-to-one assignment.
For catalog-like receipts, non-generic named titles with item IDs can be promoted
when receipt-level coherence is strong, using general signals such as item-ID
coverage, named-title ratio, catalog/curriculum title patterns, and sibling ID
format similarity.
Separately, standalone product-like titles can resolve locally when their title
structure is already specific enough to function as the exact item identity.

## Latency and caching

The pipeline records stage-level timing for each item and receipt. Each
prediction includes `latency_metrics` with normalization, routing, local
resolution, photo, adjudication, receipt context, and total item timing. The
pipeline also writes `output/latency_report.json` with aggregate median, p95,
average, and max latency by item, receipt, route, and expensive-path usage.

Use pipeline modes to compare latency/accuracy tradeoffs:

```bash
PYTHONPATH=src python3 -m exact_item_ai.main \
  --photo-anchored /path/to/photo.json \
  --unanchored /path/to/unanchored.json \
  --pipeline-mode local_only
```

Modes are `local_only`, `photo_ai`, and `full`. Expensive calls are gated and
cached in memory by photo URL, adjudication input, and shared-photo URL during a
run. The summary reports local fast-path count, external call count, cache hits,
and median/p95 receipt latency so the project can discuss responsiveness as well
as correctness.

To inspect results in a lightweight viewer:

```bash
PYTHONPATH=src python3 -m exact_item_ai.ui --output-dir output --port 8000
```

Then open `http://127.0.0.1:8000/index.html`.

## Run tests

```bash
python3 -m unittest discover -s tests
```
