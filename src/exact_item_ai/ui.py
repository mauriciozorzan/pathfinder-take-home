from __future__ import annotations

import argparse
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, Sequence

from .models import ResolutionResult


def build_html_report(all_results: Dict[str, Sequence[ResolutionResult]]) -> str:
    rows = []
    for dataset_name, results in sorted(all_results.items()):
        for result in results:
            rows.append(
                "<tr>"
                f"<td>{html.escape(dataset_name)}</td>"
                f"<td>{html.escape(result.status)}</td>"
                f"<td>{html.escape(result.route)}</td>"
                f"<td>{result.confidence:.2f}</td>"
                f"<td>{html.escape(result.merchant)}</td>"
                f"<td>{html.escape(result.input_item_name)}</td>"
                f"<td>{html.escape(result.resolved_title or '')}</td>"
                f"<td>{html.escape(result.condition or '')}</td>"
                f"<td>{result.latency_metrics.total_item_ms:.2f}</td>"
                f"<td>{'yes' if result.used_local_only else 'no'}</td>"
                f"<td>{'yes' if result.photo_analysis_attempted else 'no'}</td>"
                f"<td>{'yes' if result.photo_analysis_success else 'no'}</td>"
                f"<td>{'yes' if result.photo_result_changed else 'no'}</td>"
                f"<td>{result.photo_confidence_delta:.2f}</td>"
                f"<td>{'' if result.photo_model_confidence is None else f'{result.photo_model_confidence:.2f}'}</td>"
                f"<td>{html.escape(result.photo_evidence_summary or '')}</td>"
                f"<td>{'yes' if result.photo_candidate_specific else 'no'}</td>"
                f"<td>{'yes' if result.photo_candidate_sent_to_adjudication else 'no'}</td>"
                f"<td>{html.escape(result.photo_adjudication_block_reason or '')}</td>"
                f"<td>{html.escape(result.adjudication_decision or '')}</td>"
                f"<td>{html.escape(result.adjudication_contradiction_strength)}</td>"
                f"<td>{html.escape(result.adjudication_rationale or '')}</td>"
                f"<td>{'yes' if result.receipt_context_used else 'no'}</td>"
                f"<td>{'yes' if result.sibling_context_used else 'no'}</td>"
                f"<td>{'' if result.sibling_similarity_score is None else f'{result.sibling_similarity_score:.2f}'}</td>"
                f"<td>{'yes' if result.family_consistent_with_siblings else 'no'}</td>"
                f"<td>{'yes' if result.sibling_context_changed_status else 'no'}</td>"
                f"<td>{'yes' if result.shared_photo_used else 'no'}</td>"
                f"<td>{html.escape(result.receipt_level_assignment_basis or '')}</td>"
                f"<td>{'' if result.receipt_level_assignment_confidence is None else f'{result.receipt_level_assignment_confidence:.2f}'}</td>"
                f"<td>{html.escape('; '.join(result.receipt_level_notes))}</td>"
                f"<td>{html.escape(result.notes or '')}</td>"
                "</tr>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exact Item AI Viewer</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    input, select {{
      padding: 8px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
      position: sticky;
      top: 0;
    }}
    .th-controls {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 130px;
    }}
    .th-label {{
      font-weight: 600;
      white-space: nowrap;
    }}
    .th-select {{
      width: 100%;
      font-size: 12px;
      padding: 4px;
      box-sizing: border-box;
    }}
    .summary {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .card {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px 16px;
      min-width: 160px;
    }}
  </style>
</head>
<body>
  <h1>Exact Item AI Viewer</h1>
  <p>Filter predictions by dataset, status, route, or free-text search.</p>

  <div class="summary">
    <div class="card"><strong>Datasets</strong><br>{len(all_results)}</div>
    <div class="card"><strong>Total Items</strong><br>{sum(len(results) for results in all_results.values())}</div>
    <div class="card"><strong>Resolved</strong><br>{sum(result.status == "resolved" for results in all_results.values() for result in results)}</div>
    <div class="card"><strong>Abstained</strong><br>{sum(result.status != "resolved" for results in all_results.values() for result in results)}</div>
  </div>

  <div class="controls">
    <input id="search" placeholder="Search merchant or item name">
    <select id="dataset">
      <option value="">All datasets</option>
      <option value="photo_anchored">photo_anchored</option>
      <option value="unanchored">unanchored</option>
    </select>
    <select id="status">
      <option value="">All statuses</option>
      <option value="resolved">resolved</option>
      <option value="ambiguous">ambiguous</option>
      <option value="insufficient_evidence">insufficient_evidence</option>
    </select>
    <select id="route">
      <option value="">All routes</option>
      <option value="deterministic">deterministic</option>
      <option value="retrieval_needed">retrieval_needed</option>
      <option value="photo_assisted">photo_assisted</option>
      <option value="insufficient_evidence">insufficient_evidence</option>
    </select>
  </div>

  <table id="results">
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Status</th>
        <th>Route</th>
        <th>Confidence</th>
        <th>Merchant</th>
        <th>Input Item</th>
        <th>Resolved Title</th>
        <th>Condition</th>
        <th>Total Latency Ms</th>
        <th>Local Only</th>
        <th>Photo Tried</th>
        <th>Photo Success</th>
        <th>Photo Changed</th>
        <th>Photo Delta</th>
        <th>Photo Confidence</th>
        <th>Photo Summary</th>
        <th>Photo Candidate Specific</th>
        <th>Sent To Adjudication</th>
        <th>Photo Block Reason</th>
        <th>Adjudication</th>
        <th>Contradiction</th>
        <th>Adjudication Rationale</th>
        <th>Receipt Context</th>
        <th>Sibling Context</th>
        <th>Sibling Similarity</th>
        <th>Sibling Family</th>
        <th>Sibling Changed Status</th>
        <th>Shared Photo</th>
        <th>Receipt Assignment</th>
        <th>Receipt Assignment Confidence</th>
        <th>Receipt Notes</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <script>
    const search = document.getElementById("search");
    const dataset = document.getElementById("dataset");
    const status = document.getElementById("status");
    const route = document.getElementById("route");
    const table = document.getElementById("results");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const headers = Array.from(table.querySelectorAll("thead th"));
    const rowData = rows.map((row, index) => ({
      row,
      index,
      cells: Array.from(row.querySelectorAll("td")).map((cell) => cell.textContent.trim()),
    }));
    const blankToken = "__BLANK__";
    const columnFilters = new Map();
    let sortColumnIndex = null;
    let sortDirection = "";

    function isNumeric(value) {{
      return value !== "" && !Number.isNaN(Number(value));
    }}

    function compareValues(left, right) {{
      if (isNumeric(left) && isNumeric(right)) {{
        return Number(left) - Number(right);
      }}
      return left.localeCompare(right, undefined, {{ sensitivity: "base" }});
    }}

    function createOption(value, label) {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }}

    function buildColumnControls() {{
      for (const [index, header] of headers.entries()) {{
        const labelText = header.textContent.trim();
        header.textContent = "";

        const wrapper = document.createElement("div");
        wrapper.className = "th-controls";

        const label = document.createElement("div");
        label.className = "th-label";
        label.textContent = labelText;

        const sortSelect = document.createElement("select");
        sortSelect.className = "th-select";
        sortSelect.title = `Sort ${labelText}`;
        sortSelect.appendChild(createOption("", "Sort"));
        sortSelect.appendChild(createOption("asc", "Sort A-Z / low-high"));
        sortSelect.appendChild(createOption("desc", "Sort Z-A / high-low"));
        sortSelect.addEventListener("change", () => {{
          sortColumnIndex = sortSelect.value ? index : null;
          sortDirection = sortSelect.value;
          if (sortColumnIndex !== null) {{
            for (const otherHeader of headers) {{
              if (otherHeader === header) {{
                continue;
              }}
              const otherSortSelect = otherHeader.querySelector("select[data-kind='sort']");
              if (otherSortSelect) {{
                otherSortSelect.value = "";
              }}
            }}
          }}
          applyFilters();
        }});
        sortSelect.dataset.kind = "sort";

        const filterSelect = document.createElement("select");
        filterSelect.className = "th-select";
        filterSelect.title = `Filter ${labelText}`;
        filterSelect.appendChild(createOption("", "All values"));

        const values = Array.from(
          new Set(rowData.map((item) => item.cells[index] ?? ""))
        ).sort((a, b) => compareValues(a, b));
        for (const value of values) {{
          if (value === "") {{
            filterSelect.appendChild(createOption(blankToken, "(blank)"));
          }} else {{
            filterSelect.appendChild(createOption(value, value));
          }}
        }}
        filterSelect.addEventListener("change", () => {{
          if (!filterSelect.value) {{
            columnFilters.delete(index);
          }} else {{
            columnFilters.set(index, filterSelect.value);
          }}
          applyFilters();
        }});

        wrapper.appendChild(label);
        wrapper.appendChild(sortSelect);
        wrapper.appendChild(filterSelect);
        header.appendChild(wrapper);
      }}
    }}

    function applyFilters() {{
      const searchText = search.value.toLowerCase();
      const visible = [];
      const hidden = [];

      for (const item of rowData) {{
        const rowDataset = item.cells[0] ?? "";
        const rowStatus = item.cells[1] ?? "";
        const rowRoute = item.cells[2] ?? "";
        const haystack = item.cells.join(" ").toLowerCase();

        let matches =
          (!dataset.value || rowDataset === dataset.value) &&
          (!status.value || rowStatus === status.value) &&
          (!route.value || rowRoute === route.value) &&
          (!searchText || haystack.includes(searchText));

        if (matches) {{
          for (const [columnIndex, expected] of columnFilters.entries()) {{
            const value = item.cells[columnIndex] ?? "";
            if (expected === blankToken) {{
              if (value !== "") {{
                matches = false;
                break;
              }}
            }} else if (value !== expected) {{
              matches = false;
              break;
            }}
          }}
        }}

        if (matches) {{
          visible.push(item);
        }} else {{
          hidden.push(item);
        }}
      }}

      if (sortColumnIndex !== null && sortDirection) {{
        visible.sort((left, right) => {{
          const comparison = compareValues(
            left.cells[sortColumnIndex] ?? "",
            right.cells[sortColumnIndex] ?? "",
          );
          if (comparison !== 0) {{
            return sortDirection === "asc" ? comparison : -comparison;
          }}
          return left.index - right.index;
        }});
      }}

      for (const item of visible) {{
        item.row.style.display = "";
        tbody.appendChild(item.row);
      }}
      for (const item of hidden) {{
        item.row.style.display = "none";
        tbody.appendChild(item.row);
      }}
    }}

    buildColumnControls();
    for (const control of [search, dataset, status, route]) {{
      control.addEventListener("input", applyFilters);
      control.addEventListener("change", applyFilters);
    }}
    applyFilters();
  </script>
</body>
</html>
"""


def write_html_report(output_path: str | Path, all_results: Dict[str, Sequence[ResolutionResult]]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html_report(all_results))
    return path


def serve_output_directory(directory: str | Path, port: int = 8000) -> None:
    path = Path(directory)
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(path), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving {path} at http://127.0.0.1:{port}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the generated output viewer.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    serve_output_directory(args.output_dir, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
