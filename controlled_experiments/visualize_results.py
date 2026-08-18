"""Render completed controlled-experiment JSON results as a sortable HTML table."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def score(summary: dict[str, object], split: str, metric: str) -> float | None:
    value = summary.get(split)
    if isinstance(value, dict):
        metric_value = value.get(metric)
        return float(metric_value) if isinstance(metric_value, (int, float)) else None
    return None


def rows_for_file(path: Path) -> list[dict[str, object]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "evaluation" not in summary:
        return [summary | {"label": path.stem}]
    rows = []
    for recurrences, evaluation in summary["evaluation"].items():
        rows.append(
            summary
            | evaluation
            | {"label": f"{path.stem} (eval {recurrences})", "evaluation_recurrences": recurrences}
        )
    return rows


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def shade(value: float | None) -> str:
    if value is None:
        return ""
    alpha = 0.08 + value * 0.42
    return f"background: rgba(37, 99, 235, {alpha:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(path for path in args.results_dir.glob("*.json") if not path.name.startswith("profile_"))
    rows = [row for path in files for row in rows_for_file(path)]
    if not rows:
        raise ValueError(f"no result JSON files found in {args.results_dir}")
    output = args.output or args.results_dir / "results_dashboard.html"
    body = []
    for row in rows:
        architecture = str(row.get("architecture", "scratchpad"))
        layers = row.get("full_token_layers", row.get("layers", "—"))
        recurrence = row.get("evaluation_recurrences", row.get("recurrences", "—"))
        metrics = [
            score(row, "train", "exact_accuracy"),
            score(row, "test", "exact_accuracy"),
            score(row, "ood_long", "exact_accuracy"),
            score(row, "train", "digit_accuracy"),
            score(row, "test", "digit_accuracy"),
            score(row, "ood_long", "digit_accuracy"),
        ]
        cells = "".join(f'<td style="{shade(value)}">{percent(value)}</td>' for value in metrics)
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('task', 'multiplication')))}</td>"
            f"<td>{html.escape(architecture)}</td><td>{layers}</td><td>{recurrence}</td>"
            f"{cells}<td>{html.escape(str(row['label']))}</td></tr>"
        )
    output.write_text(
        """<!doctype html><html><head><meta charset="utf-8"><title>Controlled results</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 28px; color: #172033; }
table { border-collapse: collapse; font-size: 13px; } th, td { padding: 7px 10px; border: 1px solid #cbd5e1; }
th { background: #e2e8f0; } td:last-child { font-family: ui-monospace, monospace; font-size: 11px; }
</style></head><body><h1>Controlled-experiment results</h1>
<table><thead><tr><th>task</th><th>architecture</th><th>layers</th><th>eval repeats</th>
<th>train exact</th><th>test exact</th><th>OOD exact</th>
<th>train digit</th><th>test digit</th><th>OOD digit</th><th>source</th></tr></thead><tbody>"""
        + "\n".join(body)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
