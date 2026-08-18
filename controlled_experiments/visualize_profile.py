"""Render one or more optimization-profile JSONL files as a self-contained HTML dashboard."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Iterable


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c")


def load_profile(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    config = next((record for record in records if record.get("kind") == "configuration"), {})
    steps = [record for record in records if record.get("kind") == "step"]
    if not steps:
        raise ValueError(f"{path} contains no step records")
    return config, steps


def number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def chart(
    title: str,
    series: dict[str, list[tuple[float, float]]],
    *,
    logarithmic: bool = False,
    fixed_range: tuple[float, float] | None = None,
) -> str:
    width, height = 900, 300
    left, right, top, bottom = 70, 20, 20, 48
    points = [(x, y) for values in series.values() for x, y in values if math.isfinite(y)]
    if not points:
        return ""
    if logarithmic:
        points = [(x, y) for x, y in points if y > 0]
        if not points:
            return ""
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    x_low, x_high = min(x_values), max(x_values)
    if x_low == x_high:
        x_high += 1
    if fixed_range is None:
        transformed = [math.log10(value) for value in y_values] if logarithmic else y_values
        y_low, y_high = min(transformed), max(transformed)
        if y_low == y_high:
            y_low -= 0.5
            y_high += 0.5
        padding = (y_high - y_low) * 0.08
        y_low -= padding
        y_high += padding
    else:
        y_low, y_high = fixed_range
    inner_width, inner_height = width - left - right, height - top - bottom

    def x_coord(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * inner_width

    def y_coord(value: float) -> float:
        transformed = math.log10(value) if logarithmic else value
        return top + (y_high - transformed) / (y_high - y_low) * inner_height

    elements = [
        f'<h3>{html.escape(title)}</h3>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<line class="axis" x1="{left}" y1="{top + inner_height}" x2="{width - right}" y2="{top + inner_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_height}"/>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        y = top + inner_height * fraction
        transformed = y_high - fraction * (y_high - y_low)
        raw = 10**transformed if logarithmic else transformed
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        elements.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{number(raw)}</text>')
    for value, anchor in ((x_low, "start"), ((x_low + x_high) / 2, "middle"), (x_high, "end")):
        elements.append(
            f'<text x="{x_coord(value):.1f}" y="{height - 18}" text-anchor="{anchor}">{value:.0f}</text>'
        )
    for index, (label, values) in enumerate(series.items()):
        valid = [(x, y) for x, y in values if math.isfinite(y) and (not logarithmic or y > 0)]
        if not valid:
            continue
        path = " ".join(
            ("M" if item == 0 else "L") + f" {x_coord(x):.1f} {y_coord(y):.1f}"
            for item, (x, y) in enumerate(valid)
        )
        color = COLORS[index % len(COLORS)]
        elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + index % 4 * 195
        legend_y = 12 + index // 4 * 14
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 14}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 19}" y="{legend_y + 4}">{html.escape(label)}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def metric_series(
    steps: list[dict[str, object]],
    getter,
) -> dict[str, list[tuple[float, float]]]:
    output: dict[str, list[tuple[float, float]]] = {}
    for record in steps:
        step = float(record["step"])
        for label, value in getter(record).items():
            if value is not None:
                output.setdefault(label, []).append((step, float(value)))
    return output


def dashboard(path: Path) -> str:
    config, steps = load_profile(path)
    full_trace = "parameters" in steps[0]
    loss = metric_series(steps, lambda record: {"loss": record["loss"]})
    global_grad = metric_series(
        steps,
        lambda record: {
            "global gradient norm": record.get(
                "global_gradient_norm_before_clip", record.get("global_gradient_norm_raw")
            )
        },
    )
    block_grad = metric_series(
        steps,
        lambda record: {
            group: values["gradient_rms"]
            for group, values in record.get("parameter_groups", {}).items()
            if group.startswith("block_")
        },
    )
    update_ratio = metric_series(
        steps,
        lambda record: {
            group: values["update_to_parameter"]
            for group, values in record.get("parameter_groups", {}).items()
            if group.startswith("block_")
        },
    )
    change_ratio = metric_series(
        steps,
        lambda record: {
            call: values["change_rms"] / max(values["output_rms"], 1e-12)
            for call, values in record.get("activation", {}).items()
            if values.get("change_rms") is not None
        },
    )
    alignment = metric_series(
        steps,
        lambda record: {
            f"{pair} {group}": cosine
            for pair, groups in record.get("gradient_alignment", record.get("fixed_probe", {}))
            .get("cosine_by_group", {})
            .items()
            for group, cosine in groups.items()
            if group.startswith("block_")
        },
    )
    norm_scales = metric_series(
        steps,
        lambda record: {
            name: values["parameter_rms"]
            for name, values in (
                record.get("rmsnorm_parameters")
                or {
                    name: values
                    for name, values in record.get("parameters", {}).items()
                    if "norm.weight" in name
                }
            ).items()
        },
    )
    fixed_probe_norm = metric_series(
        steps,
        lambda record: {
            f"after {horizon} repeat(s)": value
            for horizon, value in record.get("gradient_alignment", record.get("fixed_probe", {}))
            .get("gradient_norm_by_horizon", {})
            .items()
        },
    )
    component_charts = []
    if full_trace:
        component_names = sorted(
            {
                name
                for record in steps
                for name in record.get("parameters", {})
                if name.startswith("blocks.") and name.endswith(".weight")
            }
        )
        block_indices = sorted({name.split(".")[1] for name in component_names})
        for index in block_indices:
            names = [name for name in component_names if name.startswith(f"blocks.{index}.")]
            component_gradient = metric_series(
                steps,
                lambda record, names=names: {
                    name.removeprefix(f"blocks.{index}."):
                    record.get("parameters", {}).get(name, {}).get("gradient_rms_raw")
                    for name in names
                },
            )
            component_charts.append(
                chart(f"Block {index} component gradient RMS", component_gradient, logarithmic=True)
            )
    config_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in config.items()
        if key != "kind"
    )
    charts = [
        chart("Training loss", loss, logarithmic=True),
        chart("Global gradient norm before clipping", global_grad, logarithmic=True),
        chart("Per-block gradient RMS (after global clipping)", block_grad, logarithmic=True),
        chart("Per-block update / parameter RMS", update_ratio, logarithmic=True),
        chart("Learned RMSNorm scale RMS", norm_scales, logarithmic=False),
        chart("Relative residual change for each block use", change_ratio, logarithmic=False),
        chart("Fixed-probe gradient norm by recurrence horizon", fixed_probe_norm, logarithmic=True),
        chart("Gradient cosine: answer after one versus two repeats", alignment, fixed_range=(-1.0, 1.0)),
        *component_charts,
    ]
    visible_charts = "\n".join(item for item in charts if item)
    return f"""
<section>
  <h2>{html.escape(path.name)}</h2>
  <table>{config_rows}</table>
  {visible_charts}
</section>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.profiles[0].with_suffix(".html")
    body = "\n".join(dashboard(path) for path in args.profiles)
    output.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Optimization profiles</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 28px; color: #172033; background: #f8fafc; }}
section {{ max-width: 960px; background: white; padding: 18px 28px; margin-bottom: 28px; border-radius: 10px; box-shadow: 0 1px 5px #cbd5e1; }}
h2 {{ margin: 0 0 12px; }} h3 {{ margin: 26px 0 4px; font-size: 15px; }}
table {{ border-collapse: collapse; font-size: 13px; }} th, td {{ padding: 3px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }} th {{ color: #475569; }}
svg {{ width: 100%; height: auto; overflow: visible; }} svg text {{ fill: #475569; font-size: 11px; }}
.axis {{ stroke: #64748b; stroke-width: 1; }} .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
</style></head><body><h1>Optimization profile dashboard</h1>{body}</body></html>""",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
