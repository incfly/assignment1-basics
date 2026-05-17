import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


COLORS = {
    "mmap": "#2563eb",
    "load": "#dc2626",
}


def _nice_ticks(max_value: float, count: int = 5) -> list[float]:
    if max_value <= 0:
        return [0.0]
    return [max_value * i / count for i in range(count + 1)]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _phase_label(mode: str, phase: str) -> str:
    names = {
        "mmap_open": "open",
        "load_copy": "copy",
        "iterate": "iterate",
        "done": "done",
    }
    return f"{mode}: {names.get(phase, phase)}"


def read_rows(csv_path: Path, metric: str) -> dict[str, list[dict[str, str]]]:
    rows_by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get(metric):
                rows_by_mode[row["mode"]].append(row)
    return rows_by_mode


def write_svg(csv_path: Path, output_path: Path, metric: str = "rss_bytes") -> None:
    rows_by_mode = read_rows(csv_path, metric)
    if not rows_by_mode:
        raise ValueError(f"no rows with metric {metric} in {csv_path}")

    width, height = 980, 620
    left, right, top, bottom = 86, 28, 34, 84
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_x = max(float(row["elapsed_s"]) for rows in rows_by_mode.values() for row in rows)
    max_y = max(float(row[metric]) / (1024 * 1024) for rows in rows_by_mode.values() for row in rows)
    max_x = max(max_x, 1.0)
    max_y = max(max_y, 1.0)

    def sx(x: float) -> float:
        return left + (x / max_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y / max_y) * plot_h

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.tick{font-size:12px;fill:#4b5563}.label{font-size:14px;font-weight:700}.note{font-size:12px}</style>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]

    for tick in _nice_ticks(max_x):
        x = sx(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle">{tick:.1f}s</text>')

    for tick in _nice_ticks(max_y):
        y = sy(tick)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.0f}</text>')

    svg.append(f'<text class="label" x="{width / 2:.2f}" y="{height - 24}" text-anchor="middle">elapsed seconds</text>')
    svg.append(
        f'<text class="label" x="20" y="{top + plot_h / 2:.2f}" transform="rotate(-90 20 {top + plot_h / 2:.2f})" text-anchor="middle">{html.escape(metric)} MiB</text>'
    )

    legend_x = left + 12
    legend_y = top + 20
    phase_annotations: list[tuple[float, str, str]] = []
    for i, (mode, rows) in enumerate(sorted(rows_by_mode.items())):
        points = [(sx(float(row["elapsed_s"])), sy(float(row[metric]) / (1024 * 1024))) for row in rows]
        color = COLORS.get(mode, "#16a34a")
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{_polyline(points)}"/>'
        )
        y = legend_y + i * 24
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text class="note" x="{legend_x + 32}" y="{y + 4}">{html.escape(mode)}</text>')

        peak = max(rows, key=lambda row: float(row[metric]))
        peak_x = sx(float(peak["elapsed_s"]))
        peak_y_mib = float(peak[metric]) / (1024 * 1024)
        peak_y = sy(peak_y_mib)
        svg.append(f'<circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="4" fill="{color}"/>')
        svg.append(
            f'<text class="note" x="{min(peak_x + 8, left + plot_w - 130):.2f}" y="{max(peak_y - 8, top + 14):.2f}" fill="{color}">peak {html.escape(mode)}: {peak_y_mib:.0f} MiB</text>'
        )

        seen_phases: set[str] = set()
        for row in rows:
            phase = row["phase"]
            if phase in seen_phases or phase == "startup":
                continue
            seen_phases.add(phase)
            x = sx(float(row["elapsed_s"]))
            svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="{color}" stroke-dasharray="4 5" opacity="0.35"/>')
            phase_annotations.append((x, color, _phase_label(mode, phase)))

    lanes: list[float] = []
    for x, color, label in sorted(phase_annotations):
        label_width = 7 * len(label)
        lane = 0
        while lane < len(lanes) and x - lanes[lane] < 24 + label_width:
            lane += 1
        if lane == len(lanes):
            lanes.append(-1_000_000)
        lanes[lane] = x
        label_x = min(max(x + 5, left + 4), left + plot_w - label_width - 4)
        label_y = top + 30 + lane * 19
        svg.append(
            f'<text class="note" x="{label_x:.2f}" y="{label_y:.2f}" fill="{color}">{html.escape(label)}</text>'
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render loader memory CSV as an SVG plot.")
    parser.add_argument("csv", nargs="?", type=Path, default=Path("runtime-memory.csv"))
    parser.add_argument("--output", type=Path, default=Path("runtime-memory.svg"))
    parser.add_argument("--metric", default="rss_bytes")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_svg(args.csv, args.output, metric=args.metric)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
