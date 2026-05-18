from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _points(rows: list[dict], split: str) -> list[tuple[int, float]]:
    return [(int(row["iter"]), float(row["loss"])) for row in rows if row.get("split") == split]


def _polyline(points: list[tuple[int, float]], x_min: int, x_max: int, y_min: float, y_max: float) -> str:
    coords = []
    for x, y in points:
        px = 60 + (x - x_min) / max(x_max - x_min, 1) * 700
        py = 360 - (y - y_min) / max(y_max - y_min, 1e-9) * 300
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    root = Path(args.runs_root)
    metric_files = sorted(root.glob("**/metrics.jsonl"))
    series = []
    summary_rows = []
    for metrics_path in metric_files:
        rows = _read_jsonl(metrics_path)
        valid = _points(rows, "valid")
        train = _points(rows, "train")
        points = valid or train
        if not points:
            continue
        run_dir = metrics_path.parent
        label = str(run_dir.relative_to(root))
        lr = ""
        config_path = run_dir / "config.toml"
        if config_path.exists():
            with config_path.open("rb") as f:
                lr = str(tomllib.load(f).get("optimizer", {}).get("lr", ""))
        series.append((label, points))
        summary_rows.append(
            {
                "run": label,
                "lr": lr,
                "last_iter": points[-1][0],
                "last_loss": points[-1][1],
                "min_loss": min(loss for _, loss in points),
                "split": "valid" if valid else "train",
            }
        )

    summary_path = Path(args.summary) if args.summary else root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "lr", "split", "last_iter", "last_loss", "min_loss"])
        writer.writeheader()
        writer.writerows(summary_rows)

    if not series:
        print(f"wrote {summary_path}; no loss curves found")
        return

    all_points = [point for _, points in series for point in points]
    x_min = min(x for x, _ in all_points)
    x_max = max(x for x, _ in all_points)
    y_min = min(y for _, y in all_points)
    y_max = max(y for _, y in all_points)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430">',
        '<rect width="900" height="430" fill="white"/>',
        '<line x1="60" y1="360" x2="760" y2="360" stroke="black"/>',
        '<line x1="60" y1="60" x2="60" y2="360" stroke="black"/>',
        f'<text x="60" y="390" font-size="12">iter {x_min}</text>',
        f'<text x="700" y="390" font-size="12">iter {x_max}</text>',
        f'<text x="10" y="65" font-size="12">loss {y_max:.3f}</text>',
        f'<text x="10" y="360" font-size="12">loss {y_min:.3f}</text>',
    ]
    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{_polyline(points, x_min, x_max, y_min, y_max)}"/>'
        )
        lines.append(f'<text x="785" y="{70 + idx * 20}" font-size="12" fill="{color}">{label}</text>')
    lines.append("</svg>")

    out_path = Path(args.out) if args.out else root / "loss_curves.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
