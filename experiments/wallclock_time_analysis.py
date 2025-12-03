import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt


def load_stats(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats.setdefault("forward", {})
    stats.setdefault("backward", {})
    return stats


def summarize_phase(values: Dict[str, float]) -> Tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    total = sum(values.values())
    avg = total / len(values)
    return total, avg, len(values)


def plot_summary(phase: str, total: float, avg: float, count: int, output_path: str):
    if count == 0:
        print(f"[WARN] No data for {phase}; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ["Total", "Average"]
    values = [total, avg]
    colors = ["#4f81bd", "#c0504d"]

    ax.bar(bars, values, color=colors)
    ax.set_ylabel("Time (s)")
    ax.set_title(
        f"{phase.capitalize()} Wallclock Summary\n(total={total:.4f}s, layers={count})"
    )
    for idx, val in enumerate(values):
        ax.text(idx, val, f"{val:.4f}s", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Wallclock time summary analysis")
    parser.add_argument("--stats", help="Path to layer_timer_stats.json")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "PyAO", "plots", "wallclock_analysis"),
        help="Directory to save summary graphs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stats = load_stats(args.stats)
    for phase in ("forward", "backward"):
        total, avg, count = summarize_phase(stats.get(phase, {}))
        plot_summary(
            phase,
            total,
            avg,
            count,
            os.path.join(args.output_dir, f"{phase}_wallclock_summary.png"),
        )


if __name__ == "__main__":
    main()
