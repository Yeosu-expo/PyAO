import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_stats(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    for phase in ("forward", "backward"):
        stats.setdefault(phase, {})
    return stats


def get_top_time_layers(
    baseline: Dict[str, float], optimized: Dict[str, float], top_k: int
) -> List[Tuple[str, float, float]]:
    layers = set(baseline) | set(optimized)
    entries = []
    for name in layers:
        base_val = baseline.get(name, 0.0)
        opt_val = optimized.get(name, 0.0)
        entries.append((name, base_val, opt_val, max(base_val, opt_val)))

    entries.sort(key=lambda x: x[3], reverse=True)
    return [(n, b, o) for n, b, o, _ in entries[:top_k]]


def get_top_speedup_layers(
    baseline: Dict[str, float], optimized: Dict[str, float], top_k: int
) -> List[Tuple[str, float, float, float]]:
    layers = set(baseline) | set(optimized)
    entries = []
    for name in layers:
        base_val = baseline.get(name, 0.0)
        opt_val = optimized.get(name, 0.0)
        diff = base_val - opt_val
        if diff > 0:
            entries.append((name, base_val, opt_val, diff))

    entries.sort(key=lambda x: x[3], reverse=True)
    return entries[:top_k]


def plot_grouped_bars(
    entries: List[Tuple[str, float, float]],
    title: str,
    output_path: str,
    ylabel: str = "Time (s)",
):
    if not entries:
        print(f"[WARN] No data available for {title}; skipping {output_path}")
        return

    layers = [n for n, _, _ in entries]
    baseline_vals = [b for _, b, _ in entries]
    optimized_vals = [o for _, _, o in entries]
    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.8), 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    ax.bar(x + width / 2, optimized_vals, width, label="Optimized")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Layer timing optimization analysis")
    parser.add_argument("baseline", help="Path to baseline layer_timer_stats.json")
    parser.add_argument("optimized", help="Path to optimized layer_timer_stats.json")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("/home","deepspeed", "outputs", "PyAO", "plots", "experiments", "analysis"),
        help="Directory to save comparison graphs",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of layers to show")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_stats = load_stats(args.baseline)
    optimized_stats = load_stats(args.optimized)

    for phase in ("forward", "backward"):
        base_phase = baseline_stats.get(phase, {})
        opt_phase = optimized_stats.get(phase, {})

        top_time = get_top_time_layers(base_phase, opt_phase, args.top_k)
        plot_grouped_bars(
            top_time,
            f"Top {args.top_k} layers by time ({phase})",
            os.path.join(args.output_dir, f"graph1_{phase}_top_time.png"),
        )

        top_speedups = get_top_speedup_layers(base_phase, opt_phase, args.top_k)
        plot_grouped_bars(
            [(n, b, o) for n, b, o, _ in top_speedups],
            f"Top {args.top_k} speedups ({phase})",
            os.path.join(args.output_dir, f"graph2_{phase}_speedups.png"),
        )


if __name__ == "__main__":
    main()
