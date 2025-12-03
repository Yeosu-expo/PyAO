import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_stats(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    for phase in ("forward", "backward"):
        stats.setdefault(phase, {})
    return stats


def compute_improvements(baseline: Dict[str, float], optimized: Dict[str, float]) -> List[float]:
    layers = set(baseline) | set(optimized)
    improvements = []
    eps = 1e-12
    for name in layers:
        base_val = baseline.get(name, 0.0)
        opt_val = optimized.get(name, 0.0)
        if base_val <= 0.0 and opt_val <= 0.0:
            pct = 0.0
        else:
            pct = 100.0 * (base_val - opt_val) / max(base_val, eps)
        pct = max(-100.0, min(100.0, pct))
        improvements.append(pct)
    return improvements


def plot_histogram(values: List[float], title: str, output_path: str):
    if not values:
        print(f"[WARN] No data available for {title}; skipping {output_path}")
        return

    min_val = min(values)
    if min_val < 0:
        xmin = max(-100, 5 * (int(min_val // 5) - 1))
    else:
        xmin = 0
    bins = np.linspace(xmin, 100, int((100 - xmin) // 5) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=bins, range=(xmin, 100), color="#4f81bd", edgecolor="black")
    ax.set_xlim(xmin, 100)
    ax.set_xticks(np.arange(xmin, 101, 10))
    ax.set_xlabel("Improvement (%) [-100 worse, +100 better]")
    ax.set_ylabel("Layer Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[INFO] Saved {output_path}")


def plot_cdf(values: List[float], title: str, output_path: str):
    if not values:
        print(f"[WARN] No data available for {title}; skipping {output_path}")
        return

    sorted_vals = np.sort(values)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    if sorted_vals[0] < 0:
        xmin = max(-100, 5 * (int(sorted_vals[0] // 5) - 1))
    else:
        xmin = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_vals, cumulative, marker="", color="#c0504d")
    ax.set_xlim(xmin, 100)
    ax.set_xticks(np.arange(xmin, 101, 10))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Improvement (%) [-100 worse, +100 better]")
    ax.set_ylabel("Cumulative Fraction of Layers")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_stats = load_stats(args.baseline)
    optimized_stats = load_stats(args.optimized)

    for phase in ("forward", "backward"):
        base_phase = baseline_stats.get(phase, {})
        opt_phase = optimized_stats.get(phase, {})
        improvements = compute_improvements(base_phase, opt_phase)

        plot_histogram(
            improvements,
            f"{phase.capitalize()} Improvement Distribution (Histogram)",
            os.path.join(args.output_dir, f"hist_{phase}.png"),
        )

        plot_cdf(
            improvements,
            f"{phase.capitalize()} Improvement Distribution (CDF)",
            os.path.join(args.output_dir, f"cdf_{phase}.png"),
        )


if __name__ == "__main__":
    main()
