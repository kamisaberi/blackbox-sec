#!/usr/bin/env python3
"""
scripts/plot_pareto_front.py
Pareto Frontier Visualization Script for blackbox-sec Campaigns

Author: Kamran Saberifard
License: Apache 2.0

Usage:
  python3 scripts/plot_pareto_front.py \
    --input ./reports/jailbreak_payload.json \
    --output ./reports/pareto_frontier.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class ParetoFrontPlotter:
    def __init__(self, input_json_path: Path, output_image_path: Path):
        self.input_json_path = input_json_path.resolve()
        self.output_image_path = output_image_path.resolve()

        if not self.input_json_path.exists():
            raise FileNotFoundError(f"Input JSON log file not found: {self.input_json_path}")

    def load_data(self) -> List[Dict]:
        """Loads evaluation points from JSON log file."""
        try:
            with open(self.input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "evaluations" in data:
                    return data["evaluations"]
                elif isinstance(data, list):
                    return data
                else:
                    # Fallback single evaluation entry
                    return [data]
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse JSON file: {err}")

    def compute_pareto_frontier(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Computes non-dominated Pareto frontier points (min distance, max success)."""
        if not points:
            return []

        # Sort by distance ascending, then success score descending
        sorted_points = sorted(points, key=lambda p: (p[0], -p[1]))

        frontier = []
        max_success = -1.0

        for dist, score in sorted_points:
            if score > max_success:
                frontier.append((dist, score))
                max_success = score

        return frontier

    def plot_and_save(self):
        """Generates matplotlib visualization and saves plot image."""
        evals = self.load_data()

        # Extract (perturbation_distance, fitness_score) coordinates
        points = []
        for e in evals:
            dist = float(e.get("perturbation_distance", e.get("queries_used", 0) / 1000.0))
            score = float(e.get("peak_fitness", e.get("fitness_score", 0.0)))
            points.append((dist, score))

        if not points:
            print("[WARNING] No evaluation points found to plot.")
            return

        frontier = self.compute_pareto_frontier(points)

        # Print Text Summary
        print("\n" + "=" * 60)
        print(" blackbox-sec Non-Dominated Pareto Frontier Points")
        print("=" * 60)
        print(f"{'Perturbation Distance D(x)':<30} | {'Attack Success F(x)':<25}")
        print("-" * 60)
        for dist, score in frontier:
            print(f"{dist:<30.4f} | {score:<25.4f}")
        print("=" * 60 + "\n")

        # Plot using Matplotlib if installed
        try:
            import matplotlib.pyplot as plt

            distances, scores = zip(*points)
            p_dist, p_scores = zip(*frontier) if frontier else ([], [])

            plt.figure(figsize=(10, 6), dpi=300)
            plt.scatter(distances, scores, color="blue", alpha=0.5, label="Evaluated Candidates")
            plt.step(p_dist, p_scores, color="red", where="post", linewidth=2.5, label="Pareto Frontier")

            plt.title("blackbox-sec: Perturbation Distance vs. Attack Success Rate", fontsize=12, fontweight="bold")
            plt.xlabel("Normalized Perturbation Distance D(x, x_orig)", fontsize=10)
            plt.ylabel("Objective Fitness / Attack Success Score F(x)", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(loc="lower right")
            plt.ylim(-0.05, 1.05)
            plt.xlim(-0.05, 1.05)

            self.output_image_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.output_image_path, bbox_inches="tight")
            plt.close()

            print(f"[SUCCESS] High-resolution Pareto plot saved to: {self.output_image_path.resolve()}\n")

        except ImportError:
            print("[NOTE] matplotlib is not installed. Skipping PNG image generation.")
            print("[NOTE] Install matplotlib via: pip install matplotlib\n")


def main():
    parser = argparse.ArgumentParser(description="blackbox-sec Pareto Frontier Plotting Utility")
    parser.add_argument("--input", default=Path("reports/jailbreak_payload.json"), type=Path, help="Path to input campaign log JSON")
    parser.add_argument("--output", default=Path("reports/pareto_frontier.png"), type=Path, help="Path to save output PNG plot")

    args = parser.parse_args()

    try:
        plotter = ParetoFrontPlotter(args.input, args.output)
        plotter.plot_and_save()
    except Exception as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()