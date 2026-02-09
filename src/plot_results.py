#!/usr/bin/env python3
"""
Plot SGLang Benchmark Results
Generates comparison plots from saved benchmark data in a directory.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import re


def load_results(stats_file: str, results_file: str) -> Dict:
    """Load statistics and raw results."""
    with open(stats_file, "r") as f:
        stats = json.load(f)

    with open(results_file, "r") as f:
        results = json.load(f)

    return {"stats": stats, "results": results}


def find_result_pairs(directory: str) -> List[Dict]:
    """Find all matching *_stats.json and *_results.json pairs in directory."""
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"ERROR: Directory '{directory}' does not exist")
        return []

    # Find all stats files
    stats_files = list(dir_path.glob("*_stats.json"))

    pairs = []
    for stats_file in stats_files:
        # Get the prefix (everything before _stats.json)
        prefix = str(stats_file.stem).replace("_stats", "")
        results_file = dir_path / f"{prefix}_results.json"

        if results_file.exists():
            pairs.append(
                {
                    "prefix": prefix,
                    "stats_file": str(stats_file),
                    "results_file": str(results_file),
                }
            )
        else:
            print(f"WARNING: Found {stats_file.name} but missing {results_file.name}")

    # Sort by prefix for consistent ordering
    pairs.sort(key=lambda x: x["prefix"])

    return pairs


def calculate_inter_token_latency(results: List[Dict]) -> Dict[str, float]:
    """Calculate inter-token latency percentiles from raw results."""
    itl_values = []
    for r in results:
        if r["output_tokens"] > 0:
            itl = r["decode_time"] / r["output_tokens"]
            itl_values.append(itl)

    if not itl_values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    return {
        "min": np.min(itl_values),
        "p50": np.percentile(itl_values, 50),
        "p90": np.percentile(itl_values, 90),
        "p95": np.percentile(itl_values, 95),
        "p99": np.percentile(itl_values, 99),
        "max": np.max(itl_values),
    }


def plot_comparison(configs: List[Dict], output_file: str):
    """
    Generate comparison plot.

    configs: List of dicts with:
        - name: Configuration name
        - stats: Statistics dict
        - results: Raw results list
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SGLang Benchmark Results", fontsize=16, fontweight="bold")

    metrics = [
        ("time_to_first_token", "Prefill Latency (TTFT)", axes[0, 0]),
        ("decode_latency", "Decode Latency", axes[0, 1]),
        ("inter_token_latency", "Inter-Token Latency", axes[1, 0]),
        ("total_latency", "Total End-to-End Latency", axes[1, 1]),
    ]

    percentile_keys = ["min", "p50", "p90", "p95", "p99", "max"]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c", "#9b59b6"]

    x_positions = np.arange(len(configs))
    bar_width = 0.12

    for metric_key, title, ax in metrics:
        for i, pct in enumerate(percentile_keys):
            values = []

            for config in configs:
                if metric_key == "inter_token_latency":
                    # Calculate from raw results
                    metric_data = calculate_inter_token_latency(config["results"])
                elif metric_key == "decode_latency":
                    # Get from stats
                    metric_data = config["stats"].get("decode_latency", {})
                else:
                    # Get from stats
                    metric_data = config["stats"].get(metric_key, {})

                values.append(metric_data.get(pct, 0))

            positions = x_positions + (i - 2.5) * bar_width
            ax.bar(positions, values, bar_width, label=pct, color=colors[i], alpha=0.8)

        ax.set_xlabel("Configuration", fontweight="bold")
        ax.set_ylabel("Latency (seconds)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([c["name"] for c in configs], rotation=15, ha="right")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to {output_file}")


def plot_throughput_comparison(configs: List[Dict], output_file: str):
    """Generate throughput comparison bar chart."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Throughput Comparison", fontsize=14, fontweight="bold")

    names = [c["name"] for c in configs]
    token_throughput = [c["stats"]["throughput_tokens_per_sec"] for c in configs]
    request_throughput = [c["stats"]["throughput_requests_per_sec"] for c in configs]

    x_pos = np.arange(len(names))

    ax1.bar(x_pos, token_throughput, color="#3498db", alpha=0.8)
    ax1.set_xlabel("Configuration", fontweight="bold")
    ax1.set_ylabel("Tokens/sec", fontweight="bold")
    ax1.set_title("Token Throughput")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x_pos, request_throughput, color="#e74c3c", alpha=0.8)
    ax2.set_xlabel("Configuration", fontweight="bold")
    ax2.set_ylabel("Requests/sec", fontweight="bold")
    ax2.set_title("Request Throughput")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Throughput plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SGLang benchmark results from a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
          # Plot all results in a directory
          python plot_results.py results/
          
          # Specify output files
          python plot_results.py results/ --output my_comparison.png
                """,
    )
    parser.add_argument(
        "directory", help="Directory containing *_stats.json and *_results.json files"
    )
    parser.add_argument(
        "--output",
        default="benchmark_comparison.png",
        help="Output latency comparison plot filename",
    )
    parser.add_argument(
        "--throughput-plot",
        default="throughput_comparison.png",
        help="Output throughput plot filename",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SGLang Benchmark Plotter")
    print("=" * 60)
    print(f"Directory: {args.directory}")
    print()

    # Find all result pairs in directory
    pairs = find_result_pairs(args.directory)

    if not pairs:
        print("ERROR: No result files found in directory")
        print(
            "Expected files matching pattern: <prefix>_stats.json and <prefix>_results.json"
        )
        return

    print(f"Found {len(pairs)} result pair(s):")
    for pair in pairs:
        print(f"  - {pair['prefix']}")
    print()

    # Load all configurations into a temporary list
    loaded_configs = []
    for pair in pairs:
        print(f"Loading {pair['prefix']}...")
        data = load_results(pair["stats_file"], pair["results_file"])

        loaded_configs.append(
            {"name": pair["prefix"], "stats": data["stats"], "results": data["results"]}
        )

    # SORTING LOGIC: Sort by the numeric concurrency level found in the name
    # This ensures 8 comes before 16, and 16 before 32.
    configs = sorted(
        loaded_configs,
        key=lambda x: (
            int(re.search(r"conc_(\d+)", x["name"]).group(1))
            if re.search(r"conc_(\d+)", x["name"])
            else 0
        ),
    )

    print(
        f"\nGenerating plots for {len(configs)} configuration(s) (Sorted by concurrency)..."
    )

    # Generate filenames based on directory name
    base_prefix = args.directory.strip("/").replace("/", "_")

    # Generate plots using the sorted configs
    plot_comparison(configs, f"{base_prefix}_latency.png")
    plot_throughput_comparison(configs, f"{base_prefix}_throughput.png")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
