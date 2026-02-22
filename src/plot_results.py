"""
LLM Serving Benchmark Plotting Script
======================================
Generates comparative plots between cpu_cache and non-cpu_cache runs.

Directory structure expected:
  results/
    <date>/
      workload_ctx_{ctx}_conc_16_results.json
      workload_ctx_{ctx}_conc_16_cpu_cache_results.json
      workload_ctx_{ctx}_conc_16_metrics.json
      workload_ctx_{ctx}_conc_16_cpu_cache_metrics.json

Usage:
  python plot_results.py
  python plot_results.py --results-dir /path/to/results --date 2-19 --ctx 2048 4096
  python plot_results.py --bucket-size 10 --hit-threshold 0.5
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLOR_NO_CACHE    = "#4C72B0"
COLOR_CPU_CACHE   = "#DD8452"
COLOR_HIT         = "#2ca02c"
COLOR_MISS        = "#d62728"
COLOR_RETRACTIONS = "#9467bd"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path):
    with open(path) as f:
        return json.load(f)

def load_metrics(path):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)

def get_paths(results_dir, date, ctx):
    base = results_dir / date
    return (
        base / f"workload_ctx_{ctx}_conc_16_results.json",
        base / f"workload_ctx_{ctx}_conc_16_cpu_cache_results.json",
        base / f"workload_ctx_{ctx}_conc_16_metrics.json",
        base / f"workload_ctx_{ctx}_conc_16_cpu_cache_metrics.json",
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cache_hit_rate(entry):
    total = entry["prefix_tokens"] + entry["new_input_tokens"]
    return entry["cached_tokens"] / total if total > 0 else 0.0

def group_by_turn(records):
    buckets = defaultdict(list)
    for r in records:
        buckets[r["turn_idx"]].append(r)
    return buckets

def per_session_avg_latency(records):
    buckets = defaultdict(list)
    for r in records:
        buckets[r["session_id"]].append(r["total_latency"])
    return {sid: np.mean(lats) for sid, lats in buckets.items()}

def cumulative_cpu_cache_size(records):
    session_max = {}
    times, sizes = [], []
    cumulative = 0
    for r in sorted(records, key=lambda x: x["actual_start_time"]):
        sid = r["session_id"]
        prefix = r["prefix_tokens"]
        prev = session_max.get(sid, 0)
        if prefix > prev:
            cumulative += prefix - prev
            session_max[sid] = prefix
        times.append(r["actual_start_time"])
        sizes.append(cumulative)
    return np.array(times), np.array(sizes)

def retraction_deltas_bucketed(metrics, bucket_size):
    if not metrics:
        return np.array([])
    cumulative = np.array([m.get("sglang:num_retractions_count", 0.0) for m in metrics])
    per_second = np.clip(np.diff(cumulative, prepend=cumulative[0]), 0, None)
    n = int(np.ceil(len(per_second) / bucket_size))
    bucketed = np.zeros(n)
    for i in range(n):
        bucketed[i] = per_second[i * bucket_size : (i + 1) * bucket_size].sum()
    return bucketed

def hit_miss_by_bucket(records, bucket_size, hit_threshold=0.5):
    valid = [r for r in records if r["prefix_tokens"] > 0]
    if not valid:
        return np.array([0.0]), np.array([0]), np.array([0])
    max_time = max(r["actual_start_time"] for r in records)
    n = int(np.ceil((max_time + 1) / bucket_size))
    hits   = np.zeros(n, dtype=int)
    misses = np.zeros(n, dtype=int)
    for r in valid:
        ratio = r["cached_tokens"] / r["prefix_tokens"]
        b = min(int(r["actual_start_time"] // bucket_size), n - 1)
        if ratio >= hit_threshold:
            hits[b] += 1
        else:
            misses[b] += 1
    return np.arange(n, dtype=float) * bucket_size, hits, misses


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_latency_vs_turn(ax, records_no, records_cpu, ctx):
    def summarise(records):
        by_turn = group_by_turn(records)
        turns = sorted(by_turn)
        med = [np.median([r["total_latency"] for r in by_turn[t]]) for t in turns]
        p25 = [np.percentile([r["total_latency"] for r in by_turn[t]], 25) for t in turns]
        p75 = [np.percentile([r["total_latency"] for r in by_turn[t]], 75) for t in turns]
        return np.array(turns), np.array(med), np.array(p25), np.array(p75)

    for records, label, color in [
        (records_no,  "No offload", COLOR_NO_CACHE),
        (records_cpu, "CPU cache",  COLOR_CPU_CACHE),
    ]:
        turns, med, p25, p75 = summarise(records)
        ax.plot(turns, med, marker="o", markersize=4, label=label, color=color)
        ax.fill_between(turns, p25, p75, alpha=0.18, color=color)
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Total latency (s)")
    ax.set_title(f"Latency vs Turn  [ctx={ctx}]")
    ax.legend()


def plot_cache_hit_rate_vs_turn(ax, records_no, records_cpu, ctx):
    def summarise(records):
        by_turn = group_by_turn(records)
        turns = sorted(by_turn)
        rates = [np.median([_cache_hit_rate(r) for r in by_turn[t]]) * 100 for t in turns]
        return np.array(turns), np.array(rates)

    for records, label, color in [
        (records_no,  "No offload", COLOR_NO_CACHE),
        (records_cpu, "CPU cache",  COLOR_CPU_CACHE),
    ]:
        turns, rates = summarise(records)
        ax.plot(turns, rates, marker="s", markersize=4, label=label, color=color)
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Cache-hit rate (%)")
    ax.set_title(f"Cache-Hit Rate vs Turn  [ctx={ctx}]")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()


def plot_ttft_vs_turn(ax, records_no, records_cpu, ctx):
    def summarise(records):
        by_turn = group_by_turn(records)
        turns = sorted(by_turn)
        meds = [np.median([r["time_to_first_token"] for r in by_turn[t]]) for t in turns]
        return np.array(turns), np.array(meds)

    for records, label, color in [
        (records_no,  "No offload", COLOR_NO_CACHE),
        (records_cpu, "CPU cache",  COLOR_CPU_CACHE),
    ]:
        turns, meds = summarise(records)
        ax.plot(turns, meds, marker="^", markersize=4, label=label, color=color)
    ax.set_xlabel("Turn index")
    ax.set_ylabel("TTFT (s)")
    ax.set_title(f"Time-to-First-Token vs Turn  [ctx={ctx}]")
    ax.legend()


def plot_cpu_cache_growth(ax, records_cpu, ctx):
    times, sizes = cumulative_cpu_cache_size(records_cpu)
    ax.plot(times, sizes / 1000, color=COLOR_CPU_CACHE, linewidth=1.5)
    ax.fill_between(times, sizes / 1000, alpha=0.15, color=COLOR_CPU_CACHE)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Approx. cached tokens (k)")
    ax.set_title(f"CPU Cache Growth Over Time  [ctx={ctx}]")


def plot_avg_latency_per_session(ax, records_no, records_cpu, ctx):
    avg_no  = per_session_avg_latency(records_no)
    avg_cpu = per_session_avg_latency(records_cpu)
    sessions = sorted(set(avg_no) | set(avg_cpu))
    x = np.arange(len(sessions))
    width = 0.38
    ax.bar(x - width/2, [avg_no.get(s, np.nan)  for s in sessions],
           width, label="No offload", color=COLOR_NO_CACHE,  alpha=0.85)
    ax.bar(x + width/2, [avg_cpu.get(s, np.nan) for s in sessions],
           width, label="CPU cache",  color=COLOR_CPU_CACHE, alpha=0.85)
    step = max(1, len(sessions) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(sessions[i]) for i in range(0, len(sessions), step)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Session ID")
    ax.set_ylabel("Avg total latency (s)")
    ax.set_title(f"Avg Latency per Session  [ctx={ctx}]")
    ax.legend()


def plot_hits_misses_retractions(
    records_no, metrics_no,
    records_cpu, metrics_cpu,
    ctx, bucket_size=5, hit_threshold=0.5,
):
    """
    Side-by-side: no-offload | cpu-cache.
    Stacked hit/miss bars per time bucket + retractions line on right axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Cache Hits/Misses & Retractions per {bucket_size}s Bucket  |  ctx={ctx}",
        fontsize=13,
    )

    configs = [
        ("No offload", records_no,  metrics_no,  axes[0]),
        ("CPU cache",  records_cpu, metrics_cpu, axes[1]),
    ]

    # Shared y-axis scale for bars
    y_max = max(
        (h + m).max()
        for _, rec, _, _ in configs
        for _, h, m in [hit_miss_by_bucket(rec, bucket_size, hit_threshold)]
    ) * 1.2

    for label, records, metrics, ax in configs:
        edges, hits, misses = hit_miss_by_bucket(records, bucket_size, hit_threshold)
        n = len(edges)
        x = np.arange(n)

        ax.bar(x, hits,   width=0.8, label="Hit",  color=COLOR_HIT,  alpha=0.85)
        ax.bar(x, misses, width=0.8, label="Miss", color=COLOR_MISS, alpha=0.85,
               bottom=hits)
        ax.set_ylim(0, y_max)
        ax.set_xlabel(f"Time (s)  [bucket = {bucket_size}s]")
        ax.set_ylabel("Request count")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=9)

        step = max(1, n // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{int(e)}" for e in edges[::step]])

        # Retractions on twin axis
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        if metrics:
            ret = retraction_deltas_bucketed(metrics, bucket_size)
            ret = np.pad(ret, (0, max(0, n - len(ret))))[:n]
            ax2.plot(x, ret, color=COLOR_RETRACTIONS, linewidth=2,
                     marker="o", markersize=3, label="Retractions", zorder=5)
            ax2.set_ylabel("Retractions (per bucket)", color=COLOR_RETRACTIONS)
            ax2.tick_params(axis="y", labelcolor=COLOR_RETRACTIONS)
            ax2.legend(loc="upper right", fontsize=9)
        else:
            ax2.set_yticks([])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def make_plots(results_dir, date, ctx_list, out_dir, bucket_size=5, hit_threshold=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ctx in ctx_list:
        path_no, path_cpu, path_met_no, path_met_cpu = get_paths(results_dir, date, ctx)

        missing = [p for p in (path_no, path_cpu) if not p.exists()]
        if missing:
            print(f"[WARN] Skipping ctx={ctx}: missing files: {missing}")
            continue

        print(f"Loading ctx={ctx} ...")
        records_no  = load_results(path_no)
        records_cpu = load_results(path_cpu)
        metrics_no  = load_metrics(path_met_no)
        metrics_cpu = load_metrics(path_met_cpu)

        if not metrics_no:
            print(f"  [INFO] No metrics for no-offload run ({path_met_no.name})")
        if not metrics_cpu:
            print(f"  [INFO] No metrics for cpu-cache run ({path_met_cpu.name})")

        # Fig 1: session metrics vs turn
        fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))
        fig1.suptitle(f"Session Metrics vs Turn  |  ctx={ctx}  |  {date}", fontsize=13, y=1.01)
        plot_latency_vs_turn(axes1[0], records_no, records_cpu, ctx)
        plot_cache_hit_rate_vs_turn(axes1[1], records_no, records_cpu, ctx)
        plot_ttft_vs_turn(axes1[2], records_no, records_cpu, ctx)
        fig1.tight_layout()
        out1 = out_dir / f"ctx{ctx}_session_metrics.png"
        fig1.savefig(out1, bbox_inches="tight")
        print(f"  Saved -> {out1}")

        # Fig 2: CPU cache growth
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.suptitle(f"CPU Cache Growth  |  ctx={ctx}  |  {date}", fontsize=13)
        plot_cpu_cache_growth(ax2, records_cpu, ctx)
        fig2.tight_layout()
        out2 = out_dir / f"ctx{ctx}_cpu_cache_growth.png"
        fig2.savefig(out2, bbox_inches="tight")
        print(f"  Saved -> {out2}")

        # Fig 3: avg latency per session
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        fig3.suptitle(f"Avg Latency per Session  |  ctx={ctx}  |  {date}", fontsize=13)
        plot_avg_latency_per_session(ax3, records_no, records_cpu, ctx)
        fig3.tight_layout()
        out3 = out_dir / f"ctx{ctx}_avg_latency_per_session.png"
        fig3.savefig(out3, bbox_inches="tight")
        print(f"  Saved -> {out3}")

        # Fig 4: hits/misses + retractions per time bucket
        fig4 = plot_hits_misses_retractions(
            records_no,  metrics_no,
            records_cpu, metrics_cpu,
            ctx=ctx, bucket_size=bucket_size, hit_threshold=hit_threshold,
        )
        out4 = out_dir / f"ctx{ctx}_hits_misses_retractions.png"
        fig4.savefig(out4, bbox_inches="tight")
        print(f"  Saved -> {out4}")

        plt.close("all")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot LLM benchmark results.")
    p.add_argument("--results-dir",   type=Path,  default=Path("results"))
    p.add_argument("--date",          default="2-22")
    p.add_argument("--ctx",           type=int,   nargs="+", default=[2048])
    p.add_argument("--out-dir",       type=Path,  default=Path("plots"))
    p.add_argument("--bucket-size",   type=int,   default=5,
                   help="Time bucket width in seconds (default: 5)")
    p.add_argument("--hit-threshold", type=float, default=0.5,
                   help="cached_tokens/prefix_tokens ratio to count as a hit (default: 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plots(
        results_dir=args.results_dir,
        date=args.date,
        ctx_list=args.ctx,
        out_dir=args.out_dir,
        bucket_size=args.bucket_size,
        hit_threshold=args.hit_threshold,
    )
