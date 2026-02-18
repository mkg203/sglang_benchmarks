import logging
import requests
from multiprocessing.synchronize import Event
from time import sleep
import json

METRICS_TO_TRACK = [
    "num_used_tokens",
    "token_usage",
    "max_total_num_tokens",
    "gen_throughput",
    "cache_hit_rate",
    "num_retractions",
    "kv_transfer_alloc_ms"
]

def collect_metrics(server_url: str, stop_event: Event, output_prefix: str) -> None:
    """Collect and parse metrics from the server's Prometheus endpoint."""
    logging.basicConfig(
        filename="metrics.log",
        level=logging.DEBUG,
        format='%(asctime)s - %(message)s',
        force=True 
    )

    logging.info("Metrics colletion has started")
    
    metrics = []

    with requests.session() as session:
        while not stop_event.is_set():
            try:
                response = session.get(f"{server_url}/metrics", timeout=2)
                if response.status_code != 200:
                    logging.warning(
                        f"Metrics endpoint returned {response.status_code}"
                    )
                    continue
                metrics.append(_parse_prometheus_metrics(response.text))
            except Exception as e:
                logging.warning(f"Failed to collect server metrics: {e}")
            sleep(0.5)

    with open(f"{output_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def _parse_prometheus_metrics(metrics_text: str) -> dict[str, float]:
    """Extract relevant metrics from Prometheus text format."""
    metrics = {}
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        # Expecting format: metric_name{labels} value
        parts = line.split()
        if len(parts) < 2:
            continue

        full_name, value_str = parts[0], parts[1]
        metric_name = full_name.split("{")[0]

        if any(k in metric_name for k in METRICS_TO_TRACK):
            try:
                metrics[metric_name] = float(value_str)
            except ValueError:
                pass
    return metrics


def augment_stats_with_server_metrics(stats: dict, initial: dict, final: dict) -> dict:
    """Add KV cache, cache hit rate, and preemption metrics to stats."""

    # 1. KV Cache Usage (Snapshot at end)
    used = final.get("sglang:num_used_tokens", 0)
    total = final.get("sglang:max_total_num_tokens", 0)

    stats["server_metrics"] = {
        "kv_cache_usage_tokens": used,
        "kv_cache_capacity_tokens": total,
        "kv_cache_usage_pct": (used / total * 100) if total > 0 else 0.0,
        "prefix_cache_hit_rate": final.get("sglang:cache_hit_rate", 0),
    }

    # 2. Preemptions (Delta)
    # Using 'get' with 0 default to handle cases where metric is missing
    init_retractions = initial.get("sglang:num_retractions", 0)
    final_retractions = final.get("sglang:num_retractions", 0)
    stats["server_metrics"]["num_preemptions"] = max(
        0, final_retractions - init_retractions
    )

    return stats
