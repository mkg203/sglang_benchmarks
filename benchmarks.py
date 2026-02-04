import asyncio
import argparse
import json
import time
import aiohttp
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

# --- Configuration & Constants ---
METRICS_TO_TRACK = [
    "num_used_tokens",
    "token_usage",
    "max_total_num_tokens",
    "gen_throughput",
    "cache_hit_rate",
    "num_retractions",
]


@dataclass
class RequestResult:
    """Store metrics for a single request."""

    request_id: tuple[int, int]
    session_id: int
    turn_idx: int
    prefix_tokens: int
    new_input_tokens: int
    output_tokens: int

    # Timing
    scheduled_arrival_time: float
    actual_start_time: float
    completion_time: float

    # Latencies
    time_to_first_token: float
    prefill_time: float
    decode_time: float
    total_latency: float
    server_queue_time: float
    server_e2e_latency: float

    # Meta
    cached_tokens: int


class BenchmarkRunner:
    def __init__(self, server_url: str = "http://localhost:30000"):
        self.server_url = server_url.rstrip("/")
        self.results: list[RequestResult] = []
        self.benchmark_start_time = 0.0
        self.session_histories = {}

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request_data: dict,
        request_id: tuple[int, int],
    ) -> RequestResult:
        """Send a single request and collect metrics."""
        # Calculate wait time relative to benchmark start
        target_time = self.benchmark_start_time + request_data["arrival_time"]
        wait_time = target_time - time.time()

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        actual_start_abs = time.time()

        session_id = request_data["session_id"]
        text = (
            self.session_histories.get(session_id, request_data.get("prefix_text", ""))
            + request_data["query_text"]
        )

        payload = {
            # "text": request_data.get("prefix_text", "") + request_data["query_text"],
            "text": text,
            "sampling_params": {
                "max_new_tokens": request_data["output_tokens"],
                "temperature": 0.0,
            },
        }

        try:
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as response:
                result = await response.json()
                completion_time_abs = time.time()

                meta = result.get("meta_info", {})
                # update session history
                self.session_histories[session_id] = text + result.get("text", "")

                # Server-reported metrics
                server_e2e = meta.get(
                    "e2e_latency", completion_time_abs - actual_start_abs
                )
                server_queue = meta.get("queue_time", 0)
                server_prefill = meta.get("prefill_launch_latency", 0)

                # Derived metrics
                ttft = server_queue + server_prefill
                decode_time = server_e2e - ttft

                return RequestResult(
                    request_id=request_id,
                    session_id=request_data["session_id"],
                    turn_idx=request_data["turn_idx"],
                    prefix_tokens=request_data["prefix_tokens"],
                    new_input_tokens=request_data["new_input_tokens"],
                    output_tokens=request_data["output_tokens"],
                    scheduled_arrival_time=request_data["arrival_time"],
                    actual_start_time=actual_start_abs - self.benchmark_start_time,
                    completion_time=completion_time_abs - self.benchmark_start_time,
                    time_to_first_token=ttft,
                    prefill_time=server_prefill,
                    decode_time=decode_time,
                    total_latency=server_e2e,
                    server_queue_time=server_queue,
                    server_e2e_latency=server_e2e,
                    cached_tokens=meta.get("cached_tokens", 0),
                )

        except Exception as e:
            print(f"ERROR: Request {request_id} failed: {e}")
            raise

    async def run_session(
        self, session: aiohttp.ClientSession, session_workload: list[dict], pbar: tqdm
    ) -> list[RequestResult]:
        session_results = []
        for request in session_workload:
            result = await self.send_request(
                session, request, (request["session_id"], request["turn_idx"])
            )
            session_results.append(result)
            pbar.update(1)

        return session_results

    async def run_benchmark(
        self, workload: dict[int, list[dict]]
    ) -> list[RequestResult]:
        """Run the full benchmark workload."""
        total_requests = sum(len(s_load) for s_load in workload.values())
        print(
            f"Running benchmark with {len(workload)} sessions and ({total_requests=}..."
        )

        self.benchmark_start_time = time.time()

        async with aiohttp.ClientSession() as session:
            with tqdm(
                total=total_requests, desc="Processing Requests", unit="req"
            ) as pbar:
                tasks = [
                    self.run_session(session, session_workload, pbar)
                    for session_id, session_workload in workload.items()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        self.results = [
            request
            for session_results in results
            if isinstance(session_results, list)
            for request in session_results
            if isinstance(request, RequestResult)
        ]

        duration = time.time() - self.benchmark_start_time
        print(f"Benchmark completed in {duration:.2f}s")
        print(f"Successful requests: {len(self.results)}/{len(workload)}")
        return self.results

    async def collect_metrics(self) -> Dict[str, float]:
        """Collect and parse metrics from the server's Prometheus endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/metrics") as response:
                    if response.status != 200:
                        print(f"WARNING: Metrics endpoint returned {response.status}")
                        return {}
                    return self._parse_prometheus_metrics(await response.text())
        except Exception as e:
            print(f"WARNING: Failed to collect server metrics: {e}")
            return {}

    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
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


def calculate_statistics(
    results: List[RequestResult], duration: float
) -> Dict[str, Any]:
    """Calculate comprehensive statistics from request results."""
    if not results:
        return {}

    # Extract arrays
    ttft = [r.time_to_first_token for r in results]
    prefill = [r.prefill_time for r in results]
    decode = [r.decode_time for r in results]
    total = [r.total_latency for r in results]

    inter_token_latency = [
        r.decode_time / r.output_tokens for r in results if r.output_tokens > 0
    ]

    def get_percentiles(data: List[float]) -> Dict[str, float]:
        if not data:
            return {}
        return {
            "min": float(np.min(data)),
            "p50": float(np.percentile(data, 50)),
            "p90": float(np.percentile(data, 90)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }

    total_input = sum(r.prefix_tokens + r.new_input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_tokens = total_input + total_output

    return {
        "num_requests": len(results),
        "total_duration": duration,
        "throughput_tokens_per_sec": total_tokens / duration if duration > 0 else 0,
        "throughput_requests_per_sec": len(results) / duration if duration > 0 else 0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "time_to_first_token": get_percentiles(ttft),
        "prefill_latency": get_percentiles(prefill),
        "decode_latency": get_percentiles(decode),
        "inter_token_latency": get_percentiles(inter_token_latency),
        "total_latency": get_percentiles(total),
    }


def augment_stats_with_server_metrics(stats: Dict, initial: Dict, final: Dict) -> Dict:
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


async def main():
    parser = argparse.ArgumentParser(description="SGLang Benchmark Runner")
    parser.add_argument("workload", type=Path, help="Path to workload JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Output file prefix")
    parser.add_argument(
        "--server", default="http://localhost:30000", help="SGLang server URL"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SGLang Benchmark Runner")
    print(f"Workload: {args.workload}")
    print(f"Server:   {args.server}")
    print(f"Output:   {args.output}_*")
    print("=" * 60)

    # Load workload
    n_requests: int
    with open(args.workload, "r") as f:
        data = json.load(f)
        n_requests = len(data)
        workload = {}
        for request in data:
            s_id = int(request.session_id)
            if s_id not in workload:
                workload[s_id] = [request]
                continue

            workload[s_id].append(request)
    print(f"Loaded {n_requests} requests")

    runner = BenchmarkRunner(server_url=args.server)

    # Initial Metrics
    print("\nCollecting initial server metrics...")
    initial_metrics = await runner.collect_metrics()

    # Run Benchmark
    print()
    results = await runner.run_benchmark(workload)

    # Final Metrics
    print("\nCollecting final server metrics...")
    final_metrics = await runner.collect_metrics()

    # Calculate Stats
    print("\nCalculating statistics...")
    duration = max((r.completion_time for r in results), default=0)
    stats = calculate_statistics(results, duration)

    # Add Server Metrics (KV, Preemptions, Cache Hit)
    stats = augment_stats_with_server_metrics(stats, initial_metrics, final_metrics)
    stats["raw_initial_metrics"] = initial_metrics
    stats["raw_final_metrics"] = final_metrics

    # Save
    print(f"\nSaving results to {args.output}_* ...")
    with open(f"{args.output}_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with open(f"{args.output}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    sm = stats["server_metrics"]
    ttft = stats["time_to_first_token"]
    itl = stats["inter_token_latency"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Requests:      {stats['num_requests']}")
    print(f"Duration:      {stats['total_duration']:.2f}s")
    print(f"Throughput:    {stats['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"               {stats['throughput_requests_per_sec']:.2f} req/s")

    print("-" * 60)
    print(f"Latency (P50 | P99)")
    print(
        f"TTFT:          {ttft.get('p50',0)*1000:.2f}ms | {ttft.get('p99',0)*1000:.2f}ms"
    )
    print(
        f"ITL:           {itl.get('p50',0)*1000:.2f}ms | {itl.get('p99',0)*1000:.2f}ms"
    )

    print("-" * 60)
    print("Server Metrics")
    print(
        f"KV Cache Usage:    {sm['kv_cache_usage_pct']:.2f}% ({int(sm['kv_cache_usage_tokens'])}/{int(sm['kv_cache_capacity_tokens'])})"
    )
    print(f"Prefix Cache Hit:  {sm['prefix_cache_hit_rate'] * 100:.2f}%")
    print(f"Preemptions:       {sm['num_preemptions']}")
    print("=" * 60)
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
