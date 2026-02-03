import json
import asyncio
import aiohttp
import time
import argparse
from typing import List, Dict
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path


@dataclass
class RequestResult:
    """Store metrics for a single request."""
    request_id: int
    session_id: int
    turn_idx: int
    prefix_tokens: int
    new_input_tokens: int
    output_tokens: int
    
    scheduled_arrival_time: float
    actual_start_time: float
    completion_time: float
    
    time_to_first_token: float  # TTFT (queue + prefill)
    prefill_time: float
    decode_time: float
    total_latency: float
    
    server_queue_time: float
    server_e2e_latency: float
    cached_tokens: int


class BenchmarkRunner:
    def __init__(self, server_url: str = "http://localhost:30000"):
        self.server_url = server_url
        self.results: List[RequestResult] = []
        self.benchmark_start_time = None
        self.metrics_log: List[Dict] = []  # Continuous metrics log
        self.polling_active = False
        
    async def send_request(self, session: aiohttp.ClientSession, request_data: Dict, request_id: int) -> RequestResult:
        """Send a single request and collect metrics."""
        
        # Wait until scheduled arrival time
        wait_time = request_data['arrival_time'] - (time.time() - self.benchmark_start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        actual_start = time.time()
        
        payload = {
            "text": request_data['text'],
            "sampling_params": {
                "max_new_tokens": request_data['output_tokens'],
                "temperature": 0.0
            }
        }
        
        try:
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                result = await response.json()
                completion_time = time.time()
                
                meta = result.get('meta_info', {})
                
                # DEBUG: Print first request to check format
                if request_id == 0:
                    print(f"\nDEBUG - First request meta_info keys: {list(meta.keys())}")
                    print(f"DEBUG - prefill_launch_latency value: {meta.get('prefill_launch_latency')}")
                
                # Use server-reported metrics
                server_e2e = meta.get('e2e_latency', completion_time - actual_start)
                server_queue_time = meta.get('queue_time', 0.0)
                
                # Get prefill latency - SGLang reports this in prefill_launch_latency
                server_prefill = meta.get('prefill_launch_latency', 0.0)
                if server_prefill == 0.0:
                    # Fallback: check for alternative field names
                    server_prefill = meta.get('prefill_time', 0.0)
                
                # Calculate metrics
                ttft = server_queue_time + server_prefill
                decode_time = max(0, server_e2e - ttft)  # Ensure non-negative
                
                return RequestResult(
                    request_id=request_id,
                    session_id=request_data['session_id'],
                    turn_idx=request_data['turn_idx'],
                    prefix_tokens=request_data['prefix_tokens'],
                    new_input_tokens=request_data['new_input_tokens'],
                    output_tokens=request_data['output_tokens'],
                    
                    scheduled_arrival_time=request_data['arrival_time'],
                    actual_start_time=actual_start - self.benchmark_start_time,
                    completion_time=completion_time - self.benchmark_start_time,
                    
                    time_to_first_token=ttft,
                    prefill_time=server_prefill,
                    decode_time=decode_time,
                    total_latency=server_e2e,
                    
                    server_queue_time=server_queue_time,
                    server_e2e_latency=server_e2e,
                    cached_tokens=meta.get('cached_tokens', 0)
                )
                
        except Exception as e:
            print(f"ERROR: Request {request_id} failed: {e}")
            raise
    
    async def poll_metrics(self, interval: float = 1.0):
        """Continuously poll /metrics endpoint during benchmark."""
        while self.polling_active:
            snapshot = await self.collect_metrics()
            if snapshot:
                snapshot['timestamp'] = time.time() - self.benchmark_start_time
                self.metrics_log.append(snapshot)
            await asyncio.sleep(interval)
    
    async def run_benchmark(self, workload: List[Dict]) -> List[RequestResult]:
        """Run the full benchmark."""
        
        print(f"Running benchmark with {len(workload)} requests...")
        self.benchmark_start_time = time.time()
        self.metrics_log = []
        
        # Start background metrics polling
        self.polling_active = True
        poll_task = asyncio.create_task(self.poll_metrics(interval=1.0))
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.send_request(session, req, idx)
                for idx, req in enumerate(workload)
            ]
            
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results = [r for r in self.results if isinstance(r, RequestResult)]
        
        # Stop polling
        self.polling_active = False
        await poll_task
        
        duration = time.time() - self.benchmark_start_time
        print(f"Benchmark completed in {duration:.2f}s")
        print(f"Successful requests: {len(self.results)}/{len(workload)}")
        print(f"Collected {len(self.metrics_log)} metrics snapshots")
        
        return self.results
    
    async def collect_metrics(self) -> Dict:
        """Collect server metrics from /metrics endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/metrics") as response:
                    metrics_text = await response.text()
                    return self.parse_prometheus_metrics(metrics_text)
        except Exception as e:
            print(f"WARNING: Failed to collect server metrics: {e}")
            return {}
    
    def parse_prometheus_metrics(self, metrics_text: str) -> Dict:
        """Parse key metrics from Prometheus format."""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('sglang:') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    metric_full = parts[0]
                    value = float(parts[1])
                    metric_name = metric_full.split('{')[0]
                    
                    if any(key in metric_name for key in [
                        'num_used_tokens', 'max_total_num_tokens',
                        'gen_throughput', 'cache_hit_rate',
                        'num_retractions', 'num_retracted_reqs',
                    ]):
                        metrics[metric_name] = value
        
        return metrics


def calculate_kv_cache_stats(metrics_log: List[Dict]) -> Dict:
    """Calculate KV cache and server-level stats from continuous metrics log."""
    
    if not metrics_log:
        return {}
    
    # Extract time series
    used_tokens = [m.get('sglang:num_used_tokens', 0) for m in metrics_log]
    max_tokens = [m.get('sglang:max_total_num_tokens', 0) for m in metrics_log]
    cache_hit_rates = [m.get('sglang:cache_hit_rate', 0) for m in metrics_log]
    retractions = [m.get('sglang:num_retractions_sum', 0) for m in metrics_log]
    retracted_reqs = [m.get('sglang:num_retracted_reqs', 0) for m in metrics_log]
    
    # Max total tokens is constant (pool size), take first non-zero value
    max_total = next((t for t in max_tokens if t > 0), 0)
    
    # KV cache usage percentages over time
    kv_usage_pct = []
    if max_total > 0:
        kv_usage_pct = [(u / max_total) * 100 for u in used_tokens]
    
    stats = {
        'kv_cache': {
            'max_total_tokens': max_total,
            'peak_used_tokens': float(max(used_tokens)) if used_tokens else 0,
            'avg_used_tokens': float(np.mean(used_tokens)) if used_tokens else 0,
            'peak_usage_pct': float(max(kv_usage_pct)) if kv_usage_pct else 0,
            'avg_usage_pct': float(np.mean(kv_usage_pct)) if kv_usage_pct else 0,
        },
        'prefix_cache': {
            'hit_rate_min': float(min(cache_hit_rates)) if cache_hit_rates else 0,
            'hit_rate_max': float(max(cache_hit_rates)) if cache_hit_rates else 0,
            'hit_rate_avg': float(np.mean(cache_hit_rates)) if cache_hit_rates else 0,
        },
        'preemptions': {
            # num_retractions is a histogram _sum, so total preemptions = last - first
            'total_retractions': float(retractions[-1] - retractions[0]) if len(retractions) >= 2 else 0,
            'peak_retracted_reqs': float(max(retracted_reqs)) if retracted_reqs else 0,
        }
    }
    
    return stats


async def main():
    parser = argparse.ArgumentParser(description="SGLang Benchmark Runner")
    parser.add_argument("workload", help="Path to workload JSON file")
    parser.add_argument("--output", required=True, help="Output file prefix (e.g., 'baseline_conc8')")
    parser.add_argument("--server", default="http://localhost:30000", help="SGLang server URL")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SGLang Benchmark Runner")
    print("="*60)
    print(f"Workload: {args.workload}")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}_*")
    print()
    
    # Load workload
    with open(args.workload, 'r') as f:
        workload = json.load(f)
    
    print(f"Loaded {len(workload)} requests")
    
    # Collect initial metrics
    runner = BenchmarkRunner(server_url=args.server)
    
    # Run benchmark (metrics are polled in background)
    print()
    results = await runner.run_benchmark(workload)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(results)
    
    # Calculate KV cache stats from polling log
    kv_stats = calculate_kv_cache_stats(runner.metrics_log)
    stats.update(kv_stats)
    
    # Save results
    print(f"\nSaving results to {args.output}_*")
    
    with open(f"{args.output}_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    with open(f"{args.output}_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save raw metrics log separately
    with open(f"{args.output}_metrics_log.json", 'w') as f:
        json.dump(runner.metrics_log, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Requests: {stats['num_requests']}")
    print(f"Duration: {stats['total_duration']:.2f}s")
    print(f"\nThroughput:")
    print(f"  {stats['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"  {stats['throughput_requests_per_sec']:.2f} requests/s")
    print(f"\nPrefill Latency (TTFT):")
    print(f"  P50: {stats['time_to_first_token']['p50']*1000:.2f}ms")
    print(f"  P99: {stats['time_to_first_token']['p99']*1000:.2f}ms")
    print(f"\nDecode Latency:")
    print(f"  P50: {stats['decode_latency']['p50']:.3f}s")
    print(f"  P99: {stats['decode_latency']['p99']:.3f}s")
    print(f"\nInter-Token Latency:")
    print(f"  P50: {stats['inter_token_latency']['p50']*1000:.2f}ms/token")
    print(f"  P99: {stats['inter_token_latency']['p99']*1000:.2f}ms/token")
    
    if 'kv_cache' in stats:
        print(f"\nKV Cache:")
        print(f"  Max pool size: {stats['kv_cache']['max_total_tokens']:.0f} tokens")
        print(f"  Peak used: {stats['kv_cache']['peak_used_tokens']:.0f} tokens")
        print(f"  Avg used: {stats['kv_cache']['avg_used_tokens']:.0f} tokens")
        print(f"  Peak usage: {stats['kv_cache']['peak_usage_pct']:.1f}%")
        print(f"  Avg usage: {stats['kv_cache']['avg_usage_pct']:.1f}%")
    
    if 'prefix_cache' in stats:
        print(f"\nPrefix Cache:")
        print(f"  Hit rate (avg): {stats['prefix_cache']['hit_rate_avg']*100:.1f}%")
        print(f"  Hit rate (max): {stats['prefix_cache']['hit_rate_max']*100:.1f}%")
    
    if 'preemptions' in stats:
        print(f"\nPreemptions:")
        print(f"  Total retractions: {stats['preemptions']['total_retractions']:.0f}")
        print(f"  Peak retracted requests: {stats['preemptions']['peak_retracted_reqs']:.0f}")
    
    print("="*60)
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
