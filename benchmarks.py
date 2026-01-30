"""
SGLang Benchmark Runner - Simple Data Collection
Runs a single benchmark and saves raw data. No plotting.
"""

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
        
    async def send_request(self, session: aiohttp.ClientSession, request_data: Dict, request_id: int) -> RequestResult:
        """Send a single request and collect metrics."""
        
        # Wait until scheduled arrival time
        wait_time = request_data['arrival_time'] - (time.time() - self.benchmark_start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        actual_start = time.time()
        
        payload = {
            "text": request_data['prefix_text'],
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
                
                # Use server-reported metrics
                server_e2e = meta.get('e2e_latency', completion_time - actual_start)
                server_queue_time = meta.get('queue_time', 0)
                server_prefill = meta.get('prefill_launch_latency', 0)
                
                # Calculate metrics
                ttft = server_queue_time + server_prefill
                decode_time = server_e2e - ttft
                
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
    
    async def run_benchmark(self, workload: List[Dict]) -> List[RequestResult]:
        """Run the full benchmark."""
        
        print(f"Running benchmark with {len(workload)} requests...")
        self.benchmark_start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.send_request(session, req, idx)
                for idx, req in enumerate(workload)
            ]
            
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
            self.results = [r for r in self.results if isinstance(r, RequestResult)]
        
        duration = time.time() - self.benchmark_start_time
        print(f"Benchmark completed in {duration:.2f}s")
        print(f"Successful requests: {len(self.results)}/{len(workload)}")
        
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
                    
                    # Collect key metrics
                    if any(key in metric_name for key in [
                        'num_used_tokens', 'token_usage', 'max_total_num_tokens',
                        'gen_throughput', 'cache_hit_rate'
                    ]):
                        metrics[metric_name] = value
        
        return metrics


def calculate_statistics(results: List[RequestResult]) -> Dict:
    """Calculate comprehensive statistics."""
    
    if not results:
        return {}
    
    # Extract latencies
    ttft = [r.time_to_first_token for r in results]
    prefill = [r.prefill_time for r in results]
    decode = [r.decode_time for r in results]
    total = [r.total_latency for r in results]
    
    # Per-token decode latency
    inter_token_latency = []
    for r in results:
        if r.output_tokens > 0:
            inter_token_latency.append(r.decode_time / r.output_tokens)
    
    def percentiles(data):
        if not data:
            return {}
        return {
            'min': float(np.min(data)),
            'p50': float(np.percentile(data, 50)),
            'p90': float(np.percentile(data, 90)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }
    
    # Token counts
    total_input_tokens = sum(r.prefix_tokens + r.new_input_tokens for r in results)
    total_output_tokens = sum(r.output_tokens for r in results)
    total_duration = max(r.completion_time for r in results)
    
    stats = {
        'num_requests': len(results),
        'total_duration': total_duration,
        
        # Latency distributions
        'time_to_first_token': percentiles(ttft),
        'prefill_latency': percentiles(prefill),
        'decode_latency': percentiles(decode),
        'inter_token_latency': percentiles(inter_token_latency),
        'total_latency': percentiles(total),
        
        # Throughput
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'throughput_tokens_per_sec': (total_input_tokens + total_output_tokens) / total_duration,
        'throughput_requests_per_sec': len(results) / total_duration,
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
    print("\nCollecting initial server metrics...")
    initial_metrics = await runner.collect_metrics()
    
    # Run benchmark
    print()
    results = await runner.run_benchmark(workload)
    
    # Collect final metrics
    print("\nCollecting final server metrics...")
    final_metrics = await runner.collect_metrics()
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(results)
    
    # Add server metrics
    stats['initial_server_metrics'] = initial_metrics
    stats['final_server_metrics'] = final_metrics
    
    # Extract KV cache info if available
    if 'sglang:max_total_num_tokens' in final_metrics:
        stats['peak_kv_cache_tokens'] = final_metrics['sglang:max_total_num_tokens']
    if 'sglang:num_used_tokens' in final_metrics:
        stats['used_kv_cache_tokens'] = final_metrics['sglang:num_used_tokens']
    
    # Save results
    print(f"\nSaving results to {args.output}_*")
    
    with open(f"{args.output}_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    with open(f"{args.output}_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
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
    
    if 'peak_kv_cache_tokens' in stats:
        print(f"\nKV Cache:")
        print(f"  Peak: {stats['peak_kv_cache_tokens']:.0f} tokens")
    
    print("="*60)
    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
