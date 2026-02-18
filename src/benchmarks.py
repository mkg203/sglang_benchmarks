import asyncio
import argparse
import json
import logging
import time
import aiohttp
import numpy as np
from typing import Any
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from src.metrics_tracker import collect_metrics
from multiprocessing import Process, Event
import datetime
import os
from transformers import AutoTokenizer


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
        self.failed_requests = 0

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        except OSError:
            logging.error("Could not load tokenizer. Ensure you have access to the model on Hugging Face.")
            raise
            
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request_data: dict,
        request_id: tuple[int, int],
    ) -> RequestResult:
        
        target_time = self.benchmark_start_time + request_data["arrival_time"]
        wait_time = target_time - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        session_id = request_data["session_id"]

        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
            if request_data.get("prefix_text"):
                self.session_histories[session_id].append({
                    "role": "system", 
                    "content": request_data["prefix_text"]
                })

        messages = self.session_histories[session_id]
        messages.append({"role": "user", "content": request_data["query_text"]})

        full_prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        payload = {
            "text": full_prompt_text,
            "sampling_params": {
                "max_new_tokens": request_data["output_tokens"],
                "temperature": 0.0,
                "ignore_eos": False
            },
        }
        
        actual_start_abs = time.time()

        try:
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as response:
                result = await response.json()
                completion_time_abs = time.time()

                meta = result.get("meta_info", {})
                server_e2e = meta.get("e2e_latency", completion_time_abs - actual_start_abs)
                server_queue = meta.get("queue_time", 0)
                server_prefill = meta.get("prefill_launch_latency", 0)
                
                cached_tokens = meta.get("cached_tokens", 0) 

                assistant_response = result.get("text", "")
                messages.append({"role": "assistant", "content": assistant_response})
                self.session_histories[session_id] = messages

                return RequestResult(
                    request_id=request_id,
                    session_id=session_id,
                    turn_idx=request_data["turn_idx"],
                    prefix_tokens=request_data["prefix_tokens"],
                    new_input_tokens=request_data["new_input_tokens"],
                    output_tokens=request_data["output_tokens"],
                    scheduled_arrival_time=request_data["arrival_time"],
                    actual_start_time=actual_start_abs - self.benchmark_start_time,
                    completion_time=completion_time_abs - self.benchmark_start_time,
                    time_to_first_token=server_queue + server_prefill,
                    prefill_time=server_prefill,
                    decode_time=server_e2e - (server_queue + server_prefill),
                    total_latency=server_e2e,
                    server_queue_time=server_queue,
                    server_e2e_latency=server_e2e,
                    cached_tokens=cached_tokens,
                )

        except Exception as e:
            logging.error(f"Request {request_id} failed: {e}")
            raise

    async def run_session(
        self, session: aiohttp.ClientSession, session_workload: list[dict], pbar: tqdm
    ) -> list[RequestResult]:
        session_results = []
        for request in session_workload:
            try:
                result = await self.send_request(
                    session, request, (request["session_id"], request["turn_idx"])
                )
                session_results.append(result)
            except Exception:
                self.failed_requests += 1

            pbar.update(1)

        return session_results

    async def run_benchmark(
        self, workload: dict[int, list[dict]]
    ) -> list[RequestResult]:
        """Run the full benchmark workload."""
        total_requests = sum(len(s_load) for s_load in workload.values())
        logging.info(
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
            # if isinstance(request, RequestResult)
        ]

        duration = time.time() - self.benchmark_start_time
        logging.info(f"Benchmark completed in {duration:.2f}s")
        logging.info(f"Successful requests: {len(self.results)}/{len(workload)}")
        return self.results


def calculate_statistics(
    results: list[RequestResult], duration: float
) -> dict[str, Any]:
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

    def get_percentiles(data: list[float]) -> dict[str, float]:
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


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="SGLang Benchmark Runner")
    parser.add_argument("workload", type=Path, help="Path to workload JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Output file prefix")
    parser.add_argument(
        "--server", default="http://localhost:30000", help="SGLang server URL"
    )
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info(f"SGLang Benchmark Runner")
    logging.info(f"Workload: {args.workload}")
    logging.info(f"Server:   {args.server}")
    logging.info(f"Output:   {args.output}_*")
    logging.info("=" * 60)

    # Load workload
    n_requests: int
    with open(args.workload, "r") as f:
        data = json.load(f)
        n_requests = len(data)
        workload = {}
        for request in data:
            s_id = int(request["session_id"])
            if s_id not in workload:
                workload[s_id] = [request]
                continue

            workload[s_id].append(request)
    logging.info(f"Loaded {n_requests} requests")

    runner = BenchmarkRunner(server_url=args.server)


    date = datetime.datetime.now()
    base_dir = f"results/{date.month}-{date.day}/"
    os.makedirs(base_dir, exist_ok=True)
    
    # Run Benchmark

    stop = Event()
    metrics_collection = Process(
        target=collect_metrics, args=(args.server, stop, f"{base_dir}{args.output}")
    )
    metrics_collection.start()
    
    try:
        results = await runner.run_benchmark(workload)
    finally:
        stop.set()
        metrics_collection.join()
        
    logging.info("Calculating statistics...")
    duration = max((r.completion_time for r in results), default=0)
    stats = calculate_statistics(results, duration)
    stats["failed_requests"] = runner.failed_requests

    # Save
    logging.info(f"Saving results to {args.output}_* ...")
    
    with open(f"{base_dir}{args.output}_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    with open(f"{base_dir}{args.output}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    ttft = stats["time_to_first_token"]
    itl = stats["inter_token_latency"]

    logging.info("=" * 60)
    logging.info("BENCHMARK SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Requests:      {stats['num_requests']}")
    logging.info(f"Failed Requests:      {stats['failed_requests']}")
    logging.info(f"Duration:      {stats['total_duration']:.2f}s")
    logging.info(f"Throughput:    {stats['throughput_tokens_per_sec']:.2f} tokens/s")
    logging.info(f"               {stats['throughput_requests_per_sec']:.2f} req/s")

    logging.info("-" * 60)
    logging.info(f"Latency (P50 | P99)")
    logging.info(
        f"TTFT:          {ttft.get('p50',0)*1000:.2f}ms | {ttft.get('p99',0)*1000:.2f}ms"
    )
    logging.info(
        f"ITL:           {itl.get('p50',0)*1000:.2f}ms | {itl.get('p99',0)*1000:.2f}ms"
    )

    logging.info("=" * 60)
    logging.info("✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
