from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


@dataclass
class TextSource:
    tokens: list[int]
    tokenizer: AutoTokenizer

    @classmethod
    def load(
        cls,
        text_path: Path | str,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    ) -> TextSource:
        text_path = Path(text_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        max_chars = tokenizer.max_len_single_sentence
        with open(text_path, encoding="utf-8") as f:
            text = f.read(max_chars)

        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) < 10_000:
            raise ValueError(f"text source too small: {len(tokens)} tokens. ")

        return cls(tokens=tokens, tokenizer=tokenizer)

    def get_prompt(self, num_tokens: int, offset: int = 0) -> str:
        """Get a prompt with approximately the specified number of tokens."""
        total = len(self.tokens)
        start = offset % total

        if start + num_tokens <= total:
            segment = self.tokens[start : start + num_tokens]
        else:
            segment = self.tokens[start:] + self.tokens[: num_tokens - (total - start)]

        return self.tokenizer.decode(segment, skip_special_tokens=True)


@dataclass(order=True)
class Request:
    arrival_time: float
    session_id: int = field(compare=False)
    turn_idx: int = field(compare=False)
    prefix_tokens: int = field(compare=False)
    new_input_tokens: int = field(compare=False)
    output_tokens: int = field(compare=False)
    prefix_text: str | None = field(default=None, compare=False)
    query_text: str | None = field(default=None, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def generate_multiturn_workload(
    num_sessions: int = 100,
    turns_per_session: tuple = (3, 10),
    initial_context_tokens: tuple = (1000, 4000),
    query_tokens: tuple = (50, 200),
    response_tokens: tuple = (100, 500),
    session_arrival_rate: float = 1.0,  # New sessions per second
    inter_turn_delay: tuple = (5.0, 30.0),  # User think time between turns
) -> list[Request]:
    """Generate workload with realistic arrival patterns."""

    requests = []
    session_start_time = 0.0

    for session_id in range(num_sessions):
        num_turns = random.randint(*turns_per_session)
        initial_ctx = random.randint(*initial_context_tokens)
        accumulated_tokens = initial_ctx

        turn_time = session_start_time

        for turn_idx in range(num_turns):
            q_tokens = random.randint(*query_tokens)
            r_tokens = random.randint(*response_tokens)

            requests.append(
                Request(
                    arrival_time=turn_time,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    prefix_tokens=accumulated_tokens,
                    new_input_tokens=q_tokens,
                    output_tokens=r_tokens,
                )
            )

            # Estimate completion time, then add think time
            # Rough estimate: prefill + decode time
            estimated_completion = (
                turn_time + (accumulated_tokens + q_tokens) * 0.001 + r_tokens * 0.05
            )
            turn_time = estimated_completion + random.uniform(*inter_turn_delay)

            accumulated_tokens += q_tokens + r_tokens

        # Next session arrives (Poisson)
        session_start_time += np.random.exponential(1.0 / session_arrival_rate)

    return sorted(requests, key=lambda r: r.arrival_time)


def analyze_concurrency_profile(requests: list[Request]) -> dict:
    """Analyze instantaneous concurrency over time."""

    events = []  # (time, delta) where delta is +1 for arrival, -1 for completion

    for req in requests:
        events.append((req.arrival_time, +1))
        # Estimate completion time
        prefill_time = (req.prefix_tokens + req.new_input_tokens) * 0.001
        decode_time = req.output_tokens * 0.05
        completion_time = req.arrival_time + prefill_time + decode_time
        events.append((completion_time, -1))

    events.sort(key=lambda x: x[0])

    concurrency = 0
    max_concurrency = 0
    concurrency_histogram = defaultdict(float)
    last_time = 0.0

    for time, delta in events:
        if time > last_time:
            concurrency_histogram[concurrency] += time - last_time
        concurrency += delta
        max_concurrency = max(max_concurrency, concurrency)
        last_time = time

    return {
        "max_concurrency": max_concurrency,
        "concurrency_histogram": dict(concurrency_histogram),
        "total_duration": last_time,
    }


def populate_prompts(requests: list[Request], text_source: TextSource) -> list[Request]:
    """
    populate prompt text for each request from the text source.

    turn 0: generates both prefix_text (initial context) and query_text.
    subsequent turns: generates only query_text.
    """
    # offsets per session
    session_offsets: dict[int, int] = {}

    for request in requests:
        session_id = request.session_id

        if session_id not in session_offsets:
            session_offsets[session_id] = session_id * 5000

        offset = session_offsets[session_id]

        if request.turn_idx == 0:
            request.prefix_text = text_source.get_prompt(
                request.prefix_tokens,
                offset=offset,
            )
            request.query_text = text_source.get_prompt(
                request.new_input_tokens,
                offset=offset + request.prefix_tokens,
            )
        else:
            request.query_text = text_source.get_prompt(
                request.new_input_tokens,
                offset=offset + request.turn_idx * 500,
            )

        session_offsets[session_id] = offset + 1000

    return requests


def sweep_concurrency_experiments(
    concurrency_levels: list[int],
    sessions_per_experiment: int,
    text_source: TextSource,
) -> list[dict]:
    """
    Generate workloads targeting different concurrency levels.

    Key insight: To achieve target concurrency C, we need arrival_rate ≈ C / avg_request_duration

    Args:
        concurrency_levels: List of target concurrency levels.
        sessions_per_experiment: Number of sessions per experiment.
        text_source: Text source for generating prompts.
    """
    # Estimate average request duration (prefill + decode)
    avg_prefix = 2500  # midpoint of (1000, 4000)
    avg_query = 125  # midpoint of (50, 200)
    avg_response = 300  # midpoint of (100, 500)

    # Average per-turn duration estimate
    avg_prefill_time = (avg_prefix + avg_query) * 0.001  # ~2.6s
    avg_decode_time = avg_response * 0.05  # ~15s
    avg_request_duration = avg_prefill_time + avg_decode_time  # ~17.6s

    experiments = []

    for target_concurrency in concurrency_levels:
        # Little's law: L = λW → λ = L/W
        # But we also have inter-turn delays, so adjust empirically
        arrival_rate = target_concurrency / (
            avg_request_duration + 15
        )  # +15 for think time

        workload = generate_multiturn_workload(
            num_sessions=sessions_per_experiment,
            session_arrival_rate=arrival_rate,
            inter_turn_delay=(5.0, 25.0),
        )

        # Populate prompts from text source
        workload = populate_prompts(workload, text_source)

        profile = analyze_concurrency_profile(workload)

        experiments.append(
            {
                "target_concurrency": target_concurrency,
                "arrival_rate": arrival_rate,
                "actual_max_concurrency": profile["max_concurrency"],
                "workload": [r.to_dict() for r in workload],
            }
        )

        print(
            f"Target: {target_concurrency:3d}, "
            f"Arrival rate: {arrival_rate:.3f}/s, "
            f"Actual max: {profile['max_concurrency']}"
        )

    return experiments


def get_cli_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="generate_workload",
        description="generate synthetic multi-turn workloads for llm serving framework benchmarks.",
    )
    parser.add_argument(
        "text_source",
        type=Path,
        metavar="TEXT_FILE",
        help="text source file for generating prompts",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        metavar="DIR",
        default=Path("workload/"),
        help="output directory for workload files (default: %(default)s)",
    )
    parser.add_argument(
        "--conc-levels",
        "-c",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128],
        help="target concurrency levels to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--sessions",
        "-s",
        type=int,
        default=200,
        help="number of sessions per concurrency level (default: %(default)s)",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model ID for tokenizer (default: %(default)s)",
    )

    return parser


if __name__ == "__main__":
    args = get_cli_parser().parse_args()

    text_source = TextSource.load(
        text_path=args.text_source,
        model_id=args.model_id,
    )

    experiments = sweep_concurrency_experiments(
        concurrency_levels=args.conc_levels,
        sessions_per_experiment=args.sessions,
        text_source=text_source,
    )

    if not args.out_dir.exists():
        args.out_dir.mkdir()

    for exp in experiments:
        filename = args.out_dir / f"workload_conc_{exp['target_concurrency']}.json"
        with open(filename, "w") as f:
            json.dump(exp["workload"], f, indent=2)
        print(f"Saved {filename}")
