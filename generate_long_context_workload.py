import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from generate_workload import (
    Request,
    TextSource,
    analyze_concurrency_profile,
    populate_prompts,
)


def sweep_context_length_experiments(
    context_lengths: list[int],
    num_requests: int,
    text_source: TextSource,
    arrival_rate: float = 0.5,
    turns_per_session: tuple[int, int] | int = 1,
    inter_turn_delay: tuple[float, float] = (5.0, 15.0),
) -> list[dict]:
    """
    Generate workloads for varying initial context lengths.

    Args:
        context_lengths: List of target context lengths (tokens).
        num_requests: Number of requests per experiment (total across all turns).
        text_source: Text source for generating prompts.
        arrival_rate: Arrival rate (requests per second) for new sessions.
        turns_per_session: Number of turns to reach target context. Can be:
            - int: Fixed number of turns for all sessions (1 = single-turn mode)
            - tuple (min, max): Random number of turns per session in this range
        inter_turn_delay: Range (min, max) of seconds between turns in a session.
    """
    experiments = []

    for target_ctx in context_lengths:
        requests = []
        session_start_time = 0.0

        query_tokens = (50, 100)
        response_tokens = (100, 200)

        avg_query = (query_tokens[0] + query_tokens[1]) // 2
        avg_response = (response_tokens[0] + response_tokens[1]) // 2

        if isinstance(turns_per_session, int):
            turns_range = (turns_per_session, turns_per_session)
        else:
            turns_range = turns_per_session

        avg_turns = (turns_range[0] + turns_range[1]) // 2
        num_sessions = max(1, num_requests // avg_turns)

        for session_id in range(num_sessions):
            num_turns = random.randint(*turns_range)

            if num_turns == 1:
                q_tokens = random.randint(*query_tokens)
                r_tokens = random.randint(*response_tokens)

                adjusted_target_prefix = target_ctx - avg_query - avg_response
                adjusted_target_prefix = max(adjusted_target_prefix, 100)

                ctx_len = int(adjusted_target_prefix * random.uniform(0.95, 1.05))

                requests.append(
                    Request(
                        arrival_time=session_start_time,
                        session_id=session_id,
                        turn_idx=0,
                        prefix_tokens=ctx_len,
                        new_input_tokens=q_tokens,
                        output_tokens=r_tokens,
                    )
                )

                session_start_time += np.random.exponential(1.0 / arrival_rate)
            else:
                total_added = (avg_query + avg_response) * num_turns

                initial_ctx = max(100, target_ctx - total_added)

                accumulated_tokens = int(initial_ctx * random.uniform(0.95, 1.05))
                turn_time = session_start_time

                for turn_idx in range(num_turns):
                    q_tokens = random.randint(*query_tokens)
                    r_tokens = random.randint(*response_tokens)

                    prefix_tokens = accumulated_tokens

                    requests.append(
                        Request(
                            arrival_time=turn_time,
                            session_id=session_id,
                            turn_idx=turn_idx,
                            prefix_tokens=prefix_tokens,
                            new_input_tokens=q_tokens,
                            output_tokens=r_tokens,
                        )
                    )

                    accumulated_tokens += q_tokens + r_tokens

                    prefill_time = (prefix_tokens + q_tokens) * 0.001
                    decode_time = r_tokens * 0.05
                    completion_time = turn_time + prefill_time + decode_time

                    if turn_idx < num_turns - 1:
                        turn_time = completion_time + random.uniform(*inter_turn_delay)

                session_start_time += np.random.exponential(1.0 / arrival_rate)

        workload = sorted(requests, key=lambda r: (r.session_id, r.arrival_time))
        workload = populate_prompts(workload, text_source)

        profile = analyze_concurrency_profile(workload)

        experiments.append(
            {
                "target_context_len": target_ctx,
                "arrival_rate": arrival_rate,
                "turns_per_session": turns_per_session,
                "actual_max_concurrency": profile["max_concurrency"],
                "workload": [r.to_dict() for r in workload],
            }
        )

        if isinstance(turns_per_session, int):
            turns_display = str(turns_per_session)
        else:
            turns_display = f"{turns_per_session[0]}-{turns_per_session[1]}"

        print(
            f"Context: {target_ctx}, "
            f"Turns/session: {turns_display}, "
            f"Requests: {len(workload)}, "
            f"Duration: {profile['total_duration']:.1f}s"
        )

    return experiments


def get_cli_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="generate_long_context_workload",
        description="generate workloads with increasing context lengths.",
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
        default=Path("workload_long_ctx/"),
        help="output directory for workload files (default: %(default)s)",
    )
    parser.add_argument(
        "--min-context",
        type=int,
        default=1000,
        help="minimum context length (default: %(default)s)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="maximum context length (default: 90%% of model's max_position_embeddings)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of steps between min and max context (default: %(default)s)",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=50,
        help="number of requests per context level (default: %(default)s)",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="model ID for tokenizer (default: %(default)s)",
    )
    parser.add_argument(
        "--arrival-rate",
        "-r",
        type=float,
        default=0.5,
        help="arrival rate in requests per second (default: %(default)s)",
    )
    parser.add_argument(
        "--turns-per-session",
        "-t",
        type=int,
        nargs="+",
        default=[2, 7],
        metavar=("MIN", "MAX"),
        help="number of turns to reach target context length. Single value for fixed turns, "
        "two values for random range (default: 1, single-turn)",
    )
    parser.add_argument(
        "--inter-turn-delay",
        type=float,
        nargs=2,
        default=[5.0, 15.0],
        metavar=("MIN", "MAX"),
        help="range of seconds between turns (default: %(default)s)",
    )

    return parser


if __name__ == "__main__":
    from transformers import AutoConfig

    args = get_cli_parser().parse_args()

    if args.max_context is None:
        model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        max_model_len = getattr(model_config, "max_position_embeddings", 32000)
        args.max_context = int(max_model_len * 0.9)
        print(f"Using max_context={args.max_context} (90% of {max_model_len})")

    text_source = TextSource.load(
        text_path=args.text_source,
        model_id=args.model_id,
    )

    context_lengths = np.linspace(
        args.min_context, args.max_context, args.steps, dtype=int
    ).tolist()

    context_lengths = sorted(list(set(context_lengths)))

    if len(args.turns_per_session) == 1:
        turns_per_session = args.turns_per_session[0]
        turns_display = str(turns_per_session)
    elif len(args.turns_per_session) == 2:
        turns_per_session = tuple(args.turns_per_session)
        turns_display = f"{turns_per_session[0]}-{turns_per_session[1]}"
    else:
        raise ValueError("--turns-per-session must have 1 or 2 values")

    print(f"Generating workloads for context lengths: {context_lengths}")
    print(f"Turns per session: {turns_display}")
    if isinstance(turns_per_session, tuple) or turns_per_session > 1:
        print(
            f"Inter-turn delay: {args.inter_turn_delay[0]}-{args.inter_turn_delay[1]}s"
        )

    experiments = sweep_context_length_experiments(
        context_lengths=context_lengths,
        num_requests=args.num_requests,
        text_source=text_source,
        arrival_rate=args.arrival_rate,
        turns_per_session=turns_per_session,
        inter_turn_delay=tuple(args.inter_turn_delay),
    )

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    for exp in experiments:
        ctx = exp["target_context_len"]
        turns = exp["turns_per_session"]

        if isinstance(turns, tuple):
            filename = (
                args.out_dir / f"workload_ctx_{ctx}_turns_{turns[0]}-{turns[1]}.json"
            )
        elif turns > 1:
            filename = args.out_dir / f"workload_ctx_{ctx}_turns_{turns}.json"
        else:
            filename = args.out_dir / f"workload_ctx_{ctx}.json"

        with open(filename, "w") as f:
            json.dump(exp["workload"], f, indent=2)
        print(f"Saved {filename}")
