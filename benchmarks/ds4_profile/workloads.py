# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import random
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


@cache
def _execution_token_mapping(
    tokenizer_path: str, source_vocab_size: int, seed: int
) -> tuple[int, ...]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    legal_ids = [
        token_id
        for token_id in range(len(tokenizer))
        if token_id not in special_ids
        and tokenizer.convert_ids_to_tokens(token_id) not in (None, "")
    ]
    if source_vocab_size > len(legal_ids):
        raise ValueError(
            "source vocabulary does not fit in legal execution-token pool: "
            f"{source_vocab_size} > {len(legal_ids)}"
        )
    random.Random(seed).shuffle(legal_ids)
    return tuple(legal_ids[:source_vocab_size])


def map_execution_tokens(
    sequences: list[list[int]],
    source_vocab_size: int,
    tokenizer_path: Path,
    seed: int,
) -> list[list[int]]:
    """Map logical tokens bijectively to deterministic legal Qwen tokens.

    Args:
        sequences: Logical token sequences whose equality must be preserved.
        source_vocab_size: Exclusive upper bound for logical token IDs.
        tokenizer_path: Local pinned Qwen tokenizer directory.
        seed: Seed selecting the deterministic legal-token permutation.

    Returns:
        Token sequences with identical lengths and prefix relationships.
    """
    if source_vocab_size <= 0:
        raise ValueError("source_vocab_size must be positive")
    mapping = _execution_token_mapping(str(tokenizer_path), source_vocab_size, seed)
    mapped = []
    for sequence in sequences:
        if any(token < 0 or token >= source_vocab_size for token in sequence):
            raise ValueError("logical token ID is outside source vocabulary")
        mapped.append([mapping[token] for token in sequence])
    return mapped


P_HOMOGENEOUS_LENGTHS = {
    1: (128, 512, 1024, 2048, 4096),
    2: (128, 512, 1024, 2048),
    4: (128, 512, 1024),
    8: (128, 256, 512),
}
TRACE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _turn_sort_key(turn: dict[str, Any]) -> tuple:
    return (
        turn["new_prefill_tokens"],
        turn["trajectory_id"],
        turn["turn_index"],
    )


def _turn_reference(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "completion_tokens": turn["completion_tokens"],
        "new_prefill_tokens": turn["new_prefill_tokens"],
        "prompt_tokens": turn["prompt_tokens"],
        "reasoning_mode": turn["reasoning_mode"],
        "reusable_prefix_tokens": turn["reusable_prefix_tokens"],
        "tokens_until_tool_ready": turn["tokens_until_tool_ready"],
        "trajectory_id": turn["trajectory_id"],
        "turn_index": turn["turn_index"],
    }


def _quantile_turns(turns: list[dict[str, Any]]) -> list[tuple[float, dict]]:
    ordered = sorted(turns, key=_turn_sort_key)
    return [
        (quantile, ordered[round(quantile * (len(ordered) - 1))])
        for quantile in TRACE_QUANTILES
    ]


def _mixed_turns(
    turns: list[dict[str, Any]],
    composition: str,
    batch_size: int,
    seed: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    ordered = sorted(turns, key=_turn_sort_key)
    per_request_limit = token_budget // batch_size
    eligible = [
        turn for turn in ordered if 0 < turn["new_prefill_tokens"] <= per_request_limit
    ]
    if len(eligible) < batch_size:
        raise ValueError(
            f"not enough turns for batch size {batch_size} under token budget"
        )

    if composition == "similar":
        windows = [
            eligible[index : index + batch_size]
            for index in range(len(eligible) - batch_size + 1)
        ]
        return min(
            windows,
            key=lambda window: (
                window[-1]["new_prefill_tokens"] - window[0]["new_prefill_tokens"],
                abs(len(eligible) // 2 - eligible.index(window[batch_size // 2])),
                tuple(_turn_sort_key(turn) for turn in window),
            ),
        )
    if composition == "random":
        shuffled = eligible.copy()
        random.Random(f"{seed}:{batch_size}").shuffle(shuffled)
        return shuffled[:batch_size]
    if composition == "high_skew":
        shortest = ordered[: batch_size - 1]
        remaining_budget = token_budget - sum(
            turn["new_prefill_tokens"] for turn in shortest
        )
        longest = next(
            turn
            for turn in reversed(ordered)
            if turn not in shortest and turn["new_prefill_tokens"] <= remaining_budget
        )
        return [*shortest, longest]
    raise ValueError(f"unsupported mixed composition: {composition}")


def build_workload_plan(
    turns: list[dict[str, Any]],
    seed: int,
    token_budget: int = 4096,
) -> dict[str, Any]:
    """Build deterministic controlled and trace-derived workload selections.

    Args:
        turns: Rendered DS4 turn contracts.
        seed: Seed for mixed-batch selection.
        token_budget: Maximum scheduled P tokens across one batch.

    Returns:
        JSON-serializable workload plan.
    """
    homogeneous = [
        {
            "batch_size": batch_size,
            "per_request_scheduled_tokens": per_request_tokens,
            "total_scheduled_tokens": batch_size * per_request_tokens,
        }
        for batch_size, lengths in P_HOMOGENEOUS_LENGTHS.items()
        for per_request_tokens in lengths
    ]

    quantile_buckets = []
    exact_replays = []
    for reasoning_mode in ("no_think", "think_high"):
        mode_turns = [
            turn for turn in turns if turn["reasoning_mode"] == reasoning_mode
        ]
        for quantile, turn in _quantile_turns(mode_turns):
            reference = _turn_reference(turn)
            quantile_buckets.append(
                {
                    "metric": "new_prefill_tokens",
                    "quantile": quantile,
                    "reasoning_mode": reasoning_mode,
                    "representative": reference,
                    "value": turn["new_prefill_tokens"],
                }
            )
            exact_replays.append(
                {
                    **reference,
                    "selection_metric": "new_prefill_tokens",
                    "selection_quantile": quantile,
                }
            )

    mixed_batches = []
    for composition in ("similar", "random", "high_skew"):
        for batch_size in (2, 4, 8):
            selected = _mixed_turns(
                turns,
                composition,
                batch_size,
                seed,
                token_budget,
            )
            references = [_turn_reference(turn) for turn in selected]
            mixed_batches.append(
                {
                    "batch_size": batch_size,
                    "composition": composition,
                    "seed": seed,
                    "total_scheduled_tokens": sum(
                        turn["new_prefill_tokens"] for turn in references
                    ),
                    "turns": references,
                }
            )

    cache_interleavings = []
    interleaving_seeds = (seed, seed + 1, seed + 2)
    for reasoning_mode in ("no_think", "think_high"):
        trajectory_ids = sorted(
            {
                turn["trajectory_id"]
                for turn in turns
                if turn["reasoning_mode"] == reasoning_mode
            }
        )
        for session_concurrency in (1, 2, 4, 8):
            selected_sessions = trajectory_ids[:session_concurrency]
            for mode in ("serial", "round_robin"):
                cache_interleavings.append(
                    {
                        "mode": mode,
                        "reasoning_mode": reasoning_mode,
                        "seed": None,
                        "session_concurrency": session_concurrency,
                        "trajectory_ids": selected_sessions,
                    }
                )
            for interleaving_seed in interleaving_seeds:
                cache_interleavings.append(
                    {
                        "mode": "seeded_random",
                        "reasoning_mode": reasoning_mode,
                        "seed": interleaving_seed,
                        "session_concurrency": session_concurrency,
                        "trajectory_ids": selected_sessions,
                    }
                )

    return {
        "cache_interleavings": cache_interleavings,
        "exact_replays": exact_replays,
        "mixed_batches": mixed_batches,
        "p_homogeneous": homogeneous,
        "schema_version": "1.0.0",
        "seeds": {
            "cache_interleavings": list(interleaving_seeds),
            "mixed_batches": seed,
        },
        "token_budget": token_budget,
        "trace_quantile_buckets": quantile_buckets,
    }


def build_artifacts(
    manifest_path: Path,
    normalized_turns_path: Path,
    ds4_tokenizer_path: Path,
    qwen_tokenizer_path: Path,
    output_dir: Path,
    block_size: int,
    seed: int,
) -> None:
    """Build deterministic Ticket 02 workload artifacts.

    Args:
        manifest_path: Ticket 01 manifest beside the raw snapshot.
        normalized_turns_path: Ticket 01 normalized turn Parquet.
        ds4_tokenizer_path: Pinned local DeepSeek V4 tokenizer.
        qwen_tokenizer_path: Pinned local Qwen execution tokenizer.
        output_dir: Directory for the three Ticket 02 artifacts.
        block_size: Prefix-cache block size in tokens.
        seed: Workload selection and execution mapping seed.
    """
    turns = render_turns(
        manifest_path=manifest_path,
        normalized_turns_path=normalized_turns_path,
        tokenizer_path=ds4_tokenizer_path,
        block_size=block_size,
        include_token_ids=True,
    )
    plan = build_workload_plan(turns, seed=seed)
    selected_keys = {
        (replay["trajectory_id"], replay["turn_index"])
        for replay in plan["exact_replays"]
    }
    selected_keys.update(
        (turn["trajectory_id"], turn["turn_index"])
        for batch in plan["mixed_batches"]
        for turn in batch["turns"]
    )

    selected_turns = [
        turn
        for turn in turns
        if (turn["trajectory_id"], turn["turn_index"]) in selected_keys
    ]
    logical_sequences = [
        sequence
        for turn in selected_turns
        for sequence in (
            turn["_prompt_token_ids"],
            turn["_completion_token_ids"],
        )
    ]
    ds4_tokenizer = AutoTokenizer.from_pretrained(
        ds4_tokenizer_path, local_files_only=True
    )
    execution_sequences = map_execution_tokens(
        logical_sequences,
        source_vocab_size=len(ds4_tokenizer),
        tokenizer_path=qwen_tokenizer_path,
        seed=seed,
    )
    execution_by_key = {}
    for index, turn in enumerate(selected_turns):
        key = (turn["trajectory_id"], turn["turn_index"])
        execution_by_key[key] = (
            execution_sequences[index * 2],
            execution_sequences[index * 2 + 1],
        )

    rows = []
    for turn in turns:
        row = {key: value for key, value in turn.items() if not key.startswith("_")}
        execution = execution_by_key.get((turn["trajectory_id"], turn["turn_index"]))
        row["execution_prompt_token_ids"] = (
            execution[0] if execution is not None else None
        )
        row["execution_completion_token_ids"] = (
            execution[1] if execution is not None else None
        )
        rows.append(row)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = {
        "artifact_schema_version": "1.0.0",
        "block_size": block_size,
        "dataset_revision": manifest["dataset"]["revision"],
        "ds4_tokenizer": {
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash",
            "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        },
        "execution_mapping_seed": seed,
        "manifest_sha256": _sha256(manifest_path),
        "normalized_turns_sha256": _sha256(normalized_turns_path),
        "qwen_tokenizer": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        },
        "rendered_turn_count": len(rows),
        "selected_execution_turn_count": len(selected_keys),
        "workload_seed": seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"schema_version": b"1.0.0"}
    )
    pq.write_table(table, output_dir / "rendered_turns.parquet")
    for name, artifact in (
        ("workload_plan.json", plan),
        ("provenance.json", provenance),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2, sort_keys=True)
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic DS4 profile workloads."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalized-turns", type=Path, required=True)
    parser.add_argument("--ds4-tokenizer", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    build_artifacts(
        manifest_path=args.manifest,
        normalized_turns_path=args.normalized_turns,
        ds4_tokenizer_path=args.ds4_tokenizer,
        qwen_tokenizer_path=args.qwen_tokenizer,
        output_dir=args.output_dir,
        block_size=args.block_size,
        seed=args.seed,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_deepseek_v4_encoding() -> ModuleType:
    source_path = (
        Path(__file__).parents[2] / "vllm" / "tokenizers" / "deepseek_v4_encoding.py"
    )
    if not source_path.is_file():
        try:
            distribution = importlib.metadata.distribution("vllm")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "cannot locate the installed vLLM distribution"
            ) from error
        source_path = Path(
            distribution.locate_file("vllm/tokenizers/deepseek_v4_encoding.py")
        )
    if not source_path.is_file():
        raise RuntimeError(f"cannot find DeepSeek V4 encoding at {source_path}")
    spec = importlib.util.spec_from_file_location(
        "vllm_ds4_profile_deepseek_v4_encoding", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DeepSeek V4 encoding from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepared_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for source_message in messages:
        message = {
            key: value for key, value in source_message.items() if key != "extra"
        }
        if "reasoning_content" in message:
            message["reasoning"] = message.pop("reasoning_content")
        prepared.append(message)
    return prepared


def _request_messages(messages: list[dict[str, Any]], end: int) -> list[dict[str, Any]]:
    return [{"role": "system", "tools": [BASH_TOOL]}, *messages[:end]]


def _completion_segments(
    message: dict[str, Any], thinking_mode: str, encoding: ModuleType
) -> list[tuple[str, str]]:
    reasoning = ""
    if thinking_mode == "thinking":
        reasoning = (message.get("reasoning") or "") + encoding.thinking_end_token

    tool_calls = ""
    if message.get("tool_calls"):
        calls = encoding.tool_calls_from_openai_format(message["tool_calls"])
        rendered_calls = [
            encoding.tool_call_template.format(
                dsml_token=encoding.dsml_token,
                name=call["name"],
                arguments=encoding.encode_arguments_to_dsml(call),
            )
            for call in calls
        ]
        tool_calls = "\n\n" + encoding.tool_calls_template.format(
            dsml_token=encoding.dsml_token,
            tool_calls="\n".join(rendered_calls),
            tc_block_name=encoding.tool_calls_block_name,
        )

    framing = "" if message.get("wo_eos", False) else encoding.eos_token
    return [
        ("reasoning_tokens", reasoning),
        ("assistant_content_tokens", message.get("content") or ""),
        ("tool_call_tokens", tool_calls),
        ("completion_framing_tokens", framing),
    ]


def _segment_token_counts(
    tokenizer: Any, segments: list[tuple[str, str]]
) -> tuple[str, dict[str, int]]:
    text = "".join(segment for _, segment in segments)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    bounds = []
    position = 0
    for label, segment in segments:
        bounds.append((label, position, position + len(segment)))
        position += len(segment)

    counts = {label: 0 for label, _ in segments}
    for start, _ in encoded["offset_mapping"]:
        label = next(
            (
                candidate
                for candidate, span_start, span_end in bounds
                if span_start <= start < span_end
            ),
            segments[-1][0],
        )
        counts[label] += 1
    return text, counts


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _source_attribution(
    reusable_tokens: int,
    block_size: int,
    global_prefix_tokens: int,
    task_prefix_tokens: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    boundaries = {
        "global": (0, global_prefix_tokens),
        "task": (global_prefix_tokens, task_prefix_tokens),
        "session": (task_prefix_tokens, reusable_tokens),
    }
    attributed_blocks: dict[str, list[int]] = {
        "global": [],
        "task": [],
        "session": [],
    }
    for block_start in range(0, reusable_tokens, block_size):
        if block_start < global_prefix_tokens:
            source = "global"
        elif block_start < task_prefix_tokens:
            source = "task"
        else:
            source = "session"
        attributed_blocks[source].append(block_start)

    counts = {
        source: len(blocks) * block_size for source, blocks in attributed_blocks.items()
    }
    spans = []
    for source in ("global", "task", "session"):
        blocks = attributed_blocks[source]
        if not blocks:
            continue
        source_start, source_end = boundaries[source]
        spans.append(
            {
                "source": source,
                "token_start": min(source_start, reusable_tokens),
                "token_end": min(source_end, reusable_tokens),
                "attributed_block_start": blocks[0],
                "attributed_block_end": blocks[-1] + block_size,
            }
        )
    return counts, spans


def render_turns(
    manifest_path: Path,
    normalized_turns_path: Path,
    tokenizer_path: Path,
    block_size: int = 16,
    include_token_ids: bool = False,
) -> list[dict[str, Any]]:
    """Render pinned DS4 requests and compute source-tokenizer prompt lengths.

    Args:
        manifest_path: Ticket 01 manifest beside the immutable raw snapshot.
        normalized_turns_path: Ticket 01 normalized turn Parquet.
        tokenizer_path: Local pinned DeepSeek V4 tokenizer directory.
        block_size: Token count in one prefix-cache block.
        include_token_ids: Keep logical token IDs for artifact construction.

    Returns:
        Turn dictionaries in deterministic source-path and turn order.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized_rows = pq.read_table(normalized_turns_path).to_pylist()
    normalized_by_key = {
        (row["source_path"], row["turn_index"]): row for row in normalized_rows
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    encoding = _load_deepseek_v4_encoding()
    rendered_turns = []

    for file_entry in sorted(manifest["files"], key=lambda entry: entry["path"]):
        source_path = manifest_path.parent / file_entry["path"]
        if _sha256(source_path) != file_entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {file_entry['path']}")
        trajectory = json.loads(source_path.read_text(encoding="utf-8"))
        source_messages = trajectory["messages"]
        messages = _prepared_messages(source_messages)
        assistant_indexes = [
            index
            for index, message in enumerate(source_messages)
            if message.get("role") == "assistant"
        ]
        reasoning_mode = source_path.parent.name
        thinking_mode = "thinking" if reasoning_mode == "think_high" else "chat"
        reasoning_effort = "high" if thinking_mode == "thinking" else None

        for turn_index, message_index in enumerate(assistant_indexes):
            normalized = normalized_by_key[(file_entry["path"], turn_index)]
            request = _request_messages(messages, message_index)
            prompt = encoding.encode_messages(
                request,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            segments = _completion_segments(
                messages[message_index], thinking_mode, encoding
            )
            completion, segment_counts = _segment_token_counts(tokenizer, segments)
            rendered_completion = encoding.encode_messages(
                [messages[message_index]],
                context=request,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
                add_default_bos_token=False,
            )
            if completion != rendered_completion:
                raise ValueError(
                    f"completion segmentation mismatch for {file_entry['path']} "
                    f"turn {turn_index}"
                )

            next_message_index = (
                assistant_indexes[turn_index + 1]
                if turn_index + 1 < len(assistant_indexes)
                else len(messages)
            )
            tool_results = [
                message
                for message in messages[message_index + 1 : next_message_index]
                if message.get("role") == "tool"
            ]
            rendered_tool_results = ""
            if tool_results:
                rendered_tool_results = encoding.encode_messages(
                    tool_results,
                    context=[*request, messages[message_index]],
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                    add_default_bos_token=False,
                )
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_token_ids = tokenizer.encode(
                completion, add_special_tokens=False
            )
            tool_result_token_ids = tokenizer.encode(
                rendered_tool_results, add_special_tokens=False
            )
            cumulative_token_ids = tokenizer.encode(
                prompt + completion + rendered_tool_results,
                add_special_tokens=False,
            )
            rendered_turns.append(
                {
                    "assistant_content_tokens": segment_counts[
                        "assistant_content_tokens"
                    ],
                    "completion_framing_tokens": segment_counts[
                        "completion_framing_tokens"
                    ],
                    "completion_tokens": len(completion_token_ids),
                    "cumulative_context_tokens": len(cumulative_token_ids),
                    "prompt_tokens": len(prompt_token_ids),
                    "_prompt_token_ids": prompt_token_ids,
                    "_completion_token_ids": completion_token_ids,
                    "reasoning_tokens": segment_counts["reasoning_tokens"],
                    "reasoning_mode": reasoning_mode,
                    "source_completion_tokens": normalized["completion_tokens"],
                    "source_prompt_tokens": normalized["prompt_tokens"],
                    "source_path": file_entry["path"],
                    "tokens_until_tool_ready": sum(segment_counts.values())
                    - segment_counts["completion_framing_tokens"],
                    "tool_call_tokens": segment_counts["tool_call_tokens"],
                    "tool_result_tokens": len(tool_result_token_ids),
                    "task_id": normalized["task_id"],
                    "trajectory_id": normalized["trajectory_id"],
                    "turn_index": turn_index,
                }
            )

    first_turns = [turn for turn in rendered_turns if turn["turn_index"] == 0]
    global_prefix = first_turns[0]["_prompt_token_ids"]
    for turn in first_turns[1:]:
        common = _longest_common_prefix(global_prefix, turn["_prompt_token_ids"])
        global_prefix = global_prefix[:common]

    first_turns_by_task: dict[str, list[dict[str, Any]]] = {}
    for turn in first_turns:
        first_turns_by_task.setdefault(turn["task_id"], []).append(turn)
    task_prefix_lengths = {}
    for task_id, task_turns in first_turns_by_task.items():
        task_prefix = task_turns[0]["_prompt_token_ids"]
        for turn in task_turns[1:]:
            common = _longest_common_prefix(task_prefix, turn["_prompt_token_ids"])
            task_prefix = task_prefix[:common]
        task_prefix_lengths[task_id] = len(task_prefix)

    previous_by_trajectory: dict[str, list[int]] = {}
    for turn in rendered_turns:
        prompt_token_ids = turn["_prompt_token_ids"]
        previous_token_ids = previous_by_trajectory.get(turn["trajectory_id"])
        exact_lcp_tokens = (
            _longest_common_prefix(previous_token_ids, prompt_token_ids)
            if previous_token_ids is not None
            else 0
        )
        reusable_prefix_tokens = exact_lcp_tokens // block_size * block_size
        task_prefix_tokens = task_prefix_lengths[turn["task_id"]]
        source_counts, source_spans = _source_attribution(
            reusable_prefix_tokens,
            block_size,
            len(global_prefix),
            task_prefix_tokens,
        )
        turn.update(
            {
                "exact_lcp_tokens": exact_lcp_tokens,
                "global_prefix_tokens": len(global_prefix),
                "global_reusable_tokens": source_counts["global"],
                "ideal_prefix_hit": reusable_prefix_tokens > 0,
                "new_prefill_tokens": (turn["prompt_tokens"] - reusable_prefix_tokens),
                "prefix_reuse_ratio": (reusable_prefix_tokens / turn["prompt_tokens"]),
                "prefix_source_spans": source_spans,
                "reusable_prefix_tokens": reusable_prefix_tokens,
                "reuse_distance_turns": (1 if previous_token_ids is not None else None),
                "session_reusable_tokens": source_counts["session"],
                "task_prefix_tokens": task_prefix_tokens,
                "task_reusable_tokens": source_counts["task"],
            }
        )
        previous_by_trajectory[turn["trajectory_id"]] = prompt_token_ids

    if not include_token_ids:
        for turn in rendered_turns:
            turn.pop("_prompt_token_ids")
            turn.pop("_completion_token_ids")

    return rendered_turns


if __name__ == "__main__":
    main()
