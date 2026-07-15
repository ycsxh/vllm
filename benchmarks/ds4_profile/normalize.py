# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import regex as re

PARSER_VERSION = "1.0.0"
REASONING_MODES = frozenset({"no_think", "think_high"})
SCHEMA_VERSION = "1.0.0"

TOOL_CALL_TYPE = pa.struct(
    [
        pa.field("id", pa.string()),
        pa.field("type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("arguments", pa.string()),
    ]
)
TOOL_RESULT_TYPE = pa.struct(
    [
        pa.field("tool_call_id", pa.string(), nullable=False),
        pa.field("content", pa.string()),
        pa.field("duration_ms", pa.float64()),
    ]
)
TURN_SCHEMA = pa.schema(
    [
        pa.field("assistant_content", pa.string()),
        pa.field("assistant_message_index", pa.int64(), nullable=False),
        pa.field("completion_tokens", pa.int64()),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("message_count", pa.int64(), nullable=False),
        pa.field("parallel_tool_call_count", pa.int64(), nullable=False),
        pa.field("parser_version", pa.string(), nullable=False),
        pa.field("prompt_tokens", pa.int64()),
        pa.field("reasoning_content", pa.string()),
        pa.field("reasoning_mode", pa.string(), nullable=False),
        pa.field("reasoning_tokens", pa.int64()),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_agent_type", pa.string(), nullable=False),
        pa.field("source_model", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("source_trajectory_format", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("tool_call_count", pa.int64(), nullable=False),
        pa.field("tool_calls", pa.list_(TOOL_CALL_TYPE), nullable=False),
        pa.field("tool_duration_ms", pa.float64()),
        pa.field("tool_ready", pa.bool_(), nullable=False),
        pa.field("tool_results", pa.list_(TOOL_RESULT_TYPE), nullable=False),
        pa.field("total_tokens", pa.int64()),
        pa.field("trajectory_id", pa.string(), nullable=False),
        pa.field("turn_count", pa.int64(), nullable=False),
        pa.field("turn_index", pa.int64(), nullable=False),
    ]
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_ms(message: dict[str, Any]) -> float | None:
    extra = message.get("extra", {})
    if "duration_ms" in extra:
        return float(extra["duration_ms"])
    if "duration_seconds" in extra:
        return float(extra["duration_seconds"]) * 1000
    return None


def _normalized_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        calls.append(
            {
                "id": tool_call.get("id"),
                "type": tool_call.get("type"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return calls


def _tool_calls_are_complete(tool_calls: list[dict[str, Any]]) -> bool:
    if not tool_calls:
        return False
    for tool_call in tool_calls:
        if not all(tool_call.values()) or tool_call["type"] != "function":
            return False
        try:
            json.loads(tool_call["arguments"])
        except (TypeError, json.JSONDecodeError):
            return False
    return True


def _schema_metadata(manifest: dict[str, Any]) -> dict[bytes, bytes]:
    pilot_coverage = json.dumps(
        manifest["pilot_coverage"], separators=(",", ":"), sort_keys=True
    )
    return {
        b"dataset_revision": manifest["dataset"]["revision"].encode(),
        b"parser_version": PARSER_VERSION.encode(),
        b"pilot_coverage": pilot_coverage.encode(),
        b"schema_version": SCHEMA_VERSION.encode(),
    }


def _validate_source(trajectory: dict[str, Any], source_path: str) -> None:
    trajectory_format = trajectory.get("trajectory_format", "")
    if not isinstance(trajectory_format, str) or not trajectory_format.startswith(
        "mini-swe-agent-"
    ):
        raise ValueError(f"{source_path} has unsupported trajectory format")

    config = trajectory.get("info", {}).get("config", {})
    agent_type = config.get("agent_type", "")
    if not isinstance(agent_type, str) or "minisweagent." not in agent_type:
        raise ValueError(f"{source_path} is not a mini-swe-agent trajectory")

    source_model = config.get("model", {}).get("model_name", "")
    if not isinstance(source_model, str) or "deepseek-v4" not in source_model.lower():
        raise ValueError(f"{source_path} is not a DS4 trajectory")

    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        response_model = message.get("extra", {}).get("response", {}).get("model")
        if response_model is not None and (
            not isinstance(response_model, str)
            or "deepseek-v4" not in response_model.lower()
        ):
            raise ValueError(f"{source_path} assistant response is not from DS4")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    revision = manifest.get("dataset", {}).get("revision", "")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("dataset revision must be a full immutable commit")


def normalize(
    manifest_path: Path,
    output_dir: Path,
    require_complete_pilot: bool = False,
) -> None:
    """Normalize a pinned DS4 snapshot into turn-level artifacts.

    Args:
        manifest_path: JSON manifest beside the immutable raw snapshot.
        output_dir: Directory for normalized Parquet and provenance output.
    """
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)
    fixture_root = manifest_path.parent
    file_entries = sorted(manifest["files"], key=lambda entry: entry["path"])
    if require_complete_pilot:
        expected_count = manifest["pilot_coverage"]["trajectory_count"]
        if len(file_entries) != expected_count:
            raise ValueError(
                f"complete pilot requires {expected_count} trajectories, "
                f"found {len(file_entries)}"
            )
    rows: list[dict[str, Any]] = []
    trajectory_count = 0

    for file_entry in file_entries:
        source_path = fixture_root / file_entry["path"]
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != file_entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {file_entry['path']}")

        trajectory = _load_json(source_path)
        _validate_source(trajectory, file_entry["path"])
        reasoning_mode = source_path.parent.name
        if reasoning_mode not in REASONING_MODES:
            raise ValueError(f"unsupported reasoning mode: {reasoning_mode}")
        task_id = source_path.name.removesuffix(".traj.json")
        trajectory_id = f"{task_id}:{reasoning_mode}"
        messages = trajectory["messages"]
        assistant_indexes = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        trajectory_count += 1
        config = trajectory["info"]["config"]
        for turn_index, message_index in enumerate(assistant_indexes):
            message = messages[message_index]
            next_message_index = (
                assistant_indexes[turn_index + 1]
                if turn_index + 1 < len(assistant_indexes)
                else len(messages)
            )
            turn_messages = messages[message_index:next_message_index]
            tool_calls = _normalized_tool_calls(message)
            tool_results = []
            tool_durations = []
            for turn_message in turn_messages[1:]:
                if turn_message.get("role") != "tool":
                    continue
                duration_ms = _duration_ms(turn_message)
                tool_results.append(
                    {
                        "tool_call_id": turn_message.get("tool_call_id"),
                        "content": turn_message.get("content"),
                        "duration_ms": duration_ms,
                    }
                )
                if duration_ms is not None:
                    tool_durations.append(duration_ms)
            response = message.get("extra", {}).get("response", {})
            usage = response.get("usage", {})
            completion_details = usage.get("completion_tokens_details") or {}
            rows.append(
                {
                    "assistant_content": message.get("content"),
                    "assistant_message_index": message_index,
                    "completion_tokens": usage.get("completion_tokens"),
                    "dataset_revision": manifest["dataset"]["revision"],
                    "message_count": len(turn_messages),
                    "parallel_tool_call_count": (
                        len(tool_calls) if len(tool_calls) > 1 else 0
                    ),
                    "parser_version": PARSER_VERSION,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "reasoning_content": message.get("reasoning_content"),
                    "reasoning_mode": reasoning_mode,
                    "reasoning_tokens": completion_details.get("reasoning_tokens"),
                    "schema_version": SCHEMA_VERSION,
                    "source_agent_type": config["agent_type"],
                    "source_model": config["model"]["model_name"],
                    "source_path": file_entry["path"],
                    "source_sha256": file_entry["sha256"],
                    "source_trajectory_format": trajectory["trajectory_format"],
                    "task_id": task_id,
                    "tool_call_count": len(tool_calls),
                    "tool_calls": tool_calls,
                    "tool_duration_ms": (
                        sum(tool_durations) if tool_durations else None
                    ),
                    "tool_ready": _tool_calls_are_complete(tool_calls),
                    "tool_results": tool_results,
                    "total_tokens": usage.get("total_tokens"),
                    "trajectory_id": trajectory_id,
                    "turn_index": turn_index,
                    "turn_count": len(assistant_indexes),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = TURN_SCHEMA.with_metadata(_schema_metadata(manifest))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), output_dir / "turns.parquet"
    )
    provenance = {
        "dataset_repo_id": manifest["dataset"]["repo_id"],
        "dataset_repo_type": manifest["dataset"]["repo_type"],
        "dataset_revision": manifest["dataset"]["revision"],
        "file_count": len(file_entries),
        "parser_version": PARSER_VERSION,
        "pilot_coverage": manifest["pilot_coverage"],
        "schema_version": SCHEMA_VERSION,
        "source_files": file_entries,
        "trajectory_count": trajectory_count,
        "turn_count": len(rows),
    }
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize pinned DS4 mini-swe-agent trajectories."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete-pilot", action="store_true")
    args = parser.parse_args()
    normalize(args.manifest, args.output_dir, args.require_complete_pilot)


if __name__ == "__main__":
    main()
