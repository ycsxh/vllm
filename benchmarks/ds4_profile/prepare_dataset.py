# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

DATASET_REPO_ID = "Yi30/deepseek-v4-swebench-trajectories"
DATASET_REPO_TYPE = "model"
MODEL_ID = "Qwen/Qwen3.5-4B"
SOURCE_FORMAT = "mini-swe-agent-1.1"
REASONING_MODES = ("no_think", "think_high")


class Tokenizer(Protocol):
    """Tokenizer operations required by the dataset adapter."""

    name_or_path: str
    init_kwargs: dict[str, Any]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


TokenizerLoader = Callable[..., Tokenizer]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _is_full_revision(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("repo_id") != DATASET_REPO_ID
        or dataset.get("repo_type") != DATASET_REPO_TYPE
    ):
        raise ValueError("manifest does not identify the pinned DS4 dataset")
    if not _is_full_revision(dataset.get("revision")):
        raise ValueError("dataset revision must be a full immutable commit")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty list")
    entries: list[dict[str, str]] = []
    source_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("manifest file entries must be objects")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        if (
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("manifest source paths must be relative to the snapshot")
        if path in source_paths:
            raise ValueError(f"duplicate source path in manifest: {path}")
        source_paths.add(path)
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"invalid SHA-256 for {path}")
        entries.append({"path": path, "sha256": sha256})
    return sorted(entries, key=lambda entry: entry["path"])


def _validate_source(trajectory: dict[str, Any], source_path: str) -> None:
    if trajectory.get("trajectory_format") != SOURCE_FORMAT:
        raise ValueError(f"{source_path} has unsupported trajectory format")
    config = trajectory.get("info", {}).get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{source_path} has invalid trajectory configuration")
    agent_type = config.get("agent_type")
    source_model = config.get("model", {}).get("model_name", "")
    if not isinstance(agent_type, str) or not agent_type.startswith("minisweagent."):
        raise ValueError(f"{source_path} is not a mini-swe-agent trajectory")
    if not isinstance(source_model, str) or "deepseek-v4" not in source_model.lower():
        raise ValueError(f"{source_path} is not a DS4 trajectory")

    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_path} messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError(f"{source_path} contains an invalid message")
        if message["role"] != "assistant":
            continue
        response_model = message.get("extra", {}).get("response", {}).get("model")
        if response_model is not None and (
            not isinstance(response_model, str)
            or "deepseek-v4" not in response_model.lower()
        ):
            raise ValueError(f"{source_path} assistant response is not from DS4")


def _load_tokenizer(model: str, *, revision: str) -> Tokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def _validate_tokenizer(
    tokenizer: Tokenizer, model: str, tokenizer_revision: str
) -> None:
    if tokenizer.name_or_path != model:
        raise ValueError("loaded tokenizer does not match the requested model")
    if tokenizer.init_kwargs.get("_commit_hash") != tokenizer_revision:
        raise ValueError(
            "loaded tokenizer revision does not match the requested revision"
        )


def _render_prompt(
    tokenizer: Tokenizer, messages: list[dict[str, Any]], source_identity: str
) -> tuple[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"tokenizer failed to render {source_identity}")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in prompt_ids
        )
    ):
        raise ValueError(f"tokenizer returned invalid token IDs for {source_identity}")
    return prompt, prompt_ids


def _write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        f"{json.dumps(row, separators=(',', ':'), sort_keys=True)}\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")


def prepare_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    model: str,
    tokenizer_revision: str,
    tokenizer_loader: TokenizerLoader = _load_tokenizer,
) -> None:
    """Prepare pinned DS4 assistant-turn prompts for CustomDataset.

    Args:
        manifest_path: Pinned manifest beside the immutable trajectory snapshot.
        output_dir: Directory for prompt-only data and provenance sidecars.
        model: Qwen3.5 model and tokenizer identifier.
        tokenizer_revision: Full immutable tokenizer commit.
        tokenizer_loader: Injectable external tokenizer loading boundary.

    Raises:
        ValueError: If source or tokenizer provenance and rendering are invalid.
    """
    if model != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    if not _is_full_revision(tokenizer_revision):
        raise ValueError("tokenizer revision must be a full immutable commit")

    manifest = _load_json(manifest_path)
    file_entries = _validate_manifest(manifest)
    sources: list[tuple[dict[str, str], dict[str, Any], str, str]] = []
    for file_entry in file_entries:
        relative_path = file_entry["path"]
        source_path = manifest_path.parent / relative_path
        if _sha256(source_path) != file_entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative_path}")
        trajectory = _load_json(source_path)
        _validate_source(trajectory, relative_path)
        mode = source_path.parent.name
        if mode not in REASONING_MODES:
            raise ValueError(f"unsupported reasoning mode: {mode}")
        task = source_path.name.removesuffix(".traj.json")
        sources.append((file_entry, trajectory, mode, task))

    tokenizer = tokenizer_loader(model, revision=tokenizer_revision)
    _validate_tokenizer(tokenizer, model, tokenizer_revision)
    dataset_rows: list[dict[str, str]] = []
    sidecar_rows: list[dict[str, Any]] = []

    for file_entry, trajectory, mode, task in sources:
        relative_path = file_entry["path"]
        assistant_turn = 0
        for message_index, message in enumerate(trajectory["messages"]):
            if message["role"] != "assistant":
                continue
            source_identity = f"{relative_path} assistant turn {assistant_turn}"
            prompt, prompt_ids = _render_prompt(
                tokenizer, trajectory["messages"][:message_index], source_identity
            )
            dataset_rows.append({"prompt": prompt})
            sidecar_rows.append(
                {
                    "input_tokens": len(prompt_ids),
                    "mode": mode,
                    "prompt_ids": prompt_ids,
                    "request_id": f"{relative_path}#assistant-{assistant_turn}",
                    "source_path": relative_path,
                    "source_sha256": file_entry["sha256"],
                    "task": task,
                    "turn_index": assistant_turn,
                }
            )
            assistant_turn += 1
        if assistant_turn == 0:
            raise ValueError(f"{relative_path} contains no assistant turns")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_lines(output_dir / "dataset.jsonl", dataset_rows)
    _write_json_lines(output_dir / "rows.jsonl", sidecar_rows)
    provenance = {
        "dataset": manifest["dataset"],
        "manifest_sha256": _sha256(manifest_path),
        "row_count": len(dataset_rows),
        "selection": {
            "order": "manifest_path_then_message_index",
            "selected_assistant_turns": "all",
        },
        "source_files": file_entries,
        "tokenizer": {"model": model, "revision": tokenizer_revision},
    }
    (output_dir / "provenance.json").write_text(
        f"{json.dumps(provenance, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    tokenizer_loader: TokenizerLoader = _load_tokenizer,
) -> None:
    """Run the minimal DS4 dataset adapter CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare pinned DS4 prompts as a CustomDataset JSONL."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare_dataset(
        args.manifest,
        args.output_dir,
        model=args.model,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer_loader=tokenizer_loader,
    )


if __name__ == "__main__":
    main()
