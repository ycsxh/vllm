# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks.ds4_profile.prepare_dataset import main, prepare_dataset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pinned"
TOKENIZER_REVISION = "1" * 40


class FakeTokenizer:
    name_or_path = "Qwen/Qwen3.5-4B"
    init_kwargs = {"_commit_hash": TOKENIZER_REVISION}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        contents = "|".join(message["content"] for message in messages)
        return f"<prompt>{contents}<assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(100, 100 + len(text.split("|"))))


def fake_tokenizer_loader(model: str, *, revision: str) -> FakeTokenizer:
    assert model == "Qwen/Qwen3.5-4B"
    assert revision == TOKENIZER_REVISION
    return FakeTokenizer()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _copy_snapshot(tmp_path: Path) -> Path:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    return snapshot_dir / "manifest.json"


def _write_source_and_hash(
    manifest_path: Path, manifest: dict[str, Any], source: dict[str, Any]
) -> None:
    source_path = manifest_path.parent / manifest["files"][0]["path"]
    source_path.write_text(json.dumps(source))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_prepare_dataset_emits_deterministic_prompt_only_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    kwargs = {
        "manifest_path": FIXTURE_DIR / "manifest.json",
        "output_dir": output_dir,
        "model": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_loader": fake_tokenizer_loader,
    }

    prepare_dataset(**kwargs)
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    prepare_dataset(**kwargs)

    assert {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    } == first_bytes
    assert _json_lines(output_dir / "dataset.jsonl") == [
        {
            "prompt": (
                "<prompt>You are a coding agent with bash tools.|"
                "Fix astropy issue 12907.<assistant>"
            )
        },
        {
            "prompt": (
                "<prompt>You are a coding agent with bash tools.|"
                "Fix astropy issue 12907.||def separability_matrix(...): ...|"
                "astropy/modeling/separable.py:42:def separability_matrix"
                "<assistant>"
            )
        },
        {
            "prompt": (
                "<prompt>You are a coding agent with bash tools.|"
                "Fix astropy issue 12907.<assistant>"
            )
        },
        {
            "prompt": (
                "<prompt>You are a coding agent with bash tools.|"
                "Fix astropy issue 12907.||def separability_matrix(...): ..."
                "<assistant>"
            )
        },
    ]
    rows = _json_lines(output_dir / "rows.jsonl")
    assert [row["source_path"] for row in rows] == [
        "data/no_think/astropy__astropy-12907.traj.json",
        "data/no_think/astropy__astropy-12907.traj.json",
        "data/think_high/astropy__astropy-12907.traj.json",
        "data/think_high/astropy__astropy-12907.traj.json",
    ]
    assert [row["turn_index"] for row in rows] == [0, 1, 0, 1]
    assert [row["input_tokens"] for row in rows] == [2, 5, 2, 4]
    assert all(row["input_tokens"] == len(row["prompt_ids"]) for row in rows)
    assert all("output_tokens" not in row for row in rows)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["dataset"]["revision"] == (
        "4da61f3d06b48b6817a62b99e9c47035c8e59787"
    )
    assert provenance["tokenizer"] == {
        "model": "Qwen/Qwen3.5-4B",
        "revision": TOKENIZER_REVISION,
    }
    assert provenance["selection"] == {
        "order": "manifest_path_then_message_index",
        "selected_assistant_turns": "all",
    }
    assert provenance["row_count"] == 4


def test_prepare_dataset_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    manifest_path = _copy_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(manifest["files"][0])
    manifest_path.write_text(json.dumps(manifest))

    try:
        prepare_dataset(
            manifest_path,
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=fake_tokenizer_loader,
        )
    except ValueError as error:
        assert "duplicate source path" in str(error)
    else:
        raise AssertionError("duplicate source path was accepted")


def test_prepare_dataset_rejects_bad_hash_before_loading_tokenizer(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    def unexpected_loader(model: str, *, revision: str) -> FakeTokenizer:
        raise AssertionError(f"loaded {model} at {revision}")

    try:
        prepare_dataset(
            manifest_path,
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=unexpected_loader,
        )
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("bad source hash was accepted")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("format", "unsupported trajectory format"),
        ("identity", "is not a DS4 trajectory"),
        ("assistant_identity", "assistant response is not from DS4"),
    ],
)
def test_prepare_dataset_rejects_invalid_source_contract(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path = _copy_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    source_path = manifest_path.parent / manifest["files"][0]["path"]
    source = json.loads(source_path.read_text())
    if mutation == "format":
        source["trajectory_format"] = "swe-agent-1.1"
    elif mutation == "identity":
        source["info"]["config"]["model"]["model_name"] = "Qwen/Qwen3.5-4B"
    else:
        source["messages"][2]["extra"]["response"]["model"] = "Qwen/Qwen3.5-4B"
    _write_source_and_hash(manifest_path, manifest, source)

    with pytest.raises(ValueError, match=message):
        prepare_dataset(
            manifest_path,
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=fake_tokenizer_loader,
        )
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "main", "dataset revision must be a full immutable commit"),
        ("repo_id", "some/other-dataset", "does not identify the pinned DS4"),
    ],
)
def test_prepare_dataset_rejects_invalid_manifest_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    manifest_path = _copy_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset"][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        prepare_dataset(
            manifest_path,
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=fake_tokenizer_loader,
        )


def test_prepare_dataset_rejects_unverified_tokenizer_revision(tmp_path: Path) -> None:
    class UnpinnedTokenizer(FakeTokenizer):
        init_kwargs = {"_commit_hash": "2" * 40}

    with pytest.raises(ValueError, match="loaded tokenizer revision"):
        prepare_dataset(
            FIXTURE_DIR / "manifest.json",
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=lambda model, *, revision: UnpinnedTokenizer(),
        )


def test_default_loader_does_not_rewrite_tokenizer_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnpinnedTokenizer(FakeTokenizer):
        init_kwargs = {"_commit_hash": "2" * 40}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, *, revision: str) -> UnpinnedTokenizer:
            return UnpinnedTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )

    with pytest.raises(ValueError, match="loaded tokenizer revision"):
        prepare_dataset(
            FIXTURE_DIR / "manifest.json",
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
        )


@pytest.mark.parametrize(
    ("model", "revision", "message"),
    [
        ("Qwen/Qwen3.5-0.8B", TOKENIZER_REVISION, "model must be"),
        ("Qwen/Qwen3.5-4B", "main", "revision must be a full immutable commit"),
    ],
)
def test_prepare_dataset_requires_fixed_model_and_full_tokenizer_revision(
    tmp_path: Path, model: str, revision: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_dataset(
            FIXTURE_DIR / "manifest.json",
            tmp_path / "prepared",
            model=model,
            tokenizer_revision=revision,
            tokenizer_loader=fake_tokenizer_loader,
        )


def test_prepare_dataset_rejects_invalid_rendering(tmp_path: Path) -> None:
    class EmptyPromptTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> str:
            return ""

    with pytest.raises(ValueError, match="failed to render"):
        prepare_dataset(
            FIXTURE_DIR / "manifest.json",
            tmp_path / "prepared",
            model="Qwen/Qwen3.5-4B",
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_loader=lambda model, *, revision: EmptyPromptTokenizer(),
        )
    assert not (tmp_path / "prepared").exists()


def test_cli_exposes_the_pinned_dataset_adapter(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepared"

    main(
        [
            "--manifest",
            str(FIXTURE_DIR / "manifest.json"),
            "--model",
            "Qwen/Qwen3.5-4B",
            "--tokenizer-revision",
            TOKENIZER_REVISION,
            "--output-dir",
            str(output_dir),
        ],
        tokenizer_loader=fake_tokenizer_loader,
    )

    assert len(_json_lines(output_dir / "dataset.jsonl")) == 4
