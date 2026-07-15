# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pinned"


def test_complete_snapshot_emits_auditable_manifest(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    source_files = {
        "no_think": (FIXTURE_DIR / "data/no_think/astropy__astropy-12907.traj.json"),
        "think_high": (
            FIXTURE_DIR / "data/think_high/astropy__astropy-12907.traj.json"
        ),
    }
    task_ids = [f"astropy__astropy-{index:05d}" for index in range(10)]
    for reasoning_mode, source_path in source_files.items():
        mode_dir = snapshot_dir / "data" / reasoning_mode
        mode_dir.mkdir(parents=True)
        for task_id in task_ids:
            shutil.copyfile(source_path, mode_dir / f"{task_id}.traj.json")

    manifest_path = snapshot_dir / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.prepare_snapshot",
            "--snapshot-dir",
            str(snapshot_dir),
            "--manifest",
            str(manifest_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"] == {
        "repo_id": "Yi30/deepseek-v4-swebench-trajectories",
        "repo_type": "model",
        "revision": "4da61f3d06b48b6817a62b99e9c47035c8e59787",
    }
    assert manifest["pilot_coverage"] == {
        "domains": ["astropy"],
        "reasoning_modes": ["no_think", "think_high"],
        "trajectory_count": 20,
        "unique_task_count": 10,
    }
    assert len(manifest["files"]) == 20
    assert manifest["files"][0] == {
        "path": "data/no_think/astropy__astropy-00000.traj.json",
        "sha256": ("f75519fb12540da413a8b844bb6a975fa9635b12c82a743eb5995d7c4d145432"),
    }
    assert manifest["files"][-1] == {
        "path": "data/think_high/astropy__astropy-00009.traj.json",
        "sha256": ("808fdd40dd5327611fbc86d74c94606387c3d6ed3cd1979d469e4b1ec61e9630"),
    }
