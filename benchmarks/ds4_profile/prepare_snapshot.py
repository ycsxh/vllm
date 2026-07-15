# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
from pathlib import Path

DATASET_REPO_ID = "Yi30/deepseek-v4-swebench-trajectories"
DATASET_REPO_TYPE = "model"
DATASET_REVISION = "4da61f3d06b48b6817a62b99e9c47035c8e59787"
REASONING_MODES = ("no_think", "think_high")
TRAJECTORY_COUNT = 20
UNIQUE_TASK_COUNT = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_snapshot(snapshot_dir: Path, manifest_path: Path) -> None:
    """Create an auditable manifest for the complete DS4 pilot snapshot.

    Args:
        snapshot_dir: Directory containing the immutable ``data/`` tree.
        manifest_path: Path to write the generated JSON manifest.
    """
    paths_by_mode = {
        reasoning_mode: sorted(
            (snapshot_dir / "data" / reasoning_mode).glob("*.traj.json")
        )
        for reasoning_mode in REASONING_MODES
    }
    trajectory_count = sum(len(paths) for paths in paths_by_mode.values())
    if trajectory_count != TRAJECTORY_COUNT:
        raise ValueError(
            f"complete pilot requires {TRAJECTORY_COUNT} trajectories, "
            f"found {trajectory_count}"
        )

    tasks_by_mode = {
        reasoning_mode: {path.name.removesuffix(".traj.json") for path in paths}
        for reasoning_mode, paths in paths_by_mode.items()
    }
    if tasks_by_mode["no_think"] != tasks_by_mode["think_high"]:
        raise ValueError("complete pilot requires paired reasoning modes per task")
    task_ids = sorted(tasks_by_mode["no_think"])
    if len(task_ids) != UNIQUE_TASK_COUNT:
        raise ValueError(
            f"complete pilot requires {UNIQUE_TASK_COUNT} unique tasks, "
            f"found {len(task_ids)}"
        )

    source_paths = sorted(path for paths in paths_by_mode.values() for path in paths)
    manifest = {
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "repo_type": DATASET_REPO_TYPE,
            "revision": DATASET_REVISION,
        },
        "files": [
            {
                "path": path.relative_to(snapshot_dir).as_posix(),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "pilot_coverage": {
            "domains": sorted({task_id.split("__", 1)[0] for task_id in task_ids}),
            "reasoning_modes": list(REASONING_MODES),
            "trajectory_count": TRAJECTORY_COUNT,
            "unique_task_count": UNIQUE_TASK_COUNT,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifest a complete pinned DS4 trajectory snapshot."
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare_snapshot(args.snapshot_dir, args.manifest)


if __name__ == "__main__":
    main()
