"""Helper function to instantiate a class for example"""

from datetime import datetime
import importlib
import json
import pathlib
import re
import shutil
import subprocess

import pathspec


def load_gitignore_patterns(gitignore_file: pathlib.Path) -> pathspec.PathSpec:
    if gitignore_file.exists():
        patterns = gitignore_file.read_text().splitlines()
    else:
        patterns = []
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def save_python_code(
    source_root_dir: pathlib.Path,
    target_root_dir: pathlib.Path,
    gitignore_file: pathlib.Path,
):
    """
    Copy files/dirs from source_root_dir to target_root_dir, preserving
    structure, but skipping anything matched by .gitignore at source_root_dir.
    """
    if not source_root_dir.exists():
        raise FileNotFoundError(f"Source directory {source_root_dir} does not exist.")
    if not source_root_dir.is_dir():
        raise NotADirectoryError(f"{source_root_dir} is not a directory.")

    spec = load_gitignore_patterns(gitignore_file)

    for path in source_root_dir.rglob("*"):
        # Path relative to the root for matching
        rel = path.relative_to(source_root_dir).as_posix()

        # Skip paths that match .gitignore
        if spec.match_file(rel):
            continue

        dest_path = target_root_dir / rel

        if path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_path)


def get_commit_hash() -> str:
    # Gets the full hash of HEAD
    output = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
    )
    return output.decode("utf-8").strip()


def save_commit_hash_to_json(json_path: str) -> None:
    commit_hash = get_commit_hash()
    data = {"commit_hash": commit_hash}

    # Ensure parent directory exists
    path = pathlib.Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def instantiate_dirs(root_dir: str, experiment_name: str) -> tuple[str, str, str, str]:
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"{experiment_name}_date={now}"
    save_dir = pathlib.Path(root_dir) / experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)
    return (
        str(save_dir / "logs"),
        str(save_dir / "ckpt"),
        str(save_dir / "images"),
        str(save_dir / "code"),
    )


def instantiate_from_config(config: dict, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(
    string: str,
    reload=False,
    invalidate_cache=True,
):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def parse_name_path(path: str) -> dict[str, int]:
    basename = pathlib.Path(path).stem
    parts = basename.split("_")

    info = {
        "Workflow": int(re.sub(r"[^\d]", "", parts[0])),
        "Tracklet": int(re.sub(r"[^\d]", "", parts[3])),
        "Frame": int(re.sub(r"[^\d]", "", parts[4])),
    }

    return info


def parse_name_path_tracklet(path: str) -> dict[str, int]:
    basename = pathlib.Path(path).stem
    parts = basename.split("_")

    info = {
        "Workflow": int(re.sub(r"[^\d]", "", parts[0])),
        "Tracklet": int(re.sub(r"[^\d]", "", parts[2])),
        "Shirt": int(re.sub(r"[^\d]", "", parts[1])),
    }

    return info