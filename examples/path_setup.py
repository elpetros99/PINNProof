from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root from a notebook working directory."""
    origin = (start or Path.cwd()).resolve()
    for base in (origin, *origin.parents):
        if (base / "src").is_dir() and (base / "examples").is_dir():
            return base
    raise ModuleNotFoundError(
        "Could not locate the PINNProof repository root relative to the current working directory."
    )


def configure_notebook_paths(
    start: Path | None = None,
    *,
    include_external_lib: bool = False,
    include_ecp: bool = False,
) -> Path:
    """Add repo-relative import roots used by the example notebooks."""
    repo_root = find_repo_root(start)
    candidate_paths = [
        repo_root,
        repo_root / "src",
        repo_root / "examples",
    ]
    if include_external_lib:
        candidate_paths.append(repo_root / "external_lib")
    if include_ecp:
        candidate_paths.append(repo_root / "external_lib" / "ECP")

    for path in candidate_paths:
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return repo_root
