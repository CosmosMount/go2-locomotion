"""Safe preparation of command output directories."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def prepare_output_dir(path: str | Path | None, *, temp_prefix: str) -> Path:
    """Create an empty output directory, replacing an existing one safely."""

    if path is None:
        return Path(tempfile.mkdtemp(prefix=temp_prefix)).resolve()

    output = Path(path).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.absolute()

    if output.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output}")

    resolved = output.resolve(strict=False)
    if not resolved.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"output directory must be inside {OUTPUT_ROOT}: {resolved}")
    if resolved == OUTPUT_ROOT:
        raise ValueError(f"refusing to replace the shared output root: {resolved}")

    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"output path is not a directory: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    return resolved
