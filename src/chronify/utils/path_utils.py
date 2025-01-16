import shutil

from pathlib import Path

from chronify.exceptions import InvalidOperation


def check_overwrite(path: Path, overwrite: bool) -> None:
    """Check if the path exists, handling the user flag overwrite."""
    if not path.exists():
        return

    if overwrite:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    else:
        msg = f"{path=} already exists. Choose a different path or set overwrite=True."
        raise InvalidOperation(msg)


def delete_if_exists(path: Path) -> None:
    """Delete the path if it exists, handling files and directories."""
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def to_path(path: Path | str) -> Path:
    """Convert the instance to a Path if is not already one.
    This is here because calling Path on an object that already a Path does a bunch of work.
    This is significantly faster in relative scale (but still only ~1 us).
    """
    return path if isinstance(path, Path) else Path(path)
