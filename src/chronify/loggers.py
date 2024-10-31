"""Contains logging functionality."""

import sys
from pathlib import Path
from typing import Iterable, Optional, Union

from loguru import logger


# Logger printing formats
DEFAULT_FORMAT = "<level>{level}</level>: {message}"
DEBUG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name}:{line}</cyan> | "
    "{message}"
)


def setup_logging(
    filename: Optional[Union[str, Path]] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    mode: str = "w",
    rotation: Optional[str] = "10 MB",
    packages: Optional[Iterable[str]] = None,
) -> None:
    """Configures logging to file and console.

    Parameters
    ----------
    filename
        Log filename, defaults to None for no file logging.
    console_level
        Console logging level
    file_level
        File logging level
    mode
        Mode in which to open the file
    rotation
        Size in which to rotate file. Set to None for no rotation.
    packages
        Additional packages to enable logging
    """
    logger.remove()
    logger.enable("chronify")
    for pkg in packages or []:
        logger.enable(pkg)

    logger.add(sys.stderr, level=console_level, format=DEFAULT_FORMAT)
    if filename:
        logger.add(
            filename,
            level=file_level,
            mode=mode,
            rotation=rotation,
            format=DEBUG_FORMAT,
        )
