from pathlib import Path

import pytest

from chronify.exceptions import InvalidOperation
from chronify.utils.path_utils import check_overwrite, delete_if_exists, to_path


def test_check_overwrite(tmp_path):
    file_path = tmp_path / "file.txt"
    directory = tmp_path / "test_dir"
    check_overwrite(file_path, False)
    check_overwrite(directory, False)

    directory.mkdir()
    file_path.touch()
    for path in (directory, file_path):
        assert path.exists()
        with pytest.raises(InvalidOperation):
            check_overwrite(path, False)
        assert path.exists()
        check_overwrite(path, True)
        assert not path.exists()


def test_delete_if_exists(tmp_path):
    file_path = tmp_path / "file.txt"
    directory = tmp_path / "test_dir"
    delete_if_exists(file_path)
    delete_if_exists(directory)

    directory.mkdir()
    file_path.touch()
    for path in (directory, file_path):
        assert path.exists()
        delete_if_exists(path)
        assert not path.exists()


@pytest.mark.parametrize("value", ["file.txt", Path("file.txt")])
def test_to_path(value):
    assert isinstance(to_path(value), Path)
