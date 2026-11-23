"""Tests against the plasma-sds data server — opt-in, deselected by default.

    pytest -m server         downloads into a temporary data folder (read only)
    pytest -m server_write   uploads, refuses and archives test files in the
                             public tree under test_dataset/server_checks/

Both use the SSH key saved with plasmasds_utility.set_ssh_key. The server never
deletes files, so the uploaded test files stay there; they are small, and can
be removed by hand from public_html/bolometer/aug-axuv/test_dataset/server_checks/
and public_html/bolometer/aug-axuv/archive/.
"""
import time
import uuid

import pytest
from plasmasds_utility import AlreadyOnServerError, TransferError

from axuv import datafiles
from axuv.data_index import load_index
from axuv.datafiles import data_path, local_path, on_server, server_key
from axuv.publish_data import publish

# Small keys from both trees
DOWNLOAD_KEYS = [
    "responsivity/axuv_sensitivity.csv",
    "geometry/AXUV_LOS_geom.txt",
    "traces/40673/ip-exp-40673.txt",
]
# Local copies in the user's data folder, looked up before any test changes it
EXISTING = {key: local_path(key) for key in DOWNLOAD_KEYS}


@pytest.fixture
def fresh_data_dir(tmp_path, monkeypatch):
    """An empty data folder, full server access; the user's SSH key setting is kept."""
    monkeypatch.setenv("PLASMASDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLASMASDS_SERVER_ACCESS", "all")
    monkeypatch.setenv("PLASMASDS_QUIET", "1")
    return tmp_path / "data"


@pytest.mark.server
@pytest.mark.parametrize("key", DOWNLOAD_KEYS)
def test_download_matches_index(key, fresh_data_dir):
    existing = EXISTING[key] if EXISTING[key].is_file() else None
    entry = load_index()[key]

    path = data_path(key)
    assert path.is_relative_to(fresh_data_dir)
    stat = path.stat()
    assert stat.st_size == entry["size"]
    assert int(stat.st_mtime) == entry["mtime"]  # the local copy keeps the server time
    if existing is not None:
        assert path.read_bytes() == existing.read_bytes()


@pytest.mark.server
def test_if_newer_keeps_current_copy(fresh_data_dir):
    path = data_path("responsivity/axuv_sensitivity.csv")
    inode = path.stat().st_ino
    assert data_path("responsivity/axuv_sensitivity.csv", update="if_newer").stat().st_ino == inode


@pytest.mark.server
@pytest.mark.parametrize("key", ["responsivity/axuv_sensitivity.csv", "geometry/gc_d_lines.obj"])
def test_on_server(key):
    assert on_server(key)
    assert not on_server(key.rsplit("/", 1)[0] + "/not-on-the-server.txt")


@pytest.mark.server_write
def test_upload_refuse_and_archive(fresh_data_dir):
    run = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    key = f"test_dataset/server_checks/{run}/check.txt"
    path = local_path(key)
    path.parent.mkdir(parents=True)

    # First upload, then a second one without archiving is refused
    path.write_text("version 1\n")
    assert not on_server(key)
    assert publish([key])["uploaded"] == [key]
    assert on_server(key)
    assert publish([key])["on server"] == [key]

    # Replacing keeps the old file as archive/<version>/<key>
    path.write_text("version 2\n")
    assert publish([key], replace=run)["uploaded"] == [key]

    # Both versions download into a fresh folder, over HTTPS (public tree)
    path.unlink()
    assert data_path(key).read_text() == "version 2\n"
    archived = datafiles._CLIENT.get(server_key(f"archive/{run}/{key}"), private=False)
    assert archived.read_text() == "version 1\n"

    # Archiving under a name that is taken is refused, and the file is kept
    with pytest.raises(TransferError) as error:
        datafiles.upload(key, archive_to=f"archive/{run}/{key}")
    assert not isinstance(error.value, AlreadyOnServerError)
    assert data_path(key, update="force").read_text() == "version 2\n"
