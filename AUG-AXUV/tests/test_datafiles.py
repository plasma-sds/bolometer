"""Unit tests for axuv.datafiles — run plasmasds_utility locally, never contact a server."""
import pytest
from plasmasds_utility import AuthError, TransferError

from axuv import datafiles
from axuv.datafiles import DATA_TREES, data_path, is_private, local_path, server_key


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the utility at an empty data folder and keep it off the network."""
    monkeypatch.setenv("PLASMASDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLASMASDS_SERVER_ACCESS", "local")
    monkeypatch.setenv("PLASMASDS_QUIET", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "data"


def _put(data_dir, tree, key, content=b"x"):
    path = data_dir / "bolometer" / tree / "aug-axuv" / key
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


# ── key handling ──────────────────────────────────────────────────────────────
def test_server_key_adds_folder():
    assert server_key("geometry/AXUV_LOS_geom.txt") == "aug-axuv/geometry/AXUV_LOS_geom.txt"


@pytest.mark.parametrize("group", ["geometry", "cad", "raytransfer", "jorek", "traces"])
def test_private_groups(group):
    assert is_private(f"{group}/file.h5")


@pytest.mark.parametrize("group", ["responsivity", "atomic", "dream", "emissions", "test_dataset"])
def test_public_groups(group):
    assert not is_private(f"{group}/file.h5")


def test_every_tree_is_private_or_public():
    assert set(DATA_TREES.values()) == {"private", "public"}


def test_unknown_group_raises():
    with pytest.raises(KeyError, match="unknown data key"):
        is_private("misc/file.h5")


# ── tree passed to the utility ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "key, private",
    [("geometry/AXUV_LOS_geom.txt", True), ("responsivity/axuv_sensitivity.csv", False)],
)
def test_data_path_passes_tree(monkeypatch, key, private):
    calls = []

    class FakeClient:
        def get(self, server_key, *, private, update):
            calls.append((server_key, private, update))

    monkeypatch.setattr(datafiles, "_CLIENT", FakeClient())
    data_path(key, update="if_newer")
    assert calls == [("aug-axuv/" + key, private, "if_newer")]


@pytest.mark.parametrize(
    "key, private", [("cad/vessel/a.rsm", True), ("atomic/ne.csv", False)]
)
def test_on_server_passes_tree(monkeypatch, key, private):
    calls = []

    class FakeClient:
        def on_server(self, server_key, *, private):
            calls.append((server_key, private))
            return True

    monkeypatch.setattr(datafiles, "_CLIENT", FakeClient())
    assert datafiles.on_server(key) is True
    assert calls == [("aug-axuv/" + key, private)]


# ── local copies through the real utility ─────────────────────────────────────
def test_local_path_layout(data_dir):
    assert local_path("cad/vessel/a.rsm") == data_dir / "bolometer/private/aug-axuv/cad/vessel/a.rsm"
    assert local_path("atomic/ne.csv") == data_dir / "bolometer/public/aug-axuv/atomic/ne.csv"


def test_private_local_copy(data_dir):
    path = _put(data_dir, "private", "geometry/gc_d_lines.obj")
    assert data_path("geometry/gc_d_lines.obj") == path


def test_public_local_copy(data_dir):
    path = _put(data_dir, "public", "atomic/ne.csv")
    assert data_path("atomic/ne.csv") == path


def test_public_copy_never_replaces_private(data_dir):
    _put(data_dir, "public", "geometry/AXUV_LOS_geom.txt")
    with pytest.raises(TransferError):
        data_path("geometry/AXUV_LOS_geom.txt")


def test_missing_file_raises_offline(data_dir):
    with pytest.raises(TransferError):
        data_path("dream/dream_output.h5")


def test_private_key_without_private_access(data_dir, monkeypatch):
    monkeypatch.setenv("PLASMASDS_SERVER_ACCESS", "public")
    with pytest.raises(AuthError):
        data_path("raytransfer/raytransfer_S16_norefl.h5")
