"""Unit tests for axuv.publish_data — key selection and upload outcomes, no server."""
import pytest
from plasmasds_utility import AuthError, TransferError

from axuv import datafiles, publish_data
from axuv.publish_data import local_keys, publish


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PLASMASDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PLASMASDS_SERVER_ACCESS", "local")
    monkeypatch.setenv("PLASMASDS_QUIET", "1")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "data" / "bolometer"


def _put(root, tree, key):
    path = root / tree / "aug-axuv" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_local_keys_selects_by_prefix(data_dir):
    _put(data_dir, "private", "geometry/a.txt")
    _put(data_dir, "private", "jorek/40673/P1/s1.h5")
    _put(data_dir, "private", "jorek/41007/P1/s2.h5")
    _put(data_dir, "public", "atomic/ne.csv")
    keys, misplaced = local_keys(["jorek/40673", "atomic"])
    assert keys == ["atomic/ne.csv", "jorek/40673/P1/s1.h5"]
    assert misplaced == []


def test_local_keys_prefix_matches_whole_folders(data_dir):
    _put(data_dir, "private", "jorek/40673/P1/s1.h5")
    assert local_keys(["jorek/4067"]) == ([], [])


def test_local_keys_reports_wrong_tree(data_dir):
    _put(data_dir, "public", "geometry/a.txt")
    _put(data_dir, "private", "misc/b.txt")
    keys, misplaced = local_keys(["geometry", "misc"])
    assert keys == []
    assert misplaced == ["private/aug-axuv/misc/b.txt", "public/aug-axuv/geometry/a.txt"]


def test_local_keys_ignores_hidden_and_partial(data_dir):
    _put(data_dir, "private", "geometry/.a.txt.1234.part")
    _put(data_dir, "private", "geometry/.hidden")
    assert local_keys(["geometry"]) == ([], [])


class FakeClient:
    def __init__(self, errors):
        self.errors = errors
        self.calls = []

    def upload(self, key, *, private, archive_to):
        self.calls.append((key, private, archive_to))
        if key in self.errors:
            raise self.errors[key]


def test_publish_outcomes(monkeypatch):
    client = FakeClient({
        "aug-axuv/geometry/b.txt": TransferError(
            "/x is already on the server; an upload replaces a file only when ..."
        ),
        "aug-axuv/geometry/c.txt": TransferError("timed out"),
    })
    monkeypatch.setattr(datafiles, "_CLIENT", client)
    results = publish(["atomic/ne.csv", "geometry/b.txt", "geometry/c.txt"])
    assert results["uploaded"] == ["atomic/ne.csv"]
    assert results["on server"] == ["geometry/b.txt"]
    assert results["failed"] == ["geometry/c.txt: timed out"]
    assert client.calls[0] == ("aug-axuv/atomic/ne.csv", False, None)
    assert client.calls[1][1] is True


def test_publish_replace_archives(monkeypatch):
    client = FakeClient({})
    monkeypatch.setattr(datafiles, "_CLIENT", client)
    publish(["responsivity/a.csv"], replace="v1")
    assert client.calls == [
        ("aug-axuv/responsivity/a.csv", False, "aug-axuv/archive/v1/responsivity/a.csv")
    ]


def test_publish_stops_without_access(monkeypatch):
    client = FakeClient({"aug-axuv/geometry/a.txt": AuthError("no key")})
    monkeypatch.setattr(datafiles, "_CLIENT", client)
    with pytest.raises(AuthError):
        publish(["geometry/a.txt", "geometry/b.txt"])
    assert len(client.calls) == 1


def test_on_server_message_matches_pinned_utility():
    # Guards the message match against a change in the pinned plasmasds_utility.
    import inspect

    from plasmasds_utility import _sftp

    assert publish_data._ON_SERVER in inspect.getsource(_sftp.upload)
