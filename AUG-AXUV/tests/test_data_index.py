"""Unit tests for axuv.data_index and axuv.fetch_data — no server, temporary folders."""
import os
import sys

import pytest

from axuv import datafiles, fetch_data
from axuv.data_index import INDEX_FILE, build_index, load_index, select, write_index
from axuv.datafiles import DATA_TREES


def test_tracked_index_keys_are_valid():
    index = load_index()
    assert index
    for key, entry in index.items():
        assert key.split("/", 1)[0] in DATA_TREES, key
        assert set(entry) == {"size", "mtime"}


def test_select():
    index = {"jorek/40673/P1/a.h5": {}, "jorek/40673/P10/b.h5": {}, "atomic/ne.csv": {}}
    assert select(index, ["jorek/40673/P1"]) == ["jorek/40673/P1/a.h5"]
    assert select(index, ["atomic/ne.csv", "jorek/40673/P1/"]) == [
        "atomic/ne.csv", "jorek/40673/P1/a.h5"
    ]
    assert select(index, ["jorek/4067"]) == []


def test_build_write_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PLASMASDS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = tmp_path / "data" / "bolometer" / "public" / "aug-axuv" / "atomic" / "ne.csv"
    path.parent.mkdir(parents=True)
    path.write_text("abc")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    files = build_index(["atomic/ne.csv"])
    assert files == {"atomic/ne.csv": {"size": 3, "mtime": 1_700_000_000}}
    write_index(files, tmp_path / "index.json")
    assert load_index(tmp_path / "index.json") == files


def test_fetch_prints_paths(tmp_path, monkeypatch, capsys):
    calls = []

    class FakeClient:
        def get(self, key, *, private, update):
            calls.append((key, private, update))
            return tmp_path / key

    monkeypatch.setattr(datafiles, "_CLIENT", FakeClient())
    monkeypatch.setattr(sys, "argv", ["fetch_data", "--paths", "atomic", "--update", "if_newer"])
    fetch_data.main()
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / "aug-axuv/atomic/ne.csv")]
    assert calls == [("aug-axuv/atomic/ne.csv", False, "if_newer")]


def test_fetch_unknown_prefix_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_data", "nothing/here"])
    with pytest.raises(SystemExit, match=str(INDEX_FILE.name)):
        fetch_data.main()
