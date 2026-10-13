"""Unit tests for axuv.migrate_data — key mapping and file placement in temporary folders."""
import os

import pytest

from axuv.migrate_data import data_key, place, plan


@pytest.mark.parametrize(
    "relative, key",
    [
        ("AXUV_LOS_geom.txt", "geometry/AXUV_LOS_geom.txt"),
        ("gc_d_lines.obj", "geometry/gc_d_lines.obj"),
        ("etendue_S16.h5", "geometry/etendue_S16.h5"),
        ("worst_estimation.csv", "responsivity/worst_estimation.csv"),
        ("degraded_errorbars.csv", "responsivity/degraded_errorbars.csv"),
        ("ne.csv", "atomic/ne.csv"),
        ("dream_output.h5", "dream/dream_output.h5"),
        ("raytransfer_S5_refl.h5", "raytransfer/raytransfer_S5_refl.h5"),
        ("raytransfer_S16_highres_norefl.h5", "raytransfer/raytransfer_S16_highres_norefl.h5"),
        ("step02410_out.h5", "jorek/step02410_out.h5"),
        ("40673/P45/input/step01010_out.h5", "jorek/40673/P45/step01010_out.h5"),
        ("41007/wth-exp-41007.txt", "traces/41007/wth-exp-41007.txt"),
    ],
)
def test_data_key(relative, key):
    assert data_key(relative) == key


@pytest.mark.parametrize(
    "relative",
    ["40673/P1/output/emissions_x.h5", "40673/P1/stdout/output_a.txt", "notes.txt"],
)
def test_data_key_unknown(relative):
    assert data_key(relative) is None


def test_plan(tmp_path):
    data, cad = tmp_path / "data", tmp_path / "aug"
    for path in [data / "ne.csv", data / "notes.txt", cad / "vessel" / "v.rsm"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    mapped, unmapped = plan(data, cad)
    assert [key for _, key in mapped] == ["atomic/ne.csv", "cad/vessel/v.rsm"]
    assert unmapped == [data / "notes.txt"]


def test_place_links_then_reports_exists(tmp_path):
    source = tmp_path / "a.h5"
    source.write_text("x")
    os.utime(source, (1_000_000, 1_000_000))
    target = tmp_path / "mirror" / "sub" / "a.h5"
    assert place(source, target) == "linked"
    assert os.path.samefile(source, target)
    assert target.stat().st_mtime == 1_000_000
    assert place(source, target) == "exists"


def test_place_copy_keeps_mtime(tmp_path):
    source = tmp_path / "a.h5"
    source.write_text("x")
    os.utime(source, (1_000_000, 1_000_000))
    target = tmp_path / "b.h5"
    assert place(source, target, copy=True) == "copied"
    assert not os.path.samefile(source, target)
    assert target.stat().st_mtime == 1_000_000


def test_place_never_replaces(tmp_path):
    source, target = tmp_path / "a.h5", tmp_path / "b.h5"
    source.write_text("new")
    target.write_text("old")
    assert place(source, target) == "conflict"
    assert target.read_text() == "old"
