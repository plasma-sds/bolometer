import json
from pathlib import Path

import pytest
from cherab.core import Maxwellian, Plasma, Species
from cherab.core.atomic import deuterium, neon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS
from raysect.core import Vector3D

from axuv.atomic import available_transitions, line_models


def _write(repo, kind, symbol, charge, keys, wavelength_symbols=None):
    """Write PEC keys, plus wavelengths for them under each of wavelength_symbols."""
    path = repo / "pec" / kind / symbol / f"{charge}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: {} for key in keys}))
    for wavelength_symbol in wavelength_symbols or [symbol]:
        path = repo / "wavelength" / wavelength_symbol / f"{charge}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: 100.0 for key in keys}))


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    """Tests never reach Open-ADAS; populate() is recorded, install_adf15 fails."""
    import axuv.atomic as atomic
    calls = []
    monkeypatch.setattr(atomic, "populate", lambda **kw: calls.append(kw))

    def no_network(*a, **kw):
        raise OSError("network unreachable")

    monkeypatch.setattr(atomic, "install_adf15", no_network)
    return calls


@pytest.fixture
def repo(tmp_path):
    """Fake repository: H0 (with D wavelengths) and Ne9 with excitation and
    recombination PECs, Ne8 with excitation only."""
    for kind in ("excitation", "recombination"):
        _write(tmp_path, kind, "h", 0, ["2 -> 1", "3 -> 2"], ["h", "d"])
        _write(tmp_path, kind, "ne", 9, ["2 -> 1"])
    _write(tmp_path, "excitation", "ne", 8, ["2s2 1s0.0 -> 2s1 2p1 1p1.0"])
    return tmp_path


def _plasma(repo, species):
    plasma = Plasma()
    plasma.atomic_data = OpenADAS(data_path=str(repo))
    distribution = Maxwellian(1e19, 10, Vector3D(0, 0, 0), 1e-26)
    plasma.composition = [Species(element, charge, distribution) for element, charge in species]
    return plasma


def test_available_transitions(repo):
    assert available_transitions(neon, 8, "excitation", repo) == [("2s2 1s0.0", "2s1 2p1 1p1.0")]
    assert available_transitions(neon, 9, "recombination", repo) == [(2, 1)]
    assert available_transitions(neon, 3, "excitation", repo) == []


def test_available_transitions_isotope_uses_element(repo):
    assert available_transitions(deuterium, 0, "excitation", repo) == [(2, 1), (3, 2)]


def test_line_models_needs_next_charge_for_recombination(repo, monkeypatch):
    # Cherab 1.5 line models expose no .line attribute; record the Line objects instead
    lines = []
    import axuv.atomic as atomic
    original = atomic.Line
    monkeypatch.setattr(atomic, "Line", lambda *a: lines.append(a) or original(*a))

    models = line_models(_plasma(repo, [(deuterium, 0), (neon, 8), (neon, 9)]))

    kinds = [type(m) for m in models]
    # D0: no D1 in the plasma, so excitation only; Ne8: excitation only;
    # Ne9: no Ne10, so excitation only
    assert kinds.count(RecombinationLine) == 0
    assert kinds.count(ExcitationLine) == 4
    assert {(e.symbol, c) for e, c, _ in lines} == {("D", 0), ("Ne", 8), ("Ne", 9)}

    models = line_models(_plasma(repo, [(deuterium, 0), (deuterium, 1), (neon, 9), (neon, 10)]))
    kinds = [type(m) for m in models]
    assert kinds.count(ExcitationLine) == 3
    assert kinds.count(RecombinationLine) == 3


def test_line_models_fails_for_ion_without_data(repo):
    with pytest.raises(RuntimeError, match="(?s)Ne3\\+: no excitation PEC data.*Ne4\\+"):
        line_models(_plasma(repo, [(neon, 3), (neon, 4)]), install_missing=False)


def test_line_models_installs_missing_ion(repo, monkeypatch, no_downloads):
    import axuv.atomic as atomic
    installed = []

    def fake_install(element, charge, file_path, **kw):
        installed.append((element.symbol, charge, file_path, kw["repository_path"]))
        _write(repo, "excitation", element.symbol.lower(), charge, ["3 -> 2"])

    monkeypatch.setattr(atomic, "install_adf15", fake_install)
    models = line_models(_plasma(repo, [(neon, 3)]))
    assert installed == [("Ne", 3, "adf15/pec96#ne/pec96#ne_pju#ne3.dat", str(repo))]
    assert no_downloads == [{"download": True, "repository_path": str(repo)}]
    assert len(models) == 1


def test_line_models_needs_isotope_wavelengths(repo):
    """H0 PECs alone do not cover D0: its wavelengths are stored under d."""
    (repo / "wavelength" / "d" / "0.json").unlink()
    with pytest.raises(RuntimeError, match="D0\\+: no wavelength for 2 of 2 transitions"):
        line_models(_plasma(repo, [(deuterium, 0)]))


def test_line_models_isotope_wavelength_fallback(repo):
    (repo / "wavelength" / "d" / "0.json").unlink()
    plasma = _plasma(repo, [(deuterium, 0)])
    plasma.atomic_data = OpenADAS(data_path=str(repo), wavelength_element_fallback=True)
    assert len(line_models(plasma)) == 2


def test_line_models_skips_install_when_data_present(repo, no_downloads):
    line_models(_plasma(repo, [(deuterium, 0), (neon, 9), (neon, 10)]))
    assert no_downloads == []


def test_line_models_fails_when_install_fails(repo, monkeypatch):
    import axuv.atomic as atomic

    with pytest.raises(RuntimeError, match="(?s)Ne3\\+.*network unreachable"):
        line_models(_plasma(repo, [(neon, 3)]))


def test_line_models_fails_for_ion_not_in_install_list(repo, monkeypatch):
    import axuv.atomic as atomic
    from cherab.core.atomic import tungsten
    monkeypatch.setattr(atomic, "install_adf15", lambda *a, **kw: pytest.fail("no W row"))
    with pytest.raises(RuntimeError, match="W5\\+"):
        line_models(_plasma(repo, [(tungsten, 5)]))


def test_line_models_silent_for_bare_nucleus(repo, recwarn):
    line_models(_plasma(repo, [(deuterium, 1), (neon, 10)]))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_line_models_warns_for_missing_recombination(repo):
    # Ne8 has excitation data only, and Ne9 is present to recombine from
    with pytest.warns(UserWarning, match="recombination PEC data for Ne8\\+"):
        line_models(_plasma(repo, [(neon, 8), (neon, 9)]))


def test_install_atomic_data_calls(tmp_path, monkeypatch):
    import axuv.atomic as atomic
    from cherab.core.atomic import hydrogen
    calls = []
    monkeypatch.setattr(atomic, "populate", lambda **kw: calls.append(("populate", kw)))
    monkeypatch.setattr(atomic, "install_adf15",
                        lambda *a, **kw: calls.append(("adf15", a, kw)))
    atomic.install_atomic_data(repository_path=str(tmp_path), adas_path="/adas")
    expected_kw = {"download": True, "repository_path": str(tmp_path), "adas_path": "/adas"}
    assert calls[0] == ("populate", expected_kw)
    adf15 = [c[1] for c in calls[1:]]
    assert adf15 == [(hydrogen, 0, "adf15/pec12#h/pec12#h_pju#h0.dat")] + [
        (neon, q, f"adf15/pec96#ne/pec96#ne_pju#ne{q}.dat") for q in range(10)
    ]
    assert all(c[2] == expected_kw for c in calls[1:])


def test_thermal_cx_goes_to_repository_path(monkeypatch):
    import axuv.atomic as atomic
    written = []
    original = lambda rates, repository_path=None: written.append(repository_path)
    monkeypatch.setattr(atomic.cherab_repository, "update_pec_thermal_cx_rates", original)
    with atomic._cherab_install_fixes("/repo"):
        assert atomic.cherab_install.OPENADAS_FILE_URL.startswith("https://")
        # Cherab's install_adf15 calls it without repository_path
        atomic.cherab_repository.update_pec_thermal_cx_rates({})
    assert written == ["/repo"]
    assert atomic.cherab_repository.update_pec_thermal_cx_rates is original
    assert atomic.cherab_install.OPENADAS_FILE_URL.startswith("http://")


def test_main_passes_arguments(monkeypatch):
    import sys
    import axuv.atomic as atomic
    calls = []
    monkeypatch.setattr(atomic, "install_atomic_data", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["atomic", "--repository-path", "/r", "--adas-path", "/a"])
    atomic.main()
    assert calls == [{"repository_path": "/r", "adas_path": "/a"}]


_LOCAL_REPOSITORY = OpenADAS().data_path


@pytest.mark.skipif(not (Path(_LOCAL_REPOSITORY) / "pec/excitation/ne/9.json").is_file(),
                    reason="installed Open-ADAS repository with Ne data required")
def test_installed_lines_resolve_pec_and_wavelength():
    """Every line built for a D + Ne plasma finds its PEC and its wavelength."""
    adas = OpenADAS(permit_extrapolation=True)
    species = [(deuterium, 0), (deuterium, 1)] + [(neon, q) for q in range(11)]
    for element, charge in species:
        for kind, lookup in (("excitation", adas.impact_excitation_pec),
                             ("recombination", adas.recombination_pec)):
            for transition in available_transitions(element, charge, kind):
                assert lookup(element, charge, transition) is not None
                assert adas.wavelength(element, charge, transition) > 0
