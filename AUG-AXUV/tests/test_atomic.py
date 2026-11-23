import json
from pathlib import Path

import pytest
from cherab.core import Maxwellian, Plasma, Species
from cherab.core.atomic import deuterium, neon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS
from raysect.core import Vector3D

from axuv.atomic import available_transitions, line_models
# the real function, before the no_downloads fixture replaces it
from axuv.atomic import _install_variant_file as real_install_variant_file
from tests.test_adf15 import PEC40


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
    """Tests never reach Open-ADAS; populate() is recorded, install_adf15 and
    _install_variant_file fail."""
    import axuv.atomic as atomic
    calls = []
    monkeypatch.setattr(atomic, "populate", lambda **kw: calls.append(kw))

    def no_network(*a, **kw):
        raise OSError("network unreachable")

    monkeypatch.setattr(atomic, "install_adf15", no_network)
    monkeypatch.setattr(atomic, "_install_variant_file", no_network)
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


def test_line_models_needs_next_charge_for_recombination(repo):
    models = line_models(_plasma(repo, [(deuterium, 0), (neon, 8), (neon, 9)]))

    kinds = [type(m) for m in models]
    # D0: no D1 in the plasma, so excitation only; Ne8: excitation only;
    # Ne9: no Ne10, so excitation only
    assert kinds.count(RecombinationLine) == 0
    assert kinds.count(ExcitationLine) == 4
    assert {(m.line.element.symbol, m.line.charge) for m in models} == {
        ("D", 0), ("Ne", 8), ("Ne", 9)
    }

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
    from cherab.core.atomic import molybdenum
    monkeypatch.setattr(atomic, "install_adf15", lambda *a, **kw: pytest.fail("no Mo row"))
    monkeypatch.setattr(atomic, "_install_variant_file", lambda *a, **kw: pytest.fail("no Mo row"))
    with pytest.raises(RuntimeError, match="Mo5\\+"):
        line_models(_plasma(repo, [(molybdenum, 5)]))


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
    monkeypatch.setattr(atomic, "_install_variant_file",
                        lambda *a, **kw: calls.append(("variant", a, kw)))
    atomic.install_atomic_data(repository_path=str(tmp_path), adas_path="/adas")
    expected_kw = {"download": True, "repository_path": str(tmp_path), "adas_path": "/adas"}
    # populate() runs last, so its NIST H0 wavelengths win over the ADF15 file's
    assert calls[-1] == ("populate", expected_kw)
    adf15 = [c[1] for c in calls if c[0] == "adf15"]
    assert adf15 == [(hydrogen, 0, "adf15/pec12#h/pec12#h_pju#h0.dat")] + [
        (neon, q, f"adf15/pec96#ne/pec96#ne_pju#ne{q}.dat") for q in range(10)
    ]
    assert all(c[2] == expected_kw for c in calls if c[0] == "adf15")
    variant_rows = [c[1] for c in calls if c[0] == "variant"]
    assert variant_rows == [row for row in atomic.VARIANT_FILES]
    variant_kw = {"repository_path": str(tmp_path), "adas_path": "/adas"}
    assert all(c[2] == variant_kw for c in calls if c[0] == "variant")


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


@pytest.fixture
def w_repo(repo):
    """repo plus W5 transitions in two variants, ic and ca."""
    _write(repo, "excitation", "w", 5, ["ic:2 -> ic:1", "ic:3 -> ic:1", "ca:2 -> ca:1"])
    return repo


def test_line_models_uses_chosen_variant(w_repo):
    from cherab.core.atomic import tungsten
    models = line_models(_plasma(w_repo, [(tungsten, 5)]), variants={"W": "ic"})
    assert [m.line.transition for m in models] == [("ic:2", "ic:1"), ("ic:3", "ic:1")]
    models = line_models(_plasma(w_repo, [(tungsten, 5)]), variants={"W": "ca"})
    assert [m.line.transition for m in models] == [("ca:2", "ca:1")]


def test_available_transitions_filters_variant(w_repo):
    from cherab.core.atomic import tungsten
    assert available_transitions(tungsten, 5, "excitation", w_repo, "ca") == [("ca:2", "ca:1")]
    assert len(available_transitions(tungsten, 5, "excitation", w_repo)) == 3


def test_line_models_needs_variant_choice(w_repo):
    from cherab.core.atomic import tungsten
    with pytest.raises(ValueError, match="Choose a PEC variant for W \\(ca, cl, ic, ls\\)"):
        line_models(_plasma(w_repo, [(tungsten, 5)]))


@pytest.mark.parametrize("variants, message", [
    ({"W": "xx"}, "Unknown PEC variant 'xx' for W"),
    ({"Ne": "ic"}, "No PEC variants for Ne"),
])
def test_line_models_rejects_bad_variant(repo, variants, message):
    with pytest.raises(ValueError, match=message):
        line_models(_plasma(repo, [(neon, 9)]), variants=variants)


def test_variant_for_element_not_in_plasma_is_ignored(repo):
    assert len(line_models(_plasma(repo, [(neon, 9)]), variants={"W": "ic"})) == 1


def test_line_models_installs_missing_variant(w_repo, monkeypatch):
    """W5 has ca and ic data, W6 none: only the chosen variant's W6 row is installed."""
    import axuv.atomic as atomic
    from cherab.core.atomic import tungsten
    installed = []

    def fake_install(element, charge, variant, file_path, **kw):
        installed.append((element.symbol, charge, variant, file_path))
        _write(w_repo, "excitation", "w", charge, [f"{variant}:5 -> {variant}:1"])

    monkeypatch.setattr(atomic, "_install_variant_file", fake_install)
    models = line_models(_plasma(w_repo, [(tungsten, 5), (tungsten, 6)]), variants={"W": "ic"})
    assert installed == [("W", 6, "ic", "adf15/pec40#w/pec40#w_ic#w6.dat")]
    assert len(models) == 3


def test_line_models_installs_variant_missing_next_to_another(w_repo, monkeypatch):
    """W5 has ic and ca data but not ls: the ls row is installed."""
    import axuv.atomic as atomic
    from cherab.core.atomic import tungsten
    installed = []
    monkeypatch.setattr(atomic, "_install_variant_file",
                        lambda element, charge, variant, file_path, **kw: installed.append(variant))
    with pytest.raises(RuntimeError, match="W5\\+ \\(ls\\): no excitation PEC data"):
        line_models(_plasma(w_repo, [(tungsten, 5)]), variants={"W": "ls"})
    assert installed == ["ls"]


def test_boron_variants_differ_only_for_b3():
    import axuv.atomic as atomic
    rows = {(row[2], row[1]): row[3] for row in atomic.VARIANT_FILES if row[0].symbol == "B"}
    for charge in range(5):
        same = rows[("llu", charge)] == rows[("llr", charge)]
        assert same == (charge != 3), charge
    assert rows[("llr", 3)] == "adf15/pec93#b/pec93#b_llr#b3.dat"
    assert rows[("llu", 3)] == "adf15/pec93#b/pec93#b_llu#b3.dat"


def test_variant_files_cover_all_tungsten_charges():
    import axuv.atomic as atomic
    rows = {(row[2], row[1]) for row in atomic.VARIANT_FILES if row[0].symbol == "W"}
    assert rows == {(c, q) for c in ("ca", "cl", "ic", "ls") for q in range(74)}


def test_install_variant_file_into_repository(tmp_path):
    """A pec40 file installed from adas_path is found by Cherab's OpenADAS lookups."""
    from cherab.core.atomic import tungsten
    adas = tmp_path / "adas"
    source = adas / "adf15" / "pec40#w" / "pec40#w_ic#w7.dat"
    source.parent.mkdir(parents=True)
    source.write_text(PEC40)
    repo = tmp_path / "repo"

    real_install_variant_file(tungsten, 7, "ic", "adf15/pec40#w/pec40#w_ic#w7.dat",
                              repository_path=str(repo), adas_path=str(adas))

    assert available_transitions(tungsten, 7, "excitation", repo, "ic") == [
        ("ic:487", "ic:1"), ("ic:487", "ic:2")
    ]
    adas_data = OpenADAS(data_path=str(repo))
    transition = ("ic:487", "ic:1")
    assert adas_data.wavelength(tungsten, 7, transition) == pytest.approx(12.031)
    pec = adas_data.impact_excitation_pec(tungsten, 7, transition)
    # file value 2e-12 cm^3/s at 1e10 cm^-3 and 10 eV; Cherab returns W m^3
    photon_energy = 6.62607015e-34 * 299792458 / 12.031e-9
    assert pec(1e16, 10) == pytest.approx(2e-18 * photon_energy, rel=1e-6, abs=0)

    models = line_models(_plasma(repo, [(tungsten, 7)]), variants={"W": "ic"})
    assert len(models) == 2


def test_pec_variant_argument():
    import argparse
    from axuv.atomic import add_variant_argument
    parser = argparse.ArgumentParser()
    add_variant_argument(parser)
    assert parser.parse_args([]).pec_variant is None
    args = parser.parse_args(["--pec-variant", "W=ic", "--pec-variant", "B=llr"])
    assert args.pec_variant == {"W": "ic", "B": "llr"}
    for bad in (["--pec-variant", "W"], ["--pec-variant", "W=xx"], ["--pec-variant", "Ne=ic"],
                ["--pec-variant", "W=ic", "--pec-variant", "W=ls"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_bare_boron_needs_no_variant(repo):
    from cherab.core.atomic import boron
    assert line_models(_plasma(repo, [(boron, 5)])) == []


def test_boron_defaults_to_llu(repo):
    from cherab.core.atomic import boron
    for kind in ("excitation", "recombination"):
        _write(repo, kind, "b", 3, ["llu:2 -> llu:1", "llr:3 -> llr:1"])
    models = line_models(_plasma(repo, [(boron, 3)]))
    assert [m.line.transition for m in models] == [("llu:2", "llu:1")]
    models = line_models(_plasma(repo, [(boron, 3)]), variants={"B": "llr"})
    assert [m.line.transition for m in models] == [("llr:3", "llr:1")]


def test_missing_variant_ion_skips_populate(w_repo, monkeypatch, no_downloads):
    """Only a W ion is missing: populate() holds no W data and is not run."""
    import axuv.atomic as atomic
    from cherab.core.atomic import tungsten
    monkeypatch.setattr(
        atomic, "_install_variant_file",
        lambda element, charge, variant, file_path, **kw:
            _write(w_repo, "excitation", "w", charge, [f"{variant}:5 -> {variant}:1"]),
    )
    line_models(_plasma(w_repo, [(tungsten, 6)]), variants={"W": "ic"})
    assert no_downloads == []
