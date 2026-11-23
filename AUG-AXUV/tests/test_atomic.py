import json

import pytest
from cherab.core import Maxwellian, Plasma, Species
from cherab.core.atomic import deuterium, neon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS
from raysect.core import Vector3D

from axuv.atomic import available_transitions, line_models


def _write(repo, kind, symbol, charge, keys):
    path = repo / "pec" / kind / symbol / f"{charge}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: {} for key in keys}))


@pytest.fixture
def repo(tmp_path):
    """Fake repository: H0 and Ne9 with excitation and recombination PECs."""
    for kind in ("excitation", "recombination"):
        _write(tmp_path, kind, "h", 0, ["2 -> 1", "3 -> 2"])
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


def test_line_models_warns_for_ion_without_data(repo):
    with pytest.warns(UserWarning, match="Ne3\\+"):
        models = line_models(_plasma(repo, [(neon, 3), (neon, 4)]))
    assert models == []


def test_line_models_silent_for_bare_nucleus(repo, recwarn):
    line_models(_plasma(repo, [(deuterium, 1), (neon, 10)]))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
