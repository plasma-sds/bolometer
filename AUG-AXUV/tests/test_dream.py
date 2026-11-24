import h5py
import numpy as np
import pytest
from cherab.core.atomic import boron, carbon, deuterium, neon

from axuv.dream import load_case, sample_grid, species_densities

NAMES = ["B", "D_th", "D_inj", "Ne_inj"]
Z = np.array([6, 1, 1, 10])
NR = 3


def _write(path, t, state, filetimeindex=None, ablation_from=None):
    """DREAM-like file whose step k holds the plasma state state(k)."""
    nt = len(t)
    n_i = np.array([state(k)[0] for k in range(nt)])
    T = np.array([state(k)[1] for k in range(nt)])
    Y_p = np.ones((nt, 2, 1))
    if ablation_from is not None:
        Y_p[ablation_from:, 0, 0] = 0.9
    with h5py.File(path, "w") as f:
        f["grid/t"] = t
        f["grid/r"] = np.linspace(0.1, 0.5, NR)
        f["grid/r_f"] = np.linspace(0, 0.6, NR + 1)
        f["ionmeta/names"] = np.frombuffer((";".join(NAMES) + ";").encode(), dtype="S1")
        f["ionmeta/Z"] = Z.astype(np.uint64)
        f["eqsys/n_i"] = n_i
        f["eqsys/T_cold"] = T
        f["eqsys/n_cold"] = np.full((nt, NR), 1e19)
        f["eqsys/n_hot"] = np.zeros((nt, NR))
        f["eqsys/n_re"] = np.zeros((nt, NR))
        f["eqsys/Y_p"] = Y_p
        if filetimeindex is not None:
            f["settings/init/filetimeindex"] = np.array([filetimeindex])


def _state(step):
    """Plasma state at joined step number `step`: densities and temperature tagged with it."""
    n_i = np.full((int((Z + 1).sum()), NR), 1e16 * (step + 1))
    return n_i, np.full(NR, 100.0 + step)


@pytest.fixture
def case_dir(tmp_path):
    """init_0: joined steps 0-4; init_1: steps 4-9 (TQ onset at its index 3,
    joined step 7); main: steps 7-11. Fragments start ablating at step 2."""
    dt = 1e-5
    _write(tmp_path / "init_0.h5", np.arange(5) * dt, lambda k: _state(k), ablation_from=2)
    _write(tmp_path / "init_1.h5", np.arange(6) * dt, lambda k: _state(4 + k))
    _write(tmp_path / "main.h5", np.arange(5) * dt, lambda k: _state(7 + k), filetimeindex=3)
    return tmp_path


def test_load_case_drops_repeated_steps(case_dir):
    case = load_case(case_dir)
    # steps 0-4 from init_0, 5-7 from init_1 (up to its index 3), 8-11 from main
    np.testing.assert_allclose(case.T_cold[:, 0], 100.0 + np.arange(12))
    np.testing.assert_allclose(case.t, np.arange(12) * 1e-5)
    assert case.source[4] == ("init_0.h5", 4)
    assert case.source[5] == ("init_1.h5", 1)
    assert case.source[7] == ("init_1.h5", 3)
    assert case.source[8] == ("main.h5", 1)
    assert case.t_onset == pytest.approx(7e-5)
    assert case.t_arrival == pytest.approx(2e-5)
    assert case.index_at(0.0) == 7


def test_load_case_rejects_mismatched_join(case_dir):
    _write(case_dir / "main.h5", np.arange(5) * 1e-5, lambda k: _state(8 + k), filetimeindex=3)
    with pytest.raises(ValueError, match="init_1.h5\\[3\\] and main.h5\\[0\\] differ"):
        load_case(case_dir)


def test_sample_grid_spans_arrival_to_onset(case_dir):
    case = load_case(case_dir)
    # arrival at t_rel = -5e-5; grid step 1e-5: two steps before, two after onset
    grid = sample_grid([case], step=1e-5, before_arrival=2, after_onset=2)
    np.testing.assert_allclose(grid, np.arange(-7, 3) * 1e-5)


def test_sample_grid_outside_case_raises(case_dir):
    case = load_case(case_dir)
    with pytest.raises(ValueError, match="leaves its time range"):
        sample_grid([case], step=1e-5, before_arrival=2, after_onset=8)


def test_species_densities_sums_deuterium_and_folds_boron(case_dir):
    case = load_case(case_dir)
    densities = species_densities(case, 0)
    n = 1e16
    np.testing.assert_allclose(densities[(deuterium, 0)], 2 * n)  # D_th + D_inj
    np.testing.assert_allclose(densities[(deuterium, 1)], 2 * n)
    np.testing.assert_allclose(densities[(neon, 10)], n)
    # "B" has Z = 6 in the file: charge 6 is folded into B5+
    np.testing.assert_allclose(densities[(boron, 5)], 2 * n)
    assert max(q for element, q in densities if element is boron) == 5
    assert any("charge 6 folded into B5+" in flag for flag in case.flags)


def test_species_densities_as_carbon(case_dir):
    case = load_case(case_dir)
    densities = species_densities(case, 0, elements={"B": carbon})
    assert max(q for element, q in densities if element is carbon) == 6
    assert case.flags == []
