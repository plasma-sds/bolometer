"""Tests for the spectral weighting in axuv.responsivity."""

import numpy as np
import pytest

from axuv.cameras import WAVELENGTH_BIN_EDGES
from axuv.responsivity import (
    degraded_sensitivity_function,
    get_weighted_power,
    sensitivity_function,
)


def _flat_power(value: float = 1.0) -> np.ndarray:
    """One diode with the same spectral power [W/nm] in every wavelength bin."""
    return np.full((1, len(WAVELENGTH_BIN_EDGES) - 1), value)


def test_weighting_uses_photon_energy_in_ev():
    """A single bin is weighted with the responsivity at E = hc / lambda."""
    i = len(WAVELENGTH_BIN_EDGES) // 2
    data = np.zeros((1, len(WAVELENGTH_BIN_EDGES) - 1))
    data[0, i] = 1.0
    lo, hi = sorted((1239.8 / WAVELENGTH_BIN_EDGES[i], 1239.8 / WAVELENGTH_BIN_EDGES[i + 1]))
    expected_weight = np.average(sensitivity_function(np.linspace(lo, hi, 100)))
    width = abs(WAVELENGTH_BIN_EDGES[i + 1] - WAVELENGTH_BIN_EDGES[i])
    result = get_weighted_power(data, degraded=False)
    assert result[0] == pytest.approx(expected_weight * width, rel=1e-6)


def test_degraded_differs_from_nominal():
    """The degraded curve lowers the response above ~90 eV, so a flat spectrum must differ."""
    nominal = get_weighted_power(_flat_power(), degraded=False)
    degraded = get_weighted_power(_flat_power(), degraded=True)
    assert degraded[0] < nominal[0]


def test_effective_responsivity_in_physical_range():
    """Effective responsivity of a flat spectrum lies inside the range of the curve itself."""
    width = np.abs(np.diff(WAVELENGTH_BIN_EDGES)).sum()
    energies = 1239.8 / WAVELENGTH_BIN_EDGES
    for degraded, curve in ((False, sensitivity_function), (True, degraded_sensitivity_function)):
        eta = get_weighted_power(_flat_power(), degraded=degraded)[0] / width
        values = curve(np.linspace(energies.min(), energies.max(), 1000))
        assert values.min() <= eta <= values.max()
        assert eta > 0.05  # the old nm/1239.8 conversion gave a constant 0.0156 A/W
