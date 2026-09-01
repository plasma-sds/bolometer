"""
axuv.responsivity
-----------------
Spectral responsivity of AXUV diodes and weighted-power integration.

Two responsivity curves are provided:
  - Nominal  (sensitivity_function): for undamaged diodes
  - Degraded (degraded_sensitivity_function): for diodes exposed to high
    radiation doses where the responsivity above ~89.4 eV is reduced

The CSV data files (axuv_sensitivity.csv, degraded_avg.csv) live in
axuv/data/ and are loaded lazily on first use so that importing this module
does not crash when the files are unavailable (e.g. during unit tests).

CSV format: two columns — photon_energy_eV, responsivity_A_per_W
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from axuv.cameras import WAVELENGTH_BIN_EDGES

# Responsivity data lives alongside the package source in axuv/data/
_DATA_DIR = Path(__file__).parent / "data"

# Photon energy threshold [eV] below which the nominal and degraded curves
# are identical.  Above this value the SiO2 entrance window transmittance
# drops for radiation-damaged diodes.
DEGRADED_THRESHOLD_EV: float = 89.4308
DETECTOR_CALIBRATION_AMPER_PER_WATT: float = 0.27

# Module-level cache — populated once on the first call
_sensitivity: np.ndarray | None = None
_degraded_sensitivity: np.ndarray | None = None
_worst_estimation: np.ndarray | None = None


def _load_responsivity() -> None:
    """Load the responsivity CSV files into the module cache (idempotent)."""
    global _sensitivity, _degraded_sensitivity, _worst_estimation
    if _sensitivity is not None:
        return
    _sensitivity = np.genfromtxt(_DATA_DIR / "axuv_sensitivity.csv", delimiter=",")
    _degraded_sensitivity = np.genfromtxt(_DATA_DIR / "degraded_sensitivity.csv", delimiter=",")
    try:  # This is not strictly necessary for 99% of the work
        _worst_estimation = np.genfromtxt(_DATA_DIR / "worst_estimation.csv", delimiter=",")
    except FileNotFoundError:
        _worst_estimation = None


def sensitivity_function(where: np.ndarray | float) -> np.ndarray | float:
    """
    Nominal AXUV spectral responsivity [A/W] as a function of photon energy [eV].
    Interpolates the tabulated calibration data.
    """
    _load_responsivity()
    return np.interp(where, _sensitivity[:, 0], _sensitivity[:, 1])


def degraded_sensitivity_function(where: np.ndarray | float) -> np.ndarray | float:
    """
    Degraded AXUV spectral responsivity [A/W] as a function of photon energy [eV].

    Below DEGRADED_THRESHOLD_EV the nominal curve is used; above it the
    degraded curve is applied.  Accepts scalars, lists, or NumPy arrays.
    """
    _load_responsivity()
    arr = np.atleast_1d(np.asarray(where, dtype=float))
    scalar = np.ndim(where) == 0

    result = np.where(
        arr < DEGRADED_THRESHOLD_EV,
        np.interp(arr, _sensitivity[:, 0], _sensitivity[:, 1]),
        np.interp(arr, _degraded_sensitivity[:, 0], _degraded_sensitivity[:, 1]),
    )
    return float(result[0]) if scalar else result


def worst_sensitivity(where: np.ndarray | float) -> np.ndarray | float:
    """
    Worst estimated AXUV spectral responsivity [A/W] as a function of photon energy [eV].
    """
    _load_responsivity()
    arr = np.atleast_1d(np.asarray(where, dtype=float))
    scalar = np.ndim(where) == 0

    result = np.interp(arr, _worst_estimation[:, 0], _worst_estimation[:, 1])
    return float(result[0]) if scalar else result


def get_weighted_power(
    diode_data: np.ndarray,
    degraded: bool = False,
) -> np.ndarray:
    """
    Integrates per-bin spectral power [W/nm] against the AXUV responsivity curve [A/(W nm)], resulting in current [A].
    Uses WAVELENGTH_BIN_EDGES to calculate the energy bin edges for the integration.

    Parameters
    ----------
    diode_data : ndarray of shape (num_diodes, num_energy_bins)
        Spectral power for each diode at each energy bin centre.
    degraded : bool
        If True, use the degraded-diode responsivity curve; otherwise nominal.

    Returns
    -------
    weighted_power : ndarray of shape (num_diodes,)
        Integrated power for each diode weighted by the responsivity curve.
    """
    energy_bin_edges = WAVELENGTH_BIN_EDGES / 1239.8  # nm to eV
    responsivity = degraded_sensitivity_function if degraded else sensitivity_function
    weighted_power = np.zeros(diode_data.shape[0])

    for i in range(len(energy_bin_edges) - 1):
        averaging_range = np.linspace(energy_bin_edges[i], energy_bin_edges[i+1], 100)
        average_weight = float(np.average(responsivity(averaging_range)))
        # for the integration we need to multiply by the bin width in wavelength!
        bin_width = abs(WAVELENGTH_BIN_EDGES[i+1] - WAVELENGTH_BIN_EDGES[i])
        weighted_power += diode_data[:, i] * average_weight * bin_width

    return weighted_power
