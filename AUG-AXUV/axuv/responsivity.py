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

# Responsivity data lives alongside the package source in axuv/data/
_DATA_DIR = Path(__file__).parent / "data"

# Photon energy threshold [eV] below which the nominal and degraded curves
# are identical.  Above this value the SiO2 entrance window transmittance
# drops for radiation-damaged diodes.
DEGRADED_THRESHOLD_EV: float = 89.4308

# Module-level cache — populated once on the first call
_sensitivity: np.ndarray | None = None
_degraded_sensitivity: np.ndarray | None = None


def _load_responsivity() -> None:
    """Load the responsivity CSV files into the module cache (idempotent)."""
    global _sensitivity, _degraded_sensitivity
    if _sensitivity is not None:
        return
    _sensitivity = np.genfromtxt(_DATA_DIR / "axuv_sensitivity.csv", delimiter=",")
    _degraded_sensitivity = np.genfromtxt(_DATA_DIR / "degraded_sensitivity.csv", delimiter=",")


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


def get_weighted_power(
    diode_data: np.ndarray,
    spectrum_energies: np.ndarray,
    degraded: bool = False,
) -> np.ndarray:
    """
    Integrates per-bin spectral power against the AXUV responsivity curve.

    Parameters
    ----------
    diode_data : ndarray of shape (num_diodes, num_energy_bins)
        Spectral power for each diode at each energy bin centre.
    spectrum_energies : ndarray of shape (num_energy_bins,)
        Photon energy [eV] at each bin centre.  May be in any order but must
        be consistent with the column axis of diode_data.
    degraded : bool
        If True, use the degraded-diode responsivity curve; otherwise nominal.

    Returns
    -------
    weighted_power : ndarray of shape (num_diodes,)
    """
    responsivity = degraded_sensitivity_function if degraded else sensitivity_function
    n = len(spectrum_energies)
    weighted_power = np.zeros(diode_data.shape[0])

    for i, energy in enumerate(spectrum_energies):
        # Half-widths to adjacent bin centres; mirror boundary conditions
        dE_1 = (
            abs(energy - spectrum_energies[i - 1]) / 2
            if i > 0
            else abs(energy - spectrum_energies[i + 1]) / 2
        )
        dE_2 = (
            abs(energy - spectrum_energies[i + 1]) / 2
            if i < n - 1
            else abs(energy - spectrum_energies[i - 1]) / 2
        )

        # spectrum_energies is in decreasing order, so range is [energy+dE_1, energy−dE_2]
        averaging_range = np.linspace(energy + dE_1, energy - dE_2, 100)
        average_weight = float(np.average(responsivity(averaging_range)))
        weighted_power += diode_data[:, i] * average_weight

    return weighted_power
