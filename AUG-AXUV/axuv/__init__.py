"""
axuv
====
Synthetic AXUV bolometer diagnostics package for ASDEX Upgrade (AUG).

This package supports a three-stage synthetic signal workflow:

  Stage 1 — Geometry matrix  (raytransfer_sensitivity.py)
    Computes how much emission from each plasma voxel reaches each diode
    via Raysect ray tracing.
    Output: raytransfer_<sectors>_[no]refl.h5

  Stage 2 — Plasma emissions  (calculate_emissions_for_raytransfer.py)
    Interpolates JOREK plasma data onto the voxel grid, evaluates cherab
    spectral emission models, and applies the geometry matrix to produce
    per-diode spectral power.
    Output: raytransfer_emissions_<time>.h5

  Stage 3 — Synthetic signals  (emission_to_measurement.py)
    Folds the spectral power through the AXUV diode responsivity curve to
    produce a single scalar signal per diode per timestep.

Submodules
----------
geometry       Coordinate transforms (toroidal ↔ Cartesian, 3-D → R-z)
               and shared axis/origin constants.
interpolation  Interpolates JOREK unstructured-grid data onto a rectangular
               poloidal grid; stores the grid extent constants.
cameras        Camera hardware constants, sector → camera mapping
               (SECTOR_CAMERAS), and the Raysect scene builder.
               Also, the spectral configuration and WAVELENGTH_BIN_EDGES.
plasma         Per-voxel emission function used in Stage 2.
responsivity   AXUV spectral responsivity curves and weighted-power
               integration (get_weighted_power).
io             Environment-variable path constants, AXUV geometry DataFrame,
               and HDF5 data-file loading helpers.
               NOTE: io.py loads files on import — import it explicitly
               rather than relying on package-level re-exports.
plotting       Matplotlib visualisation helpers for interpolated fields,
               diode lines of sight, and voxel emission maps.

Quick start
-----------
    # Light utilities — no file I/O at import time
    import axuv
    result = axuv.toroidal_to_cartesian(R=1.8, phi=22.5, z=0.0)

    # Heavy scene construction — explicit submodule import
    from axuv.cameras import create_observable_world
    from axuv.io import AXUV_DF
    world, cameras = create_observable_world(["S16"], AXUV_DF)

    # Responsivity (loads CSV files lazily on first call)
    from axuv.responsivity import get_weighted_power
"""

__version__ = "0.1.0"

# ── Public API — safe to import at package level ──────────────────────────────
# None of these trigger file I/O or heavy optional-dependency imports.

from axuv.geometry import (
    point3d_to_rz,
    toroidal_to_cartesian,
    XAXIS,
    YAXIS,
    ZAXIS,
    ORIGIN,
)

from axuv.interpolation import (
    interpolate_param,
    interpolate_parameters,
    POLOIDAL_RMIN,
    POLOIDAL_RMAX,
    POLOIDAL_ZMIN,
    POLOIDAL_ZMAX,
)

# SECTOR_CAMERAS is a plain dict — importing it does not construct any Raysect
# objects and is safe at package level.
from axuv.cameras import SECTOR_CAMERAS, WAVELENGTH_BIN_EDGES

# get_weighted_power is pure NumPy; responsivity CSVs are loaded lazily.
from axuv.responsivity import get_weighted_power

__all__ = [
    # geometry
    "point3d_to_rz",
    "toroidal_to_cartesian",
    "XAXIS", "YAXIS", "ZAXIS", "ORIGIN",
    # interpolation
    "interpolate_param",
    "interpolate_parameters",
    "POLOIDAL_RMIN", "POLOIDAL_RMAX",
    "POLOIDAL_ZMIN", "POLOIDAL_ZMAX",
    # cameras
    "SECTOR_CAMERAS",
    "WAVELENGTH_BIN_EDGES",
    # responsivity
    "get_weighted_power",
]
