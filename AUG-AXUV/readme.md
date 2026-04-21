# AUG-AXUV

Analysis and synthetic diagnostic toolkit for the AXUV bolometer system at
ASDEX Upgrade (AUG).

The repository is split into two independent Python packages with different
dependency stacks, reflecting the two main use cases:

| Package | Purpose | Key dependencies |
|---|---|---|
| `axuv` | Synthetic forward model (3-D ray tracing) | Raysect, Cherab |
| `axuv_measurement` | Experimental signal analysis | `aug_sfutils` (IPP-only) |

---

## Repository layout

```
AUG-AXUV/                          ← repo root (add this to sys.path)
│
├── axuv/                          ← synthetic diagnostics package
│   ├── __init__.py
│   ├── geometry.py                ← coordinate transforms, axis constants
│   ├── cameras.py                 ← Raysect scene builder, SECTOR_CAMERAS dict
│   ├── plasma.py                  ← Cherab spectral emission helpers
│   ├── responsivity.py            ← AXUV diode responsivity curves
│   ├── interpolation.py           ← JOREK → poloidal-grid interpolation
│   ├── io.py                      ← ray-transfer HDF5 loading, path constants
│   ├── plotting.py                ← visualisation helpers
│   ├── raytransfer_sensitivity.py ← Stage 1 CLI script (geometry matrix)
│   ├── calculate_emissions_for_raytransfer.py  ← Stage 2 (plasma emissions)
│   ├── emission_to_measurement.py ← Stage 3 (synthetic signals)
│   └── scripts/                   ← SLURM submission scripts
│
├── axuv_measurement/              ← experimental signal analysis package
│   ├── __init__.py                ← re-exports full public API
│   ├── config.py                  ← signal names, diagnostic aliases,
│   │                                 channel ranges, ellipse mask, databases
│   ├── io.py                      ← HDF5 export/import pipeline
│   │                                 (axuv_to_hdf, calibrate_and_smooth, …)
│   ├── geometry.py                ← LOS class, q-surface utilities,
│   │                                 intersection-data generator
│   ├── processing.py              ← downsampling, repair, ridge filter,
│   │                                 interpolation helpers
│   ├── plotting.py                ← all plot_* / radiation_* / save_* functions
│   └── animation.py               ← poloidal scatter and line-plot animations
│
├── tests/                         ← unit tests (cover axuv/ only; no AUG
│   ├── test_cameras.py              server required)
│   ├── test_geometry.py
│   ├── test_interpolation.py
│   └── test_plasma.py
│
├── measurement/                   ← scripts, notebooks, and calibration files
│   │                                 for working with experimental data
│   ├── amplification_lookup.py
│   ├── blc_xvu_calibration.py
│   ├── xvu_amplification.py
│   └── *.ipynb
│
├── aug/                           ← AUG vessel / diagnostic geometry data
│   ├── vessel/, divertor/, …
│
├── sightline_DREAMoutput_Ne_SPI.py  ← simple LOS model on DREAM SPI output
├── requirements.txt
└── pyrightconfig.json
```

---

## Setup

### On TOKI (IPP server) — full functionality

```bash
module load aug_sfutils
source AUG-AXUV/.venv/bin/activate
```

Both packages are importable once the repo root is on `sys.path`.  The
virtual environment contains all dependencies from `requirements.txt`; only
`aug_sfutils` is loaded separately via the module system.

### Elsewhere — synthetic package only

```bash
pip install -r requirements.txt   # installs Raysect, Cherab, h5py, …
# aug_sfutils is not available outside IPP; axuv_measurement will not work
```

### Making the packages importable

Neither package is installed via pip by default.  Add the repo root to your
Python path — either in your shell:

```bash
export PYTHONPATH="/path/to/AUG-AXUV:$PYTHONPATH"
```

or at the top of a script / notebook:

```python
import sys
sys.path.insert(0, "/path/to/AUG-AXUV")
```

---

## `axuv` — synthetic diagnostics

A three-stage forward model that converts a plasma emission profile into a
predicted per-diode signal.

### Stage 1 — Geometry matrix

Computes how much emission from each plasma voxel reaches each diode via
Raysect ray tracing.  Outputs `raytransfer_<sectors>_[no]refl.h5`.

```bash
# Sector 5, no reflections (fastest, good for iteration)
python axuv/raytransfer_sensitivity.py --sectors S5

# Sector 16, with reflections (requires AUG CAD files)
python axuv/raytransfer_sensitivity.py --sectors S16 --reflections

# Both sectors, custom resolution, explicit output path
python axuv/raytransfer_sensitivity.py --sectors S5 S16 \
    --resolution-r 30 --resolution-z 55 --pixel-samples 10000 \
    --output /data/results/raytransfer_both.h5

# Mask voxel grid to actual plasma extent from a JOREK file
python axuv/raytransfer_sensitivity.py --sectors S5 \
    --jorek-file data/step02410_out.h5
```

### Stage 2 — Plasma emissions

Interpolates JOREK data onto the voxel grid, evaluates Cherab spectral
emission models, and folds with the geometry matrix.

```bash
python axuv/calculate_emissions_for_raytransfer.py
```

### Stage 3 — Synthetic signals

Folds spectral power through the AXUV diode responsivity curve to produce a
scalar signal per diode per timestep.

```bash
python axuv/emission_to_measurement.py
```

### Quick import examples

```python
import axuv

# Coordinate transform (no I/O at import time)
xyz = axuv.toroidal_to_cartesian(R=1.8, phi=22.5, z=0.0)

# Camera config dict — also safe at package level
from axuv.cameras import SECTOR_CAMERAS, create_observable_world
from axuv.io import AXUV_DF
world, cameras = create_observable_world(["S16"], AXUV_DF)

# Responsivity (CSV loaded lazily on first call)
from axuv.responsivity import get_weighted_power
```

---

## `axuv_measurement` — experimental signal analysis

Tools for extracting, calibrating, and analysing experimental AXUV signals
from AUG shotfiles.  Requires `aug_sfutils` and access to the AUG shot-file
server.

### Workflow overview

1. **Export** raw shotfile data to HDF5:
   `axuv_measurement.io.axuv_to_hdf(shot)`
2. **Calibrate and smooth** to a working HDF5 file:
   `axuv_measurement.io.calibrate_and_smooth(shot)`  ← pending logic rewrite
3. **Analyse** using plotting, processing, and geometry helpers.

### Sub-module overview

| Module | Key contents |
|---|---|
| `config` | `SIGNAL_NAMES`, diagnostic aliases (`D16`, `DHT`, …), channel ranges, ellipse mask, pre-loaded CSV databases |
| `io` | `axuv_to_hdf`, `calibrate_and_smooth`, `process_signals`, `read_data_base`, `calculate_indices` |
| `geometry` | `LOS` class, `plot_qsurfaces`, `get_intersecting_LOSs`, `generate_isect_data3d`, `_load_isect_matrix`, `_filter_by_polygon` |
| `processing` | `downsample`, `repair_2d_data`, `ridge_filter`, `upsample_interpolate`, `find_roots` |
| `plotting` | `plot_diag_non_mapped`, `plot_diff`, `plot_current`, `radiation_poloidal`, `radiation_inside_q2surface`, `plot_overview_1`, `save_sqrt`, `sum_LOSs`, `save_LOS_sums`, … |
| `animation` | `animate_poloidal`, `interp_anim`, `animate_with_q2` and their per-frame `update_*` callbacks |

### Quick import examples

```python
# Import what you need directly from the relevant sub-module
from axuv_measurement.config import D16, DHT, VERT_S16_RANGE
from axuv_measurement.io import read_data_base, calculate_indices
from axuv_measurement.plotting import plot_current, radiation_poloidal
from axuv_measurement.processing import ridge_filter, downsample
from axuv_measurement.geometry import LOS, plot_qsurfaces
from axuv_measurement.animation import animate_poloidal

# Or import everything via the package-level re-export
import axuv_measurement as am
am.plot_current(41000, start=2.3, end=2.45)
```

### Notes

- `axuv_to_hdf` and `calibrate_and_smooth` in `io.py` are intentionally kept
  verbatim pending a logic rewrite.
- The `measurement/` directory contains standalone scripts and Jupyter
  notebooks that import from `axuv_measurement`.
- The `radiation_inside_q2surface` function replaces the former pair
  `radiation_inside_q2surface` / `radiation_inside_q2surface_2`; pass
  `use_sqrt=True` for the square-root-weighted variant.

---

## Tests

Tests live in `tests/` and cover the `axuv` synthetic package only; they
require Raysect and Cherab but **not** `aug_sfutils`, so they can run in any
CI environment.

```bash
pytest tests/
```

To skip tests that require optional heavy dependencies (Raysect scene
construction, CAD file access):

```bash
pytest tests/ -m "not integration"
```

---

## Notes and caveats

- `aug_sfutils` is available only on the IPP TOKI server to users in the AUG
  group.  All `axuv_measurement` functionality is therefore IPP-only.
- The synthetic `axuv` package requires the AXUV diode geometry data and
  JOREK plasma input; reflection calculations additionally require access to
  the AUG vessel CAD files.
- For Ne spectral lines: lines 150–151 in `populate()` in
  `cherab/core/cherab/openadas/repository/create.py` can be used as a
  template to add Ne 9+ data, after which `populate()` can be called to
  download the data from OpenADAS.