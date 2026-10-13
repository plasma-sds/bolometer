# AUG-AXUV

Analysis and synthetic diagnostic toolkit for the AXUV bolometer system at
ASDEX Upgrade (AUG).

## Important requirements
- Linux (Cherab requirement)
- Python 3.12

---

## Repository layout

The repository is split into two independent Python packages with different
dependency stacks, reflecting the two main use cases:

| Package | Purpose | Key dependencies |
|---|---|---|
| `axuv` | Synthetic forward model (3-D ray tracing) | Raysect, Cherab |
| `axuv_measurement` | Experimental signal analysis | `aug_sfutils` (IPP-only) |


```
AUG-AXUV/                          ← repo root
│
├── axuv/                          ← synthetic diagnostics package
│   ├── __init__.py
│   ├── geometry.py                ← coordinate transforms, axis constants
│   ├── cameras.py                 ← Raysect scene builder, SECTOR_CAMERAS dict
│   ├── plasma.py                  ← Cherab spectral emission helpers
│   ├── responsivity.py            ← AXUV diode responsivity curves
│   ├── interpolation.py           ← JOREK → poloidal-grid interpolation
│   ├── datafiles.py               ← data keys, local paths via plasmasds_utility
│   ├── io.py                      ← ray-transfer HDF5 loading, output paths,
│   │                                 WTH_THRESHOLDS (W_th line levels)
│   ├── plotting.py                ← visualisation helpers
│   ├── cad_files.py               ← AUG vessel CAD mesh keys/helpers
│   ├── raytransfer_sensitivity.py ← Stage 1 CLI script (geometry matrix)
│   ├── calculate_emissions_for_raytransfer.py  ← Stage 2 (plasma emissions)
│   ├── emission_to_measurement.py ← Stage 3 (synthetic signals)
│   └── shell_scripts/             ← SLURM submission + plotting wrappers
│                                     (see shell_scripts/run_scripts.md)
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
│   ├── plotting_from_shotfiles.py ← saves experimental camera figures
│   │                                 (with W_th vertical lines)
│   └── animation.py               ← poloidal scatter and line-plot animations
│
├── tests/                         ← unit tests (cover axuv/ only; no AUG
│   ├── test_cameras.py              server required)
│   ├── test_geometry.py
│   ├── test_interpolation.py
│   └── test_plotting.py
│
├── publications/                  ← figure-generating scripts for the paper
│   ├── plot_time_traces.py        ← W_th time traces; writes
│   │                                 output/threshold_times.json
│   ├── plot_sensitivity_matrices.py, plot_etendue.py, plot_sightlines.py,
│   │   plot_degraded_sensitivity.py, compare_highres_lowres.py,
│   │   plot_1D_dream_spectra.py, verify_uniform_emitter.py
│
├── measurement_utils/             ← scripts and calibration files for
│   │                                 working with experimental data
│   ├── amplification_lookup.py
│   ├── blc_xvu_calibration.py
│   ├── xvu_amplification.py
│   ├── signals.txt
│   └── data/
│
├── output/                        ← generated figures, CSVs, threshold_times.json
│
├── sightline_DREAMoutput_Ne_SPI.py  ← simple LOS model on DREAM SPI output
└── pyproject.toml                 ← editable-install / packaging config
```

---

## Setup

### Prerequisites

- Linux (Cherab requirement).
- Python 3.12. Cherab 1.5 requires Raysect 0.8.1 and NumPy < 2, which have no wheels for newer Python versions, so `requires-python` is capped at `<3.13`.
- [uv](https://docs.astral.sh/uv/) (recommended) or pip.

### Install with uv (recommended)

`uv.lock` pins the tested dependency set. From the `AUG-AXUV` folder:

```bash
cd AUG-AXUV
uv sync                          # creates .venv with Python 3.12, installs locked deps + pytest
source .venv/bin/activate
```

`uv sync` downloads Python 3.12 if it is not available on the system. Add `--extra measurement` to include `aug_sfutils`.

### Install with pip

pip ignores `uv.lock` and resolves within the bounds in `pyproject.toml`. Create the environment with a Python 3.12 interpreter:

```bash
cd AUG-AXUV
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .                 # add ".[measurement]" for aug_sfutils
pip install pytest               # the dev group in pyproject.toml is not installed by pip install -e
```

Both methods install `axuv` and `axuv_measurement` in editable mode, which is the only supported install method: the packages read their data files relative to the source tree, and the data is not included in a built wheel.

### Atomic data (Cherab / OpenADAS)

Stage 2 and `sightline_DREAMoutput_Ne_SPI.py` evaluate emission with `cherab.openadas.OpenADAS`, which reads a local atomic-data repository (`~/.cherab/openadas/`). Populate it once per user after installing:

```bash
python -c "from cherab.openadas.repository import populate; populate()"
```

This downloads the default OpenADAS rate data. For Ne 9+ lines, see the note under [Notes and caveats](#notes-and-caveats).

### Data files

The input data is not tracked in git. It is served by the plasma-sds data server and fetched with [plasmasds_utility](https://github.com/plasma-sds/plasmasds_utility) under the client prefix `bolometer`, with the AUG-AXUV files below `aug-axuv/`.
The code names each file by its *data key*, the path below `aug-axuv/`, and gets its local path from `axuv.datafiles.data_path(key)`, which downloads the file first if it is not on disk.

| Data key | Contents | Tree | Used by |
|---|---|---|---|
| `geometry/AXUV_LOS_geom.txt` | diode and aperture geometry | private | `axuv.io`, camera setup (all stages) |
| `geometry/gc_d_lines.obj` | poloidal cross-section outline | private | `axuv.io.load_gc_d_lines`, plotting |
| `geometry/etendue_S5.h5`, `geometry/etendue_S16.h5` | ray-traced etendue (Stage 1 product) | private | Stage 3, `plot_etendue.py` |
| `responsivity/*.csv` | nominal, degraded and worst-case responsivity, error bars | public | `axuv.responsivity` (Stage 3) |
| `atomic/ne.csv` | Ne line list for Cherab | public | Stage 2, sightline model |
| `cad/{vessel,divertor,icrh,a_b_coils}/...` | AUG vessel CAD meshes (`.rsm`) | private | Stage 1 with `--reflections` |
| `raytransfer/raytransfer_{S5,S16}_{refl,norefl}.h5` | geometry matrices produced by Stage 1 | private | Stage 2, publication scripts |
| `jorek/step02410_out.h5`, `jorek/<shot>/<phase>/stepNNNNN_out.h5` | JOREK output files | private | Stage 1 (`--jorek-file` masking), Stage 2 |
| `traces/<shot>/{frad,ip,wth}-{exp,jorek}-<shot>.txt` | time traces | private | `plot_time_traces.py` |
| `dream/dream_output.h5` | DREAM run | public | sightline model, `plot_1D_dream_spectra.py` |

`DATA_TREES` in `axuv/datafiles.py` maps the first folder of a key to its tree.
Private data needs an SSH key accepted by the data server (`plasmasds_utility.set_ssh_key`); public data is downloaded over HTTPS.
Local copies are kept in `<data dir>/bolometer/{private,public}/aug-axuv/<key>`, by default under `~/.local/share/plasmasds/`.
Data copied by hand into the old layout (`axuv/data/` and `aug/`) is placed at the local paths of its keys with `python -m axuv.migrate_data` (hard links, no extra disk space; `--dry-run` shows the mapping, `--copy` copies instead).

The Stage 1 `--jorek-file` and the Stage 2 input file are paths; print the local path of a key with:

```bash
python -c "from axuv.datafiles import data_path; print(data_path('jorek/step02410_out.h5'))"
```

### Environment variables

These optional variables are read:

| Variable | Default | Meaning |
|---|---|---|
| `PLASMASDS_DATA_DIR` | `~/.local/share/plasmasds` | where plasmasds_utility keeps the local copies of the data |
| `PLASMASDS_SERVER_ACCESS` | `all` (`public` when `CI` is set) | `all`, `public` (no private server), or `local` (no network: local copies only) |
| `PLASMASDS_QUIET` | unset | when set, the utility's once-per-process notices go to its log only |
| `AXUV_OUTPUT_DIR` | `output/` in the `AUG-AXUV` folder | root of everything the workflow writes: Stage 1 files, Stage 2 emissions, Stage 3 results, figures |
| `currentdir` | unset | dataset of a Stage 2 run, e.g. `40673/P1`; used when `--dataset` is not given |

Stage 2 writes to `$AXUV_OUTPUT_DIR/<dataset>/emissions/`, with the dataset given by `--dataset` (or `currentdir`), or to the directory given by `--output-dir`.
The shell scripts in `axuv/shell_scripts/` put their logs in `$AXUV_OUTPUT_DIR/<dataset>/stdout/`.

The scripts in `axuv/shell_scripts/` still take their JOREK inputs and ray-transfer matrices from `axuv/data/` (`datadir`), and the cluster scripts hardcode the server path of the checkout in `maindir`; adjust them before use. The parallel submission scripts also need GNU `parallel`.

### `aug_sfutils` and shotfile access

`axuv_measurement` needs `aug_sfutils`, installed from PyPI by the `measurement` extra. The package installs anywhere, but reading AUG shotfiles requires access to the AUG shotfile system, available on the IPP TOKI cluster to members of the AUG group. Without it, only the synthetic `axuv` package is functional.

On TOKI, `module load aug_sfutils` provides the IPP-maintained build instead. A virtual environment only sees module-provided packages if it was created with `--system-site-packages`; otherwise use the `measurement` extra.

---

## `axuv` — synthetic diagnostics

A three-stage forward model that converts a plasma emission profile into a
predicted per-diode signal.

### Stage 1 — Geometry matrix

Computes how much emission from each plasma voxel reaches each diode via
Raysect ray tracing.  Outputs `$AXUV_OUTPUT_DIR/raytransfer_<sectors>_[no]refl.h5`
unless `--output` is given. Move the file into the data directory to use it as a
Stage 2 input.

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
    --jorek-file <local path of jorek/step02410_out.h5>
```

### Stage 2 — Plasma emissions

Interpolates JOREK data onto the voxel grid, evaluates Cherab spectral
emission models, and folds with the geometry matrix.

```bash
# The ray-transfer matrix defaults to the data key raytransfer/raytransfer_S16_norefl.h5
python axuv/calculate_emissions_for_raytransfer.py \
    <local path of jorek/40673/P1/step01010_out.h5> --dataset 40673/P1
# writes $AXUV_OUTPUT_DIR/40673/P1/emissions/emissions_norefl_lowres_masked_<time>.h5
```

### Stage 3 — Synthetic signals

Folds spectral power through the AXUV diode responsivity curve to produce a
scalar signal per diode per timestep. Takes the emissions directory produced by
Stage 2 as a positional argument.

```bash
python axuv/emission_to_measurement.py output/40673/P1/emissions
# writes $AXUV_OUTPUT_DIR/40673/P1/<variant>/<combination>/
```

`W_th` vertical lines are overlaid by default at the levels in `WTH_THRESHOLDS`
(`axuv/io.py`); the crossing times are read from `$AXUV_OUTPUT_DIR/threshold_times.json`,
which `publications/plot_time_traces.py` must have produced first. Override the
levels with `--thresholds T1 T2`, or disable the overlay with `--no-thresholds`
(the no-line figures are always saved either way).

To regenerate **all** synthetic figures in one go (runs `plot_time_traces.py`
first, then Stage 3 on every dataset), use the wrapper script:

```bash
axuv/shell_scripts/plot_and_save_synthetic_measurements.sh
```

See `axuv/shell_scripts/run_scripts.md` for the threshold-plotting workflow and
the matching experimental wrapper.

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

- `axuv_to_hdf` and `calibrate_and_smooth` in `io.py` are due for a logic rewrite.
- The `measurement_utils/` directory contains standalone scripts and
  calibration files that import from `axuv_measurement`.
- `plotting_from_shotfiles.py` saves the experimental AXUV camera figures with
  `W_th` vertical lines (levels from `WTH_THRESHOLDS`, times read from
  `output/threshold_times.json`). Toggle the overlay with the
  `PLOT_THRESHOLD_LINES` flag at the top of that file. To regenerate everything
  on the AUG server, use `axuv/shell_scripts/plot_and_save_experimental_measurements.sh`.
- The `radiation_inside_q2surface` function replaces the former pair
  `radiation_inside_q2surface` / `radiation_inside_q2surface_2`; pass
  `use_sqrt=True` for the square-root-weighted variant.

---

## Tests

Tests live in `tests/` and cover the `axuv` synthetic package only; they
require Raysect and Cherab but **not** `aug_sfutils`, and no data files: the
`axuv.datafiles` tests run plasmasds_utility on a temporary local tree.

```bash
pytest tests/
```

---

## Notes and caveats

- Reading shotfiles with `aug_sfutils` requires AUG shotfile access (IPP TOKI,
  AUG group), so `axuv_measurement` is effectively IPP-only.
- The synthetic `axuv` package requires the AXUV diode geometry data and
  JOREK plasma input; reflection calculations additionally require access to
  the AUG vessel CAD files.
- For Ne spectral lines: lines 150–151 in `populate()` in
  `cherab/core/cherab/openadas/repository/create.py` can be used as a
  template to add Ne 9+ data, after which `populate()` can be called to
  download the data from OpenADAS.
