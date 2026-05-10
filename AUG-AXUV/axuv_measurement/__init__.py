"""
axuv  –  AXUV bolometry analysis package for ASDEX-Upgrade SPI experiments.

This package re-exports the complete public API from its sub-modules so that
existing code using ``from axuv import *`` or ``import axuv; axuv.plot_current(…)``
continues to work without modification.

Sub-module layout
-----------------
config      : constants, signal names, diagnostic aliases, databases
io          : HDF5 I/O pipeline (axuv_to_hdf, calibrate_and_smooth, …)
geometry    : LOS class, q-surface utilities, intersection-data generator
processing  : downsampling, repair, ridge filter, interpolation helpers
plotting    : all plot_* / radiation_* / save_* / sum_* functions
animation   : animate_* / interp_* / update_* animation functions
"""

# ── config ──────────────────────────────────────────────────────────────────
from .config import (
    ROOTFOLDER,
    _make_signals,
    SIGNAL_NAMES,
    SIG_KEY_LIST,
    D15, D16, D01, DVC, D13, DHC, DHT,
    VERT, HORIZ, DIAGNAMES_IN_S5_AND_S16, NEWAXUV,
    VERT_S5_RANGE, HORIZ_S5_RANGE, VERT_S16_RANGE, HORIZ_S16_RANGE,
    ELLIPSE_U, ELLIPSE_V, ELLIPSE_A, ELLIPSE_B,
    ELLIPSE_T, ELLIPSE_R, ELLIPSE_XY, ELLIPSE,
    SPI_DB, SPI_COLUMN_NAMES,
    CURRENT_DB,
    DESUBLIM_DB,
    FULL_DB,
    LOS_DB,
)

# ── io ───────────────────────────────────────────────────────────────────────
from .io import (
    get_AXUV_signals,
)

# ── geometry ─────────────────────────────────────────────────────────────────
from .geometry import (
    _load_isect_matrix,
    _filter_by_polygon,
    LOS,
    show_poloidal,
    plot_qsurfaces,
    generate_isect_data3d,
    get_intersecting_LOSs,
)

# ── processing ───────────────────────────────────────────────────────────────
from .processing import (
    find_roots,
    upsample_interpolate,
    downsample,
    repair_2d_data,
    ridge_filter,
)

# ── plotting ─────────────────────────────────────────────────────────────────
from .plotting import (
    set_plt_rcparams,
    plot_1D,
    plot_full_integral,
    plot_integral_combined,
    plot_current,
    plot_timeslice,
    radiation_poloidal,
    radiation_inside_q2surface,
    plot_overview_1,
    save_sqrt,
    sum_LOSs,
    save_LOS_sums,
)

# ── animation ────────────────────────────────────────────────────────────────
from .animation import (
    animate_poloidal,
    update_animate_poloidal,
    interp_anim,
    update_interp_anim,
    animate_with_q2,
    update_with_q2,
)

__all__ = [
    # config
    "ROOTFOLDER", "_make_signals", "SIGNAL_NAMES", "SIG_KEY_LIST",
    "D15", "D16", "D01", "DVC", "D13", "DHC", "DHT",
    "VERT", "HORIZ", "DIAGNAMES_IN_S5_AND_S16", "NEWAXUV",
    "VERT_S5_RANGE", "HORIZ_S5_RANGE", "VERT_S16_RANGE", "HORIZ_S16_RANGE",
    "ELLIPSE_U", "ELLIPSE_V", "ELLIPSE_A", "ELLIPSE_B",
    "ELLIPSE_T", "ELLIPSE_R", "ELLIPSE_XY", "ELLIPSE",
    "SPI_DB", "SPI_COLUMN_NAMES", "CURRENT_DB", "DESUBLIM_DB", "FULL_DB", "LOS_DB",
    # io
    "get_AXUV_signals",
    # geometry
    "_load_isect_matrix", "_filter_by_polygon", "LOS",
    "show_poloidal", "plot_qsurfaces", "generate_isect_data3d", "get_intersecting_LOSs",
    # processing
    "find_roots", "upsample_interpolate", "downsample", "repair_2d_data", "ridge_filter",
    # plotting
    "set_plt_rcparams", "plot_1D", "plot_full_integral",
    "plot_integral_combined", "plot_current", "plot_timeslice",
    "radiation_poloidal", "radiation_inside_q2surface",
    "plot_overview_1", "save_sqrt", "sum_LOSs", "save_LOS_sums",
    # animation
    "animate_poloidal", "update_animate_poloidal",
    "interp_anim", "update_interp_anim",
    "animate_with_q2", "update_with_q2",
]