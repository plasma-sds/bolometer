# %%
"""
plot_radiated_power.py
----------------------
Poloidal maps of the total radiated power per voxel, one figure per JOREK
timestep, plus an animated GIF per discharge.

This mirrors the plotting method of ``plot_sensitivity_matrices.py`` (same
poloidal voxel grid, ``plot_sensitivity_map`` renderer, PFC overlay and
``make_gif`` stitching), but colours each voxel by the radiated power rather
than by a diode sensitivity.

Two modes (``MODES``):
  · "total"    — total emitted power density, integrated over wavelength and
                 4*pi sr.  Real units: [W/m^3].
  · "weighted" — the same emission folded through the *degraded* AXUV spectral
                 responsivity curve.  Units are arbitrary: [a. u.].

Per-voxel spectral emissivity is read from the Stage-2 emission HDF5 files
produced by ``calculate_emissions_for_raytransfer.py`` (one ``emissions_*.h5``
per timestep); the poloidal grid geometry is read from a ray-transfer file.  A
single emission ``VARIANT`` is selected below, and its *resolution* must match
``RAYTRANSFER_PATH`` (the per-voxel emissivity is the same for the refl/norefl
variants of a given resolution).

The colour scale is fixed across all timesteps of a given (shot, mode) so the
frames animate cleanly, and the time is shown in a fixed-width font as
``2.3 s + XXX ms`` so the title does not jump horizontally during the GIF.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from axuv.io import PROJECT_ROOT, gc_d_lines, open_emission_data
from axuv.plotting import plot_sensitivity_map, set_plt_rcparams
from axuv.responsivity import get_weighted_power

# LaTeX/serif article styling (also enables constrained_layout so nothing clips)
set_plt_rcparams()

project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"

# An output directory can hold several emission variants
# (emissions_{refl|norefl}_{low|high}res[_masked]_<time>.h5) with overlapping
# time bases.  Pick exactly one variant so frames are not duplicated.  Note the
# per-voxel `emissions` are the pure plasma emissivity and are identical between
# the refl/norefl variants of the same resolution, so only the *resolution* has
# to match the ray-transfer grid below.
VARIANT = "refl_lowres_masked"

# The ray-transfer file only supplies the voxel grid geometry here; radiated
# power per voxel is a plasma-only quantity, so a single (sector-independent)
# grid is enough.  Its resolution must match VARIANT (lowres here).
RAYTRANSFER_PATH = data_dir / "raytransfer_S16_reflections_lowres.h5"

# discharge -> (directory of emissions_*.h5, absolute start time [s])
# Start times mirror plot_time_traces.py / emission_to_measurement.py.
SHOTS = {
    "40673": (data_dir / "40673" / "P1" / "output", 2.3276),
    "41007": (data_dir / "41007" / "P1" / "output", 2.3417),
}

MODES = ["total", "weighted"]

# Since we use imshow, set the figure size explicitly.  A fixed size (no
# bbox="tight") keeps every saved frame pixel-identical, which is what makes the
# GIF animate without shifting; constrained_layout keeps the axis labels,
# colorbar and title inside the figure so they are never clipped.  The width is
# kept snug against the (tall, equal-aspect) poloidal axes + colorbar so there is
# no wide empty margin beside the figure.
figsize = (5.4, 6.5)
frame_dpi = 150

output_root = project_dir / "output" / "radiated_power"
output_root.mkdir(parents=True, exist_ok=True)

# Emission filenames follow  emissions_<VARIANT>_<time>.h5 ; strip this fixed
# prefix (and the .h5 suffix) to recover the physical time [s].
_EMISSION_GLOB = f"emissions_{VARIANT}_*.h5"
_TIME_PREFIX = f"emissions_{VARIANT}_"

print(f"Project root: {project_dir}")


# %% Helpers
def plot_PFCs(ax, gc_d_lines, linewidth=0.5, color="white", linestyle="-"):
    """Draw the plasma-facing-component (vessel) contours."""
    for line in gc_d_lines:
        ax.plot(line[0], line[1], lw=linewidth, c=color, ls=linestyle)


def make_gif(image_dir, pattern, output_path, duration=150, scale=0.5):
    """Create an animated GIF from PNGs in image_dir matching pattern.

    Files are sorted numerically by the integer embedded in their stem.
    duration: frame duration in milliseconds.
    scale: resize factor applied to each frame before encoding (reduces file size).
    """
    from PIL import Image

    paths = sorted(
        image_dir.glob(pattern),
        key=lambda p: int("".join(filter(str.isdigit, p.stem))),
    )
    if not paths:
        print(f"No images found for pattern '{pattern}' in {image_dir}")
        return
    frames = []
    for p in paths:
        img = Image.open(p)
        if scale != 1.0:
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
            )
        frames.append(img)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF ({len(frames)} frames): {output_path}")


def emission_file_time(path):
    """Physical time [s] parsed from an emission filename."""
    return float(path.name.removeprefix(_TIME_PREFIX).removesuffix(".h5"))


def per_voxel_power(emissions, wavelengths, mode):
    """Reduce the per-voxel spectral emissivity to a single scalar per voxel.

    :param emissions:   (n_voxels, n_bins) spectral emissivity [W m^-3 sr^-1 nm^-1].
    :param wavelengths: (n_bins + 1,) wavelength bin edges [nm].
    :param mode:        "total" -> radiated power density [W/m^3] (4*pi * integral
                        over wavelength); "weighted" -> folded through the degraded
                        AXUV responsivity [a. u.].
    """
    if mode == "total":
        bin_widths = np.diff(wavelengths)  # nm, (n_bins,)
        return 4 * np.pi * (emissions * bin_widths[np.newaxis, :]).sum(axis=1)
    if mode == "weighted":
        # get_weighted_power folds each bin through the responsivity curve and
        # sums; it treats axis 0 as an independent-channel axis, so the voxel
        # axis works in place of the diode axis.
        return get_weighted_power(emissions, degraded=True)
    raise ValueError(f"Unknown mode: {mode!r}")


def mode_cbar_label(mode):
    if mode == "total":
        return r"Radiated power density [W/m$^3$]"
    return r"Weighted radiated power [a. u.]"


def format_title(shot, t_abs, base_s):
    """Plot title: shot number + fixed-width time (base seconds + offset in ms).

    e.g. shot=40673, base_s=2.3, t_abs=2.3276 -> ``#40673  2.3 s + 27.60 ms``.
    The ms field is zero-padded (and the shot number is constant per animation)
    so the title keeps a constant width across the frames of a GIF.
    """
    ms = (t_abs - base_s) * 1e3
    return rf"\#{shot}\quad \texttt{{{base_s:.1f}\,s + {ms:05.2f}\,ms}}"


# %% Load the poloidal voxel grid once
# Only the grid geometry is needed, so read those two datasets directly rather
# than pulling the multi-GB sensitivity matrix through load_raytransfer_data.
with h5py.File(RAYTRANSFER_PATH, "r") as h5f:
    grid_centres = h5f["grid_centres"][()]
    voxel_map = h5f["voxel_map"][()]
n_active_voxels = int((voxel_map >= 0).sum())
print(
    f"Grid loaded from {RAYTRANSFER_PATH.name}: "
    f"R resolution {grid_centres.shape[0]}, Z resolution {grid_centres.shape[1]}, "
    f"{n_active_voxels} active voxels"
)


# %% Render frames and build one GIF per (shot, mode)
for shot, (emissions_dir, start_time) in SHOTS.items():
    emissions_dir = Path(emissions_dir)
    files = sorted(emissions_dir.glob(_EMISSION_GLOB), key=emission_file_time)
    if not files:
        print(
            f"[{shot}] No '{_EMISSION_GLOB}' files in {emissions_dir} — skipping."
        )
        continue

    # Absolute time axis: first timestep -> start_time, then elapsed on top
    # (mirrors emission_to_measurement.py).
    file_times = np.array([emission_file_time(f) for f in files])
    abs_times = start_time + (file_times - file_times.min())
    base_s = 2.3

    # Preload the per-voxel arrays once per mode so the colour scale can be
    # fixed across every frame.
    print(f"[{shot}] {len(files)} timestep(s) from {emissions_dir}")
    for mode in MODES:
        per_voxel = []
        for f in files:
            emission_dict = open_emission_data(str(f))
            emissions = emission_dict["emissions"]
            if emissions.shape[0] != n_active_voxels:
                raise ValueError(
                    f"{f.name}: emissions has {emissions.shape[0]} voxels but the "
                    f"grid {RAYTRANSFER_PATH.name} has {n_active_voxels}. VARIANT "
                    f"({VARIANT}) and RAYTRANSFER_PATH resolutions must match."
                )
            per_voxel.append(
                per_voxel_power(emissions, emission_dict["wavelengths"], mode)
            )

        # Fixed colour limits over all frames so the colorbar is identical
        # throughout the GIF (log scale, so keep vmin a fixed fraction of vmax).
        vmax = max(float(np.nanmax(v)) for v in per_voxel)
        vmin = 1e-4 * vmax

        frame_dir = output_root / shot / mode
        frame_dir.mkdir(parents=True, exist_ok=True)

        for idx, (values, t_abs) in enumerate(zip(per_voxel, abs_times)):
            fig, ax = plt.subplots(figsize=figsize, dpi=frame_dpi)
            ax.set_facecolor("black")
            ax, im = plot_sensitivity_map(
                ax, grid_centres, voxel_map, values, vmin=vmin, vmax=vmax
            )
            plot_PFCs(ax, gc_d_lines)
            plt.colorbar(im, ax=ax, label=mode_cbar_label(mode))
            ax.set_xlabel("R [m]")
            ax.set_ylabel("z [m]")
            ax.set_title(format_title(shot, t_abs, base_s))
            # No bbox="tight": constrained_layout fits the labels while keeping
            # every frame the same pixel size for a stable animation.
            fig.savefig(frame_dir / f"frame_{idx:04d}.png", dpi=frame_dpi)
            plt.close(fig)
            print(f"  [{shot} | {mode}] frame {idx + 1}/{len(files)} done", end="\r")
        print()

        make_gif(frame_dir, "frame_*.png", output_root / f"{shot}_{mode}.gif")

print("Done.")
