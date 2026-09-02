from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from axuv.geometry import point3d_to_rz
from axuv.interpolation import (
    POLOIDAL_RMIN, POLOIDAL_RMAX, POLOIDAL_ZMIN, POLOIDAL_ZMAX
)


# Font enlargement factor of the diode time-evolution figures, which are
# printed at a smaller size than the rest of the article figures.
FONT_SCALE = 1.5


def set_plt_rcparams(scale=1.0):
    """Sets the default matplotlib rcParams for LaTeX plotting in articles.

    :param scale: multiplier applied to the base font size, e.g. FONT_SCALE.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.family": "serif",  # tells matplotlib to use \rmfamily in the LaTeX doc
        "font.size": 18 * scale,
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        "image.cmap": 'inferno',
        "lines.linewidth": 2,
    })


def save_png_pdf(fig, path, **kwargs):
    """Saves *fig* both as a PNG and as a PDF, next to each other.

    :param path:   output path; its suffix is replaced by .png and .pdf
    :param kwargs: forwarded to `Figure.savefig`
    """
    path = Path(path)
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), **kwargs)


def add_direction_arrow(ax, orientation, pad=0.05):
    """Draws a vertical arrow left of *ax*, pointing at a coordinate label.

    The diode number axis of the time-evolution plots maps to a machine
    coordinate: R for the vertical cameras (D16, DVC) and z for the horizontal
    ones. The arrow marks the direction of that coordinate: it points down at
    an "R" label at the bottom for the vertical cameras, and up at a "Z" label
    at the top for the horizontal ones.

    The arrow goes left of the y tick labels and of the y axis label, so their
    placement has to be known; the figure is therefore drawn once here.

    :param ax:          axis holding the pcolormesh
    :param orientation: "vertical" or "horizontal" camera type
    :param pad:         gap between the y axis label and the arrow, in units
                        of the axis width
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    left = min(
        to_axes.transform((artist.get_window_extent(renderer).x0, 0))[0]
        for artist in [ax.yaxis.label, *ax.get_yticklabels()]
    )
    x = left - pad

    if orientation == "vertical":
        label, label_y, tail, tip = "R", 0.04, 0.99, 0.13
    else:
        label, label_y, tail, tip = "Z", 0.96, 0.01, 0.87

    ax.annotate(
        "", xy=(x, tip), xytext=(x, tail), xycoords="axes fraction",
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linewidth": plt.rcParams["lines.linewidth"],
            "mutation_scale": plt.rcParams["font.size"],
        },
    )
    ax.text(x, label_y, label, transform=ax.transAxes,
            ha="center", va="center", clip_on=False)

def plot_interpolated(interpolated, title="", cbarlabel="", gc_d_lines=None, show=True):
    """
    Plots interpolated values along with the contours of plasma facing components.
    Takes the values, the figure title and colorbar label as parameters.
    """
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=[6, 6.5])
    ax1.set_aspect(1)
    ax1.set_xlim(1, 2.2)
    ax1.set_ylim(-1.2, 1)
    ims = ax1.imshow(interpolated.T, extent=(1, 2.2, -1.2, 1), origin="lower")
    ax1.set_xlabel("R [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_title(title)

    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax1.plot(line[0], line[1], lw=0.5, c="white")
    cbar = plt.colorbar(ims)
    cbar.set_label(cbarlabel)
    if show:
        plt.show()
    else:
        return ax1, ims


def get_value_at(im, x, y):
    """Function to get the data value at a given axis (x, y) coordinate from an imshow() plot"""
    extent = im.get_extent()
    x_min, x_max, y_min, y_max = extent
    data = im.get_array()

    rows, cols = data.shape

    # Convert axis coordinates (x, y) to array indices
    col = int(round((x - x_min) / (x_max - x_min) * (cols - 1)))
    row = int(round((y - y_min) / (y_max - y_min) * (rows - 1)))

    # Ensure indices are within bounds
    if 0 <= row < rows and 0 <= col < cols:
        return data[row, col]
    else:
        return None  # Out of bounds


def show_camera_lines_of_sight(cameralist, gc_d_lines=None, show=True):
    """
    Plots diode lines of sight with plasma facing components.
    Takes a list of BolometerCamera objects.
    Also shows slit center locations.
    """
    fig, ax = plt.subplots(figsize=[4, 5])
    for camera in cameralist:
        for foil in camera.foil_detectors:
            # print(foil.slit.centre_point)
            slit_centre = foil.slit.centre_point
            slit_centre_rz = point3d_to_rz(slit_centre)
            ax.plot(slit_centre_rz[0], slit_centre_rz[1], "ko")
            origin, hit, _ = foil.trace_sightline()
            centre_rz = point3d_to_rz(foil.centre_point)
            ax.plot(centre_rz[0], centre_rz[1], "kx")
            origin_rz = point3d_to_rz(origin)
            hit_rz = point3d_to_rz(hit)
            ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], "r", lw=0.5)

    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax.plot(line[0], line[1], lw=0.5, c="k")
    ax.set_xlabel("R")
    ax.set_ylabel("z")
    ax.set_title("Diode lines of sight")
    ax.axis("equal")

    ax.set_xlim(0.95, 2.4)
    ax.set_ylim(-1.2, 1.2)
    if show:
        plt.show()
    else:
        return fig, ax


def show_camera_lines_of_sight_3D(cameralist):
    """
    Plots diode lines of sight with plasma facing components.
    Takes a list of BolometerCamera objects.
    Also shows slit center locations.
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for camera in cameralist:
        for foil in camera.foil_detectors:
            origin, hit, _ = foil.trace_sightline()
            # LOS
            ax.plot(
                [origin.x, hit.x], [origin.y, hit.y], [origin.z, hit.z], "r", lw=0.5
            )
            ax.plot(
                foil.slit.centre_point.x,
                foil.slit.centre_point.y,
                foil.slit.centre_point.z,
                "ko",
            )
            ax.plot(foil.centre_point.x, foil.centre_point.y, foil.centre_point.z, "kx")

    ax.axis("equal")
    ax.view_init(elev=30, azim=120)
    plt.show()


# ── Voxel-grid visualisation ──────────────────────────────────────────────────

def plot_voxel_data(
    ax, grid_centres, voxel_values, cmap="inferno", vmin=None, vmax=None, title=None
):
    """
    Renders a RayTransferGrid as a PatchCollection. Useful for both visualising
    the voxel grid and overlaying either the sensitivity matrix data or radiation data.

    :param ax:           Existing axis to plot on, or None to create a new one.
    :param grid_centres: (np.ndarray) – An array with shape (n_radius, n_height, 2)
                         containing the (R, Z) pairs of the grid centres.
                         Created by np.stack((cell_r_grid, cell_z_grid), axis=-1)
    :param voxel_values: 1-D array of values to colour by, or None to draw
                         empty cell outlines only.
    :param cmap:         Colormap to use for colouring the voxels.
    :param vmin:         Minimum value for the colorbar.
    :param vmax:         Maximum value for the colorbar.
    :param title:        Title for the plot.
    """
    # determine the R and Z resolution based on neighboring cells in grid_centres
    dR = abs(grid_centres[1, 0, 0] - grid_centres[0, 0, 0])
    dZ = abs(grid_centres[0, 1, 1] - grid_centres[0, 0, 1])

    patches = [
        Rectangle(
            (grid_centres[i, j, 0] - dR / 2,   # R_centre − half-width
             grid_centres[i, j, 1] - dZ / 2),  # Z_centre − half-height
            dR,   # width  in R
            dZ,   # height in Z
        )
        for i in range(grid_centres.shape[0])
        for j in range(grid_centres.shape[1])
    ]
    p = PatchCollection(patches, cmap=cmap)

    if voxel_values is None:
        p.set_edgecolor("black")
        p.set_facecolor("none")
    else:
        p.set_array(voxel_values)
        vmax = vmax if vmax is not None else max(voxel_values)
        vmin = vmin if vmin is not None else min(voxel_values)
        p.set_clim(vmin, vmax)

    if ax is None:
        _, ax = plt.subplots()
    ax.add_collection(p)
    ax.autoscale_view()
    ax.axis("equal")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_sensitivity_map(
    ax, grid_centres, voxel_map, sensitivity_1d,
    cmap="inferno", vmin=None, vmax=None
):
    """
    Plot a 1-D active-voxel sensitivity array on the full 2-D poloidal grid.
    Sensitivity is plotted with logarithmic colormap.

    :param ax:             Existing axis or None to create one.
    :param grid_centres:   (nx, ny, 2) – (R, Z) of each grid cell centre.
    :param voxel_map:      (nx, 1, ny) or (nx, ny) – 1-D voxel index per cell, -1 if inactive.
    :param sensitivity_1d: (num_cells,) – value for each active voxel (one diode, one bin).
    :param cmap:           Colormap name.
    :param vmin, vmax:     Colorbar limits.
    :returns:              (ax, im) – axis and AxesImage (for attaching a colorbar).
    """
    vm = voxel_map[:, 0, :] if voxel_map.ndim == 3 else voxel_map  # (nx, ny)
    nx, ny = vm.shape

    data_2d = np.full((nx, ny), np.nan)
    active = vm >= 0
    data_2d[active] = sensitivity_1d[vm[active]]

    dR = abs(grid_centres[1, 0, 0] - grid_centres[0, 0, 0])
    dZ = abs(grid_centres[0, 1, 1] - grid_centres[0, 0, 1])
    extent = (
        grid_centres[0, 0, 0] - dR / 2,
        grid_centres[-1, 0, 0] + dR / 2,
        grid_centres[0, 0, 1] - dZ / 2,
        grid_centres[0, -1, 1] + dZ / 2,
    )

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(
        data_2d.T,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=cmap_obj,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlim(POLOIDAL_RMIN - 0.1, POLOIDAL_RMAX + 0.1)
    ax.set_ylim(POLOIDAL_ZMIN - 0.1, POLOIDAL_ZMAX + 0.2)
    return ax, im


def plot_etendue(raytraced_etendue, raytraced_error, aug_etendue):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(raytraced_etendue, label="Raytraced")
    ax.errorbar(range(len(raytraced_etendue)), raytraced_etendue, yerr=raytraced_error, fmt='none', ecolor='gray', capsize=3)
    ax.plot(aug_etendue, label="AUG")
    ax.set_xlabel("Diode index")
    ax.set_ylabel(r"Etendue [m$^2$]")
    ax.legend()

    return fig, ax
