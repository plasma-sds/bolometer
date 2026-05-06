import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from axuv.geometry import point3d_to_rz


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


def show_camera_lines_of_sight(cameralist, gc_d_lines=None):
    """
    Plots diode lines of sight with plasma facing components.
    Takes a list of BolometerCamera objects.
    Also shows slit center locations.
    """
    _, ax = plt.subplots(figsize=[4, 5])
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
    plt.show()


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
def has_colorbar(fig) -> bool:
    """Returns True if the figure already contains a colorbar axes."""
    return any(ax.get_label() == "<colorbar>" for ax in fig.axes)


def plot_voxel_data(
    ax, voxels, voxel_values, cmap="inferno", vmin=None, vmax=None, title=None
):
    """
    Renders a ToroidalVoxelGrid as a PatchCollection.

    :param voxel_values: 1-D array of values to colour by, or None to draw
                         empty cell outlines only.
    """
    patches = [
        MplPolygon([(v.x, v.y) for v in voxel.vertices], closed=True)
        for voxel in voxels
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
    ax.set_xlim(voxels.min_radius, voxels.max_radius)
    ax.set_ylim(voxels.min_height, voxels.max_height)
    ax.axis("equal")
    if title is not None:
        ax.set_title(title)
    return ax


# --- Animation
def plot_init():
    """
    Creates a two-panel figure: large top panel for voxel emission,
    small bottom panel for the spectral responsivity curve.
    """
    fig = plt.figure(figsize=(4, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.9, 0.1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2])
    return fig, [ax1, ax2]


def plot_voxel_radiation(
    index,
    axlist,
    emission_data,
    voxel_grid,
    energies,
    gc_d_lines=None,
    bins_per_frame=10,
):
    """
    FuncAnimation callback: sums emission over `bins_per_frame` spectral bins
    and marks the corresponding energy range on the responsivity curve.

    All data is passed explicitly — no implicit global dependencies.
    """
    from axuv.responsivity import sensitivity_function

    ax1, ax2 = axlist
    ax1.clear()
    ax2.clear()

    lo = bins_per_frame * index
    hi = lo + bins_per_frame
    plot_voxel_data(
        ax=ax1,
        voxels=voxel_grid,
        voxel_values=np.sum(emission_data[lo:hi, :], axis=0),
        title="Emitted radiation\n(sum over spectral bin group)",
    )
    ax1.set_facecolor("k")
    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax1.plot(line[0], line[1], lw=0.5, c="white")
    ax1.set_xlim(1, 2.2)
    ax1.set_ylim(-1.2, 1)
    ax1.set_xlabel("R")
    ax1.set_ylabel("z")

    ax2.plot(energies, sensitivity_function(energies))
    ax2.axvline(energies[lo], color="red")
    ax2.axvline(energies[min(hi, len(energies) - 1)], color="red")
    ax2.set_xscale("log")
    ax2.set_xlim(0.9, 5000)
    ax2.grid()
    ax2.set_title("Spectral responsivity")
    ax2.set_xlabel("Photon energy [eV]")

    return [ax1, ax2]


def animate_voxel_emissions(
    emission_data,
    voxel_grid,
    energies,
    gc_d_lines=None,
    bins_per_frame=10,
    interval=500,
):
    """
    Creates an animation of voxel emission stepping through spectral bin groups.

    :param emission_data: ndarray (num_wl_bins, num_voxels) — note: rows = bins,
                          cols = voxels (as stored by calculate_emissions_for_raytransfer.py).
    :param voxel_grid:    ToroidalVoxelGrid object.
    :param energies:      1-D array of photon energies [eV] corresponding to rows.
    :param gc_d_lines:    optional plasma-facing component contours.
    :param bins_per_frame: number of wavelength bins summed per animation frame.
    :param interval:      frame delay in milliseconds.
    :returns:             HTML5 video string for display in Jupyter notebooks.
    """
    n_frames = emission_data.shape[0] // bins_per_frame
    fig, axlist = plot_init()

    ani = animation.FuncAnimation(
        fig,
        plot_voxel_radiation,
        frames=range(n_frames),
        interval=interval,
        blit=False,
        fargs=(axlist, emission_data, voxel_grid, energies, gc_d_lines, bins_per_frame),
    )
    return ani.to_html5_video()


def plot_etendue(raytraced_etendue, raytraced_error, aug_etendue):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(raytraced_etendue, label="Raytraced")
    ax.errorbar(range(len(raytraced_etendue)), raytraced_etendue, yerr=raytraced_error, fmt='none', ecolor='gray', capsize=3)
    ax.plot(aug_etendue, label="AUG")
    ax.set_xlabel("Diode index")
    ax.set_ylabel("Etendue")
    ax.legend()

    return fig, ax
