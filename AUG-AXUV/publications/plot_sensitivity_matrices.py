# %%
import h5py
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from axuv.plotting import plot_sensitivity_map, set_plt_rcparams
from axuv.io import PROJECT_ROOT, _GC_LINES_PATH
from axuv.cameras import WAVELENGTH_BIN_EDGES


project_dir = PROJECT_ROOT
data_dir = project_dir  / "axuv" / "data"

wavelength_centers = (WAVELENGTH_BIN_EDGES[:-1] + WAVELENGTH_BIN_EDGES[1:]) / 2

# Update default matplotlib parameters for LaTeX plotting for articles
set_plt_rcparams()
figsize = (5.5, 7)  # Since we use imshow, we need to set a figure size explicitly

with open(_GC_LINES_PATH, "rb") as fp:
    gc_d_lines = pickle.load(fp)

def plot_PFCs(ax, gc_d_lines, linewidth=0.5, color="white", linestyle="-"):
    for line in gc_d_lines:
        ax.plot(line[0], line[1], lw=linewidth, c=color, ls=linestyle)

RAYTRANSFER_PATHS = [
    data_dir / "raytransfer_S16_refl.h5",
    data_dir / "raytransfer_S5_refl.h5",
]

print(f"Project root: {project_dir}")
# %% Test plotting
raytransfer_path = RAYTRANSFER_PATHS[1]
try:
    with h5py.File(raytransfer_path, "r") as h5f:
        sensitivity_matrix = h5f["sensitivity_matrix"][()]
        grid_centres = h5f["grid_centres"][()]
        voxel_map = h5f["voxel_map"][()]
        inverse_voxel_map = h5f["inverse_voxel_map"][()]
        laplacian = h5f["laplacian"][()]
        mask = h5f["mask"][()]
        completed_bins = h5f["completed_bins"][()]
    print(f"Sensitivity matrix loaded from {raytransfer_path}")
    print(f"Number of diodes = {sensitivity_matrix.shape[0]}, number of active voxels = {sensitivity_matrix.shape[1]}, number of spectral bins = {sensitivity_matrix.shape[2]}")
    print(f"Resolution in R: {grid_centres.shape[0]}, Resolution in Z: {grid_centres.shape[1]}")
    reflections = False if "norefl" in str(raytransfer_path) else True
    # Create folder in project_dir/output/ named only the raytransfer filename without extension
    output_dir = Path(project_dir) / "output" / raytransfer_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Could not load sensitivity matrix from {raytransfer_path}: {e}")

# Fix the vmin and vmax values for one sensitivity dataset, so that the colorbars are indentical on all figures
vmax = sensitivity_matrix.max()
vmin = 1e-6 * vmax

for wavelength_bin in [0, 230]:

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor("black")
    ax, im = plot_sensitivity_map(
        ax, grid_centres, voxel_map, sensitivity_matrix[20, :, wavelength_bin], vmin=vmin, vmax=vmax
    )
    plot_PFCs(ax, gc_d_lines)
    energy = 1239.8 / wavelength_centers[wavelength_bin]
    plt.colorbar(im, ax=ax, label=r"Sensitivity [m$^2$ sr]")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"{wavelength_centers[wavelength_bin]:.2f} nm - {energy:.2f} eV")
    
    plt.savefig(output_dir / f"sensitivity_map_{wavelength_bin}.eps", format="eps", bbox_inches="tight")
    plt.savefig(output_dir / f"sensitivity_map_{wavelength_bin}.png", format="png", dpi=300, bbox_inches="tight")


# %% Load and plot raytransfer data
for raytransfer_path in RAYTRANSFER_PATHS:

    try:
        with h5py.File(raytransfer_path, "r") as h5f:
            sensitivity_matrix = h5f["sensitivity_matrix"][()]
            grid_centres = h5f["grid_centres"][()]
            voxel_map = h5f["voxel_map"][()]
            inverse_voxel_map = h5f["inverse_voxel_map"][()]
            laplacian = h5f["laplacian"][()]
            mask = h5f["mask"][()]
            completed_bins = h5f["completed_bins"][()]
        print(f"Sensitivity matrix loaded from {raytransfer_path}")
        print(f"Number of diodes = {sensitivity_matrix.shape[0]}, number of active voxels = {sensitivity_matrix.shape[1]}, number of spectral bins = {sensitivity_matrix.shape[2]}")
        print(f"Resolution in R: {grid_centres.shape[0]}, Resolution in Z: {grid_centres.shape[1]}")
        reflections = False if "norefl" in str(raytransfer_path) else True
        # Create folder in project_dir/output/ named only the raytransfer filename without extension
        output_dir = Path(project_dir) / "output" / raytransfer_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Could not load sensitivity matrix from {raytransfer_path}: {e}")

    # Fix the vmin and vmax values for one sensitivity dataset, so that the colorbars are indentical on all figures
    vmax = sensitivity_matrix.max()
    vmin = 1e-4 * vmax

    # Only plot in two chosen spectral bins
    spectral_bins = [0, sensitivity_matrix.shape[2] - 1]

    # Choose one diode to plot all the spectral bins
    diode_to_plot = 32

    for i in range(sensitivity_matrix.shape[0]):
        for j in spectral_bins:
            # create subfolder for the bin
            bin_output_dir = output_dir / f"bin_{j}"
            bin_output_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_facecolor("black")
            ax, im = plot_sensitivity_map(
                ax, grid_centres, voxel_map, sensitivity_matrix[i, :, j], vmin=vmin, vmax=vmax
            )
            plot_PFCs(ax, gc_d_lines)
            plt.colorbar(im, ax=ax, label=r"Sensitivity [m$^2$ sr]")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            fig.savefig(bin_output_dir / f"diode_{i}.png", dpi=300)
            plt.close(fig)
            print(f"Diode {i}, bin {j} done", end="\r")

        print(f"Diode {i} done", end="\r")

    # Create subfolder for spectral plotting
    spectral_output_dir = output_dir / "spectral"
    spectral_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot all bins for the chosen diode
    for j in range(sensitivity_matrix.shape[2]):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("black")
        ax, im = plot_sensitivity_map(
            ax, grid_centres, voxel_map, sensitivity_matrix[diode_to_plot, :, j], vmin=vmin, vmax=vmax
        )
        plot_PFCs(ax, gc_d_lines)
        plt.colorbar(im, ax=ax, label=r"Sensitivity [m$^2$ sr]")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

        fig.savefig(spectral_output_dir / f"bin_{j}.png", dpi=300)
        plt.close(fig)
        print(f"Diode {diode_to_plot}, bin {j} done", end="\r")
    print("\n")
