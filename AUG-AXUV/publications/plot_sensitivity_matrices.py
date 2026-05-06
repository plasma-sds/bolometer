# %%
import h5py
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from axuv.plotting import plot_voxel_data, set_plt_rcparams
from axuv.io import PROJECT_ROOT, _GC_LINES_PATH


project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"

# Update default matplotlib parameters for LaTeX plotting for articles
set_plt_rcparams()

with open(_GC_LINES_PATH, "rb") as fp:
    gc_d_lines = pickle.load(fp)

print(f"Project root: {project_dir}")

# %% Load and plot raytransfer data
RAYTRANSFER_PATHS = [
    data_dir / "raytransfer_S16_norefl.h5",
    data_dir / "raytransfer_S05_norefl.h5",
    data_dir / "raytransfer_S16_refl.h5",
    data_dir / "raytransfer_S05_refl.h5",
]

raytransfer_path = RAYTRANSFER_PATHS[0]

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

dR = abs(grid_centres[10, 0, 0] - grid_centres[0, 0, 0]) / 10
dZ = abs(grid_centres[0, 10, 1] - grid_centres[0, 0, 1]) / 10
print(f"dR: {dR:.2e}, dZ: {dZ:.2e}")
print(gc_d_lines[0])

# %%
# Loop through the diodes and if the raytransfer file contains reflections loop through the spectral bins
# and plot the sensitivity matrix for each diode and spectral bin
for i in range(sensitivity_matrix.shape[0]):
    if not reflections:
        # Plot the sensitivity matrix for each diode as PatchCollection using voxel_map and sensitivity_matrix
        fig, ax = plt.subplots()
        # ax.set_facecolor("black")
        for line in gc_d_lines:
            ax.add_line(line)  #, color="white", linewidth=.5)
        plot_voxel_data(ax, grid_centres, sensitivity_matrix[i, :, 0])
        fig.savefig(output_dir / f"diode_{i}.png", dpi=300)

    # elif reflections:
    #     # Create subdirectories for each diode
    #     diode_output_dir = output_dir / f"diode_{i}"
    #     diode_output_dir.mkdir(parents=True, exist_ok=True)
    #     for j in range(sensitivity_matrix.shape[2]):
    #         # Plot the sensitivity matrix for each spectral bin and save to diode_output_dir
    #         fig, ax = plt.subplots()
    #         ax.set_facecolor("black")
    #         for line in gc_d_lines:
    #             ax.plot(line, color="white", linewidth=.5)
    #         plot_voxel_data(ax, grid_centres, sensitivity_matrix[i, :, j])
    #         fig.savefig(diode_output_dir / f"bin_{j}.png", dpi=300)
