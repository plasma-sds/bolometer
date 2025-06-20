import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colorbar as cbar
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import h5py
import os
import pickle
from IPython.display import HTML
from cherab.tools.inversions import ToroidalVoxelGrid


plt.rcParams.update({'font.size': 16, "figure.dpi" : 150,
                     'figure.constrained_layout.use': True})

WORKDIR = os.path.dirname(os.path.realpath(__file__)) + "/"
BASEDIR = os.path.realpath(os.path.join(WORKDIR, "../../")) + "/"
SAVEDIR = WORKDIR + "data/"

SPECTRAL_BINS = 100
SPECTRAL_PARTS = 3

sensitivity = np.genfromtxt(SAVEDIR + "axuv_sensitivity.csv", delimiter=",")
degraded_sensitivity = np.genfromtxt(SAVEDIR + "degraded_avg.csv", delimiter=",")

measured_times = ["02410"]  #, "04090", "05150", "05970", ...
# If one has more timesteps, the time evolution of the measurements can be visualized.

times = np.zeros(len(measured_times))
for i, timestep in enumerate(measured_times):
    with h5py.File(SAVEDIR + "JOREK/jorekdata/jorek" + timestep + ".h5") as h5file:
        time = h5file["t_now"][()] * h5file["t_norm"][()]
        times[i] = np.round(time[0]*1000, 1)

with open(WORKDIR + "obj/gc_d_lines.obj", "rb") as fp:
    # in a list of 2-element (R, z) lists of arrays 
    gc_d_lines = pickle.load(fp)

def open_voxel_measurements(timestep):
    fname = "voxel_emissions_" + timestep + ".h5"
    with h5py.File(SAVEDIR + fname, "r") as file:
        emissions = file["emissions"][()]
        wavelength_bin_widths = file["wavelength_bin_widths"][()]
        denergies = 1239.8 / wavelength_bin_widths  # width of spectral bins in eV
        wavelengths = file["wavelengths"][()]
        energies = file["energies"][()]
        diode_names = file["diode_names"][()]
        diode_measurements = file["diode_measurements"][()]

    return wavelengths, energies, emissions, diode_names, diode_measurements, denergies

def load_voxels(name="voxel_grid.pickle"):
    try:
        with open(SAVEDIR + name, "rb") as f:
            grid_data = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            "Geometry data not found: please create the voxels first."
        )

    voxel_grid = ToroidalVoxelGrid(grid_data['voxel_data'])
    grid_laplacian = grid_data['laplacian']
    sensitivity_matrix = grid_data['sensitivity_matrix']
    print("Voxel data loaded.")

    return voxel_grid, grid_laplacian, sensitivity_matrix

def sensitivity_function(where):
    return np.interp(where, sensitivity[:, 0], sensitivity[:, 1])

def degraded_sensitivity_function(where):
    if isinstance(where, (list, np.ndarray)):
        toreturn = np.zeros(len(where))
        for index, loc in enumerate(where):
            if loc < 89.4308:
                toreturn[index] = np.interp(loc, sensitivity[:, 0], sensitivity[:, 1])
            else:
                toreturn[index] = np.interp(loc, degraded_sensitivity[:, 0], degraded_sensitivity[:, 1])
        return toreturn
    elif isinstance(where, (float, int)):
        if where < 89.4308:
            return np.interp(where, sensitivity[:, 0], sensitivity[:, 1])
        else:
            return np.interp(where, degraded_sensitivity[:, 0], degraded_sensitivity[:, 1])
    else:
        print("Unsupported type passed to degraded_sensitivity_function()")

def has_colorbar(fig):
    for ax in fig.axes:
        if isinstance(ax, cbar.Colorbar):
            return True
    return False
    
def plot_voxel_data(ax, voxels, voxel_values, cmap="inferno", vmin=None, vmax=None, title=None):
    patches = []
    for voxel in voxels:
        polygon = Polygon([(v.x, v.y) for v in voxel.vertices], closed=True)
        patches.append(polygon)

    p = PatchCollection(patches, cmap=cmap)
    if voxel_values is None:
        # Plot just the outlines of the grid cells
        p.set_edgecolor('black')
        p.set_facecolor('none')
    else:
        p.set_array(voxel_values)
        vmax = vmax or max(voxel_values)
        vmin = vmin or min(voxel_values)
        p.set_clim([vmin, vmax])

    if ax is None:
        _, ax = plt.subplots()
    ax.add_collection(p)
    ax.set_xlim(voxels.min_radius, voxels.max_radius)
    ax.set_ylim(voxels.min_height, voxels.max_height)
    ax.axis("equal")
    if title is not None:
        ax.set_title(title)
    return ax

def plot_init():
    fig = plt.figure(figsize=(4, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.9, 0.1, 1])

    # Create two axes
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2])

    axlist = [ax1, ax2]
    return fig, axlist

def plot_voxel_radiation(index, axlist, emission_data):
    print(index, end="\r")
    ax1, ax2 = axlist
    ax1.clear()
    ax2.clear()
    plot_voxel_data(ax=ax1, voxels=voxel_grid, 
                    voxel_values=np.sum(emission_data[10*index:10*index+10, :],
                                        axis=0), title="Emitted radiation by the"
                                        "\nvoxels in one spectral bin")

    ax1.set_facecolor('k')
    for line in gc_d_lines:
        ax1.plot(line[0], line[1], lw=.5, c="white")
    ax1.set_xlim(1, 2.2)
    ax1.set_ylim(-1.2, 1)
    ax1.set_xlabel('R')
    ax1.set_ylabel('z')

    ax2.plot(energies, sensitivity_function(energies))
    ax2.axvline(energies[10*index], color="red")
    ax2.axvline(energies[10*index+10], color="red")
    ax2.set_xscale("log")
    ax2.set_xlim(0.9, 5000)
    ax2.grid()
    ax2.set_title("Spectral responsivity")
    ax2.set_xlabel("Photon energy [eV]")
    
    axlist2 = [ax1, ax2]
    return axlist2

def animate_voxel_emissions(timestep):
    fig, axlist = plot_init()
    print(len(axlist))
    _, _, emissions, _, _, _ = open_voxel_measurements(timestep)

    ani = animation.FuncAnimation(fig, plot_voxel_radiation, init_func=plot_init, 
                                frames=range(29), interval=500, blit=False, fargs=(axlist, emissions))
    # writervideo = animation.FFMpegWriter(fps=5)
    video = ani.to_html5_video()

    return video

def get_weighted_power(diode_data, spectrum_energies, degraded=0):
    weighted_power = np.zeros(diode_data.shape[0])
    for i, energy in enumerate(spectrum_energies):
        try:
            dE_1 = np.abs(energy - spectrum_energies[i-1]) / 2
        except IndexError:
            dE_1 = np.abs(energy - spectrum_energies[i+1]) / 2

        try:
            dE_2 = np.abs(energy - spectrum_energies[i+1]) / 2
        except IndexError:
            dE_2 = np.abs(energy - spectrum_energies[i-1]) / 2

        # since spectrum_energies is decreasing in energy:
        averaging_range = np.linspace(energy+dE_1, energy-dE_2, 100)

        if degraded == 1:
            average_weight = np.average(degraded_sensitivity_function(averaging_range))
        elif degraded == 0:
            average_weight = np.average(sensitivity_function(averaging_range))

        weighted_power[:] += diode_data[:, i] * average_weight


    return weighted_power

def time_evolution(diode_first, diode_last, times, measured_times, noise=False, degraded=False):
    numofdiodes = diode_last-diode_first
    diode_data_evolution = np.zeros([numofdiodes, len(measured_times)])
    if noise:
        noise_array = np.random.normal(1.0, 0.1, numofdiodes)
    for i, time in enumerate(measured_times):
        _, energies, _, _, diode_measurements, _ = open_voxel_measurements(time)
        diode_data = diode_measurements[diode_first:diode_last]
        diode_data_evolution[:, i] = get_weighted_power(diode_data, energies, degraded=degraded)
        if noise:
            diode_data_evolution[:, i] *= noise_array

    fig, ax = plt.subplots(figsize=(8,4.5))
    pcm = ax.pcolormesh(times, range(numofdiodes), diode_data_evolution, norm="log", vmin=1e-5, cmap="inferno")
    return fig, ax, pcm, diode_data_evolution

voxel_grid, grid_laplacian, sensitivity_matrix = load_voxels(name="voxel_grid.pickle")

wavelengths, energies, emissions, diode_names, diode_measurements, denergies = open_voxel_measurements("02410")
