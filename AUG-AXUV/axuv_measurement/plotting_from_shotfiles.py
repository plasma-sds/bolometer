"""
IPython Notebook for plotting AXUV signals from shotfiles
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

from axuv_measurement.config import (
    D16, DHT, DVC, DHC
)
from axuv_measurement.io import get_AXUV_signals

latex = True

if latex:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.family": "serif",  # tells matplotlib to use \rmfamily in the LaTeX doc
    })
else:
    plt.rcParams.update({"text.usetex": False})

plt.close('all')
plt.rcParams.update({'font.size': 16,
                     "figure.dpi" : 300,
                     'figure.constrained_layout.use': True,
                     'image.cmap': 'inferno'})


project_dir = "/tokp/work/lefer/AUG-AXUV"

def plot_one_camera(shotno, data, time, camera, project_dir, vmin=1e4, vmax=1e8, save=False, remove_offset=True):
    """
    Plots the AXUV signal data as pcolormesh for one camera in one shot.
    Optionally removes the starting time offset from the time array to start from 0.

    Parameters
    ----------
    shotno : int
        Shot number.
    data : np.ndarray
        AXUV signal data.
    time : np.ndarray
        Time array.
    camera : str
        Camera name.
    project_dir : str
        Project directory.
    vmin : float, optional
        Minimum value for color scale.
    vmax : float, optional
        Maximum value for color scale.
    save : bool, optional
        Whether to save the figure.
    remove_offset : bool, optional
        Whether to remove the starting time offset from the time array.
    """
    if remove_offset:
        time -= time[0]

    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('k')
    pcm = ax.pcolormesh(time * 1e3, np.arange(0, 48, 1), data, norm="log", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm)


    cbar.set_label(r'Line integrated brightness [W/m$^2$]')
    plt.ylabel('Diode index')

    ax.set_xlabel("Time [ms]")

    if save:
        plt.savefig(project_dir + "/output/" + str(shotno) + "/" + camera + "_notitle.png")

    title_suffix = camera.replace("_", " ")
    if "vert" in title_suffix:
        title_suffix = title_suffix.replace("vert", "vertical")
    elif "horiz" in title_suffix:
        title_suffix = title_suffix.replace("horiz", "horizontal")

    title = str(shotno) + " " + title_suffix
    plt.title(title)
    if save:
        plt.savefig(project_dir + "/output/" + str(shotno) + "/" + camera + ".png")
    else:
        plt.show()

def save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, project_dir, vmin=1e4, vmax=1e8, save=False):
    plot_one_camera(shotno, vert_data, vert_time, vert_camera, project_dir, vmin=vmin, vmax=vmax, save=save)
    plot_one_camera(shotno, horiz_data, horiz_time, horiz_camera, project_dir, vmin=vmin, vmax=vmax, save=save)


save_figures = True
# %%
shotno = 40673  # S16
starttime = 2.3276
endtime = starttime + 1.9e-3
vert_camera = D16
horiz_camera = DHT

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)

save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, project_dir, save=save_figures)

# %%
shotno = 40673  # S05
starttime = 2.3276
endtime = starttime + 1.9e-3
vert_camera = DVC
horiz_camera = DHC

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, project_dir, save=save_figures)

# %%
shotno = 41007  # S16
starttime = 2.3417
endtime = starttime + 5.2e-3
vert_camera = D16
horiz_camera = DHT

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, project_dir, save=save_figures)

# %%
shotno = 41007  # S05
starttime = 2.3417
endtime = starttime + 5.2e-3
vert_camera = DVC
horiz_camera = DHC

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, project_dir, save=save_figures)
