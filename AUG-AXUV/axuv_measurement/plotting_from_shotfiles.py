"""
IPython Notebook for plotting AXUV signals from shotfiles
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

from axuv_measurement.config import (
    D16, DHT, DVC, DHC
)
from axuv_measurement.io import get_AXUV_signals, PROJECT_ROOT
from axuv_measurement.plotting import set_plt_rcparams


set_plt_rcparams()
project_dir = PROJECT_ROOT

def plot_one_camera(shotno, data, time, camera, project_dir, vmin=1e4, vmax=1e8, save=False, remove_offset=False):
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('k')
    pcm = ax.pcolormesh(time, np.arange(1, 49, 1), data, norm="log", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm)


    cbar.set_label(r"$I$ [W/m²]")
    plt.ylabel('Diode number')

    ax.set_xlabel("Time [s]")
   
    if save:
        savename = str(shotno) + "_" + camera + "_notitle.png"
        plt.savefig(project_dir / "output" / str(shotno) / savename, dpi=300, bbox_inches="tight")

    # Add vertical lines for shot 40673 and 41007 corresponding to the 80% and 20% thermal energy
    # in the AUG SPI experiments
    firstcolor = "cyan"
    secondcolor = "lime"
    labelsize = 16
    savename = str(shotno) + "_" + camera + "_with_lines.png"
    
    if str(shotno) == "40673":
        t1, t2 = 2.32805, 2.3288
        ax.axvline(t1, ls=":", color=firstcolor)
        ax.axvline(t2, ls=":", color=secondcolor)
        ax.set_xticks(np.arange(2.328, 2.3295, 0.0005))

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks([t1, t2])
        ax_top.set_xticklabels([r"80\% $W_{\mathrm{th}}$", r"20\% $W_{\mathrm{th}}$"])
        ax_top.tick_params(direction='out', length=5, colors='black', labelsize=labelsize)
        ax_top.spines['top'].set_visible(False)

        fig.savefig(project_dir / "output" / str(shotno) / savename, dpi=300, bbox_inches="tight")

    elif str(shotno) == "41007":
        t1, t2 = 2.3428, 2.3445
        ax.axvline(t1, ls=":", color=firstcolor)
        ax.axvline(t2, ls=":", color=secondcolor)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks([t1, t2])
        ax_top.set_xticklabels([r"80\% $W_{\mathrm{th}}$", r"20\% $W_{\mathrm{th}}$"])
        ax_top.tick_params(direction='out', length=5, colors='black', labelsize=labelsize)
        ax_top.spines['top'].set_visible(False)

        fig.savefig(project_dir / "output" / str(shotno) / savename, dpi=300, bbox_inches="tight")
    

    title_suffix = camera.replace("_", " ")
    if "vert" in title_suffix:
        title_suffix = title_suffix.replace("vert", "vertical")
    elif "horiz" in title_suffix:
        title_suffix = title_suffix.replace("horiz", "horizontal")

    title = str(shotno) + " " + title_suffix
    plt.title(title)
    if save:
        savename = str(shotno) + "_" + camera + ".png"
        plt.savefig(project_dir / "output" / str(shotno) / savename, dpi=300, bbox_inches="tight")
    
    plt.show()
    plt.close(fig)

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
