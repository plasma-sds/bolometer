"""
axuv/animation.py  –  animated poloidal-scatter and line-plot animations.
"""
import bisect
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aug_sfutils as sf
from scipy.interpolate import griddata

from .config import ELLIPSE, NEWAXUV, ROOTFOLDER, D16, DHT
from .geometry import (
    _load_isect_matrix, _filter_by_polygon,
    plot_qsurfaces, get_intersecting_LOSs,
)
from .processing import repair_2d_data


# ---------------------------------------------------------------------------
# Scatter animation (raw or ridge-filter data on poloidal cross-section)
# ---------------------------------------------------------------------------

def animate_poloidal(data3d, timesignal, shotno, sector,
                     maximum: bool = True, step: int = 1,
                     multiplier: int = 20,
                     save: bool = False, given_name=None):
    """Animate scatter plots of the intersection data on the poloidal cross-section.

    The intersection matrix is loaded once and shared across all frames.

    :param data3d:      (T × 48 × 48) array from ``ridge_filter`` or
                        ``generate_isect_data3d``
    :param timesignal:  time array corresponding to the first axis of *data3d*
    :param shotno:      AUG discharge number (int or str)
    :param sector:      tokamak sector (5 or 16)
    :param maximum:     if True, mark the interpolated maximum location each frame
    :param step:        number of timesteps to advance per animation frame
    :param multiplier:  scatter dot-size scaling factor
    :param save:        if True, write an MP4 file
    :param given_name:  path prefix for the output file; auto-generated if None
    :returns:           the FuncAnimation object
    """
    imat = _load_isect_matrix(sector)
    key  = 'S' + str(sector)

    fig, ax = plt.subplots(dpi=150, figsize=(4, 6))
    fig.set_facecolor('black')
    ax.set_aspect(1)
    ax.set_facecolor('black')

    # White axes styling for the dark background
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='both', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    timeindex = 0
    time      = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res = _filter_by_polygon(isecs, ELLIPSE)

    grid_R, grid_z = np.meshgrid(
        np.linspace(1, 2.2, 1200), np.linspace(-1, 1.1, 2200), indexing='ij'
    )

    Rloc = Zloc = None
    if maximum:
        points      = res[:, :2]
        values      = res[:, 2]
        grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')
        valueindex  = np.unravel_index(np.nanargmax(grid_interp), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        Zloc = grid_z[0, valueindex[1]]

    lines = []
    gc_d  = sf.getgc()
    for gc in gc_d.values():
        structure = ax.plot(gc.r, gc.z, lw=0.5, color='white')
        lines.append(structure)

    _, equ = plot_qsurfaces(shotno, timesignal[-1], axes=ax)

    spi1 = ax.plot([1.5, 2.275708], [-0.153213,  0.308526], lw=0.5, ls='--', c='grey', label='SPI1')
    spi2 = ax.plot([1.5, 2.295673], [-0.071579,  0.324483], lw=0.5, ls='-',  c='grey', label='SPI2')
    spi3 = ax.plot([1.5, 2.271826], [ 0.0975929, 0.361589], lw=0.5, ls='-.', c='grey', label='SPI3')
    lines.extend([spi1, spi2, spi3])

    scatter = plt.scatter(res[:, 0], res[:, 1],
                          s=res[:, 2] * multiplier, c=res[:, 2], cmap='inferno')
    if maximum:
        ax.plot(Rloc, Zloc, '1', mew=2, ms=20, c='lime')

    plt.xlim(1, 2.4)
    plt.xlabel('R [m]')
    plt.ylim(-1.3, 1.3)
    plt.ylabel('z [m]')
    plt.title('#{} S-{} @ {:.5f} s'.format(shotno, sector, time))

    numframes = len(timesignal) // step
    anim = animation.FuncAnimation(
        fig, update_animate_poloidal,
        frames=range(numframes),
        fargs=(imat, data3d, timesignal, shotno, sector, lines, equ,
               maximum, grid_R, grid_z, ax, ELLIPSE, step, gc_d, multiplier),
        blit=False,
    )

    if save:
        fps = 2 if step > 2 else 5
        writer = animation.FFMpegWriter(fps=fps)
        if given_name is None:
            given_name = 'plots/{}/{}'.format(shotno, key)
        anim.save(given_name + '_' + str(shotno) + '.mp4', writer=writer)
        plt.close()

    return anim


def update_animate_poloidal(i, imat, data3d, timesignal, shotno, sector,
                             lines, equ, maximum, grid_R, grid_z,
                             axis, ellipse, step, gc_d, multiplier=20):
    """Per-frame update callback for :func:`animate_poloidal`."""
    timeindex = i * step
    time      = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res = _filter_by_polygon(isecs, ellipse)
    axis.clear()

    for line in lines:
        axis.add_line(line[0])

    plot_qsurfaces(equ, time, axes=axis)

    Rloc = Zloc = None
    if maximum:
        points      = res[:, :2]
        values      = res[:, 2]
        grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')
        valueindex  = np.unravel_index(np.nanargmax(grid_interp), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        Zloc = grid_z[0, valueindex[1]]

    scatter = axis.scatter(res[:, 0], res[:, 1],
                           s=res[:, 2] * multiplier, c=res[:, 2],
                           cmap='inferno', zorder=10)
    if maximum:
        axis.plot(Rloc, Zloc, '1', mew=2, ms=20, c='lime')

    axis.set_title('#{} S-{} @ {:.5f} s'.format(shotno, sector, time), color='white')
    axis.set_xlabel('R [m]', color='white')
    axis.set_ylabel('z [m]', color='white')
    axis.set_xlim(1, 2.4)
    axis.set_ylim(-1.3, 1.3)
    return scatter,


# ---------------------------------------------------------------------------
# Interpolated pcolormesh animation
# ---------------------------------------------------------------------------

def interp_anim(data3d, timesignal, shotno, sector, multiplier=20, step=1):
    """Animate the griddata-interpolated radiation map on the poloidal plane.

    The intersection matrix is loaded once here and passed to every frame via
    ``fargs`` – no file I/O occurs inside the per-frame update callback.

    :param data3d:     (T × 48 × 48) array
    :param timesignal: time array corresponding to the first axis of *data3d*
    :param shotno:     AUG discharge number (int or str)
    :param sector:     tokamak sector (5 or 16)
    :param multiplier: value scaling factor applied before interpolation
    :param step:       timestep stride between animation frames
    :returns:          the FuncAnimation object
    """
    # Load the intersection matrix ONCE here; pass as farg to avoid per-frame I/O
    imat = _load_isect_matrix(sector)
    key  = 'S' + str(sector)

    fig, ax = plt.subplots(dpi=150, figsize=(4, 6))
    fig.set_facecolor('black')
    ax.set_aspect(1)
    ax.set_facecolor('black')

    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(axis='both', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    timeindex = 0
    time      = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :] * multiplier
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res = _filter_by_polygon(isecs, ELLIPSE)

    R_axis = np.linspace(1, 2.2, 600)
    z_axis = np.linspace(-1, 1.1, 1100)
    grid_R, grid_z = np.meshgrid(R_axis, z_axis, indexing='ij')

    points      = res[:, :2]
    values      = res[:, 2]
    grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')

    plt.pcolormesh(R_axis, z_axis, grid_interp.T)
    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    plt.title('# {} S-{} @ {:.4f} s'.format(shotno, sector, time))

    numframes = len(timesignal) // step
    anim = animation.FuncAnimation(
        fig, update_interp_anim,
        frames=range(numframes),
        fargs=(imat, data3d, timesignal, shotno, sector,
               grid_R, grid_z, ax, ELLIPSE, step, multiplier, R_axis, z_axis),
        blit=False,
    )

    return anim


def update_interp_anim(i, imat, data3d, timesignal, shotno, sector,
                       grid_R, grid_z, axis, ellipse, step, multiplier,
                       R_axis, z_axis):
    """Per-frame update callback for :func:`interp_anim`.

    Receives *imat* via ``fargs``; no file I/O is performed here.
    """
    timeindex = i * step
    time      = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :] * multiplier
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res = _filter_by_polygon(isecs, ellipse)

    points      = res[:, :2]
    values      = res[:, 2]
    grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')

    axis.clear()
    pcm = axis.pcolormesh(R_axis, z_axis, grid_interp.T)
    axis.set_title('# {} S-{} @ {:.4f} s'.format(shotno, sector, time), color='white')
    axis.set_xlabel('R [m]', color='white')
    axis.set_ylabel('z [m]', color='white')
    axis.set_xlim(1, 2.4)
    axis.set_ylim(-1.3, 1.3)
    return pcm,


# ---------------------------------------------------------------------------
# Channel line-plot animation with q=2 / LCFS markers
# ---------------------------------------------------------------------------

def animate_with_q2(shotno, diagname, start, end,
                    steps: int = 5, scale: str = 'linear',
                    diag2=None, repair: bool = False):
    """Animate per-channel line plots with q=2 and LCFS boundary markers.

    For a single diagnostic (*diag2* is None) one panel is shown; passing
    *diag2* creates a two-panel side-by-side layout.

    :param shotno:   AUG discharge number (int or str)
    :param diagname: primary custom diagnostic name
    :param start:    start of the time window [s]
    :param end:      end of the time window   [s]
    :param steps:    number of timesteps between animation frames
    :param scale:    y-axis scale: ``'linear'`` or ``'log'``
    :param diag2:    optional second diagnostic for the right panel
    :param repair:   if True, run :func:`~axuv.processing.repair_2d_data` on
                     both data arrays before animating
    :returns:        the FuncAnimation object
    """
    # FIX: coerce to str before string concatenation
    shotno = str(shotno)
    when   = start

    filename = ROOTFOLDER + 'smoothed_data/' + shotno + '_sm.h5'
    f = h5py.File(filename, 'r')

    signalnames = list(f['/'.join([shotno, diagname])].keys())
    equ, firstsignal, lastsignal, sep1, sep2 = get_intersecting_LOSs(
        shotno, when, diag1=diagname, diag2=None, plot=False, equ=None
    )

    timesignalname = next((s for s in signalnames if '_time' in s), None)
    timesignal     = f['/'.join([shotno, diagname, timesignalname])][()]
    signalnames.pop(signalnames.index(timesignalname))

    firstch = signalnames.index(firstsignal + '_data') + 0.5
    lastch  = signalnames.index(lastsignal  + '_data') + 1.5
    sepch1  = signalnames.index(sep1        + '_data') + 0.5
    sepch2  = signalnames.index(sep2        + '_data') + 1.5

    numofsignals = len(signalnames)
    ti_start     = bisect.bisect_left(timesignal, start)
    ti_end       = bisect.bisect_left(timesignal, end) + 1
    timesignal   = timesignal[ti_start:ti_end]
    lenofsignals = ti_end - ti_start

    data2d = np.zeros([numofsignals, lenofsignals])
    for i in range(numofsignals):
        data2d[i] = f['/'.join([shotno, diagname, signalnames[i]])][()][ti_start:ti_end]

    if repair:
        data2d = repair_2d_data(data2d)

    # Optional second diagnostic
    equ2 = signalnames2 = diag2_name = data2d2 = None
    numofsignals2 = 0
    firstch2 = lastch2 = sepch21 = sepch22 = None

    if diag2 is not None:
        signalnames2 = list(f['/'.join([shotno, diag2])].keys())
        equ2, firstsignal2, lastsignal2, sep21, sep22 = get_intersecting_LOSs(
            shotno, when, diag1=diag2, diag2=None, plot=False, equ=None
        )
        timesignalname2 = next((s for s in signalnames2 if '_time' in s), None)
        signalnames2.pop(signalnames2.index(timesignalname2))
        numofsignals2 = len(signalnames2)

        data2d2 = np.zeros([numofsignals2, lenofsignals])
        for i in range(numofsignals2):
            data2d2[i] = f['/'.join([shotno, diag2, signalnames2[i]])][()][ti_start:ti_end]

        if repair:
            data2d2 = repair_2d_data(data2d2)

        firstch2 = signalnames2.index(firstsignal2 + '_data') + 0.5
        lastch2  = signalnames2.index(lastsignal2  + '_data') + 1.5
        sepch21  = signalnames2.index(sep21        + '_data') + 0.5
        sepch22  = signalnames2.index(sep22        + '_data') + 1.5
        diag2_name = diag2

    f.close()

    # --- Build figure --------------------------------------------------------
    if diag2 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=[6, 4])
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])
        # FIX: was data2d.T[0] with numofsignals; should use data2d2 and numofsignals2
        l21 = ax2.step(range(1, numofsignals2 + 1), data2d2.T[0], where='mid')
        l22 = ax2.axvline(firstch2, c='orange', lw=0.5, ls='--')
        l23 = ax2.axvline(lastch2,  c='orange', lw=0.5, ls='--')
        l24 = ax2.axvline(sepch21,  c='blue',   lw=0.5, ls='--')
        l25 = ax2.axvline(sepch22,  c='blue',   lw=0.5, ls='--')
        ax2.set_title('{}'.format(diag2.split('_')[0]))
        ax2.set_xlim(1, numofsignals2 + 1)
        ax2.set_ylim(0, 1.05 * data2d2.max())

    l1 = ax1.step(range(1, numofsignals + 1), data2d.T[0], where='mid')
    l2 = ax1.axvline(firstch, c='orange', lw=0.5, ls='--', label='"q=2"')
    l3 = ax1.axvline(lastch,  c='orange', lw=0.5, ls='--')
    l4 = ax1.axvline(sepch1,  c='blue',   lw=0.5, ls='--', label='"LCFS"')
    l5 = ax1.axvline(sepch2,  c='blue',   lw=0.5, ls='--')

    other = [firstch, lastch, sepch1, sepch2]
    lines = [l1] if diag2 is None else [l1, l2, l3, l4, l5, l21, l22, l23, l24, l25]

    title = ax1.set_title('#{} {} @ {:.6f} s'.format(
        shotno, diagname.split('_')[0], when))
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, numofsignals + 1)
    ax1.set_ylim(0, 1.05 * data2d.max())

    axes_arg = (ax1, ax2) if diag2 is not None else ax1

    anim = animation.FuncAnimation(
        fig, update_with_q2,
        frames=range(int(np.floor(lenofsignals / steps))),
        fargs=(fig, shotno, data2d, diagname, equ, timesignal,
               signalnames, axes_arg, steps, scale,
               equ2, signalnames2, diag2_name, data2d2,
               title, lines, other, numofsignals2),
        blit=False,
    )

    plt.close(fig)
    return anim


def update_with_q2(index, fig, shotno, data2d, diagname, equ, timesignal,
                   signalnames, axes, steps, scale,
                   equ2=None, signalnames2=None, diag2=None, data2d2=None,
                   title=None, lines=None, other=None, numofsignals2=0):
    """Per-frame update callback for :func:`animate_with_q2`."""
    index = index * steps

    if not isinstance(axes, tuple):
        # Single-diagnostic layout
        axes.clear()
        numofsignals = len(signalnames)
        axes.set_xlim(1, numofsignals + 1)

        if scale == 'linear':
            axes.set_ylim(0, 1.05 * data2d.max())
        elif scale == 'log':
            axes.set_ylim(1e6, 1e8)
            axes.set_yscale('log')

        axes.step(range(1, numofsignals + 1), data2d.T[index], where='mid')
        axes.axvline(other[0], c='orange', lw=0.5, ls='--', label='"q=2"')
        axes.axvline(other[1], c='orange', lw=0.5, ls='--')
        axes.axvline(other[2], c='blue',   lw=0.5, ls='--', label='"LCFS"')
        axes.axvline(other[3], c='blue',   lw=0.5, ls='--')
        axes.set_title('#{} {} @ {:.6f} s'.format(
            shotno, diagname.split('_')[0], timesignal[index]))
        axes.legend(loc='upper right')

    else:
        ax1, ax2 = axes
        numofsignals  = len(signalnames)
        # FIX: use numofsignals2 (passed via fargs) for the second panel
        ax1.set_xlim(1, numofsignals  + 1)
        ax2.set_xlim(1, numofsignals2 + 1)

        if scale == 'linear':
            ax1.set_ylim(0, 1.05 * data2d.max())
            ax2.set_ylim(0, 1.05 * data2d2.max())
        elif scale == 'log':
            for ax in (ax1, ax2):
                ax.set_ylim(1e6, 1e8)
                ax.set_yscale('log')

        lines[0][0].set_data(range(1, numofsignals  + 1), data2d.T[index])
        # FIX: was range(1, numofsignals + 1) – correct range for second panel
        lines[5][0].set_data(range(1, numofsignals2 + 1), data2d2.T[index])
        title.set_text('#{} {} @ {:.6f} s'.format(
            shotno, diagname.split('_')[0], timesignal[index]))
```Now let me finish with the package `__init__.py` and the updated `axuv_scripts.py`: