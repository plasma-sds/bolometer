"""
axuv/plotting.py  –  most of the plotting functions, radiation mapping, and
                     LOS-sum helpers.
"""
import h5py
import bisect
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

try:
    import aug_sfutils as sf
except ImportError:
    print("aug_sfutils not found. Some functionality will be limited.")

from .config import (
    ROOTFOLDER, VERT, D16, DVC, DHT, DHC,
    ELLIPSE,
    VERT_S5_RANGE, HORIZ_S5_RANGE, VERT_S16_RANGE, HORIZ_S16_RANGE,
)

from .geometry import (
    plot_qsurfaces, generate_isect_data3d,
    _load_isect_matrix, _filter_by_polygon,
    get_intersecting_LOSs,
)
from .processing import ridge_filter, find_roots


def set_plt_rcparams():
    """Sets the default matplotlib rcParams for LaTeX plotting in articles."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "font.family": "serif",  # tells matplotlib to use \rmfamily in the LaTeX doc
        "font.size": 16,
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        "image.cmap": 'inferno',
        "lines.linewidth": 2,
    })



# ---------------------------------------------------------------------------
# 1-D line plots
# ---------------------------------------------------------------------------

def plot_1D(shotno, diagname, channel, start=2.3, end=2.4):
    """Plot one channel (given as a 1-based index) as a function of time."""
    shotno = str(shotno)
    signalnames, timesignal, indices, _, f = read_data_base(
        diagname, start=start, end=end, shotno=shotno
    )
    datatoplot = f['/'.join([shotno, diagname, signalnames[channel - 1]])][()][indices[0]:indices[1]]

    plt.figure(figsize=[8, 5])
    plt.plot(timesignal, datatoplot)
    plt.gca().set_ylim(None, None)
    plt.xlabel('Time [s]')
    plt.title(diagname + ' channel #' + str(channel) + '\n' + signalnames[channel - 1])
    plt.show()
    f.close()


def plot_full_integral(shotno, diagname, yscale='linear', start=2.3, end=2.4):
    """Plot the sum of all channel signals of *diagname* as a function of time."""
    shotno = str(shotno)
    signalnames, timesignal, indices, lenofsignals, f = read_data_base(
        diagname, start=start, end=end, shotno=shotno
    )

    datatoplot = np.zeros(lenofsignals)
    for signal in signalnames:
        datatoplot += f['/'.join([shotno, diagname, signal])][()][indices[0]:indices[1]]
    f.close()

    plt.figure(figsize=[8, 5])
    plt.plot(timesignal, datatoplot)
    ax = plt.gca()
    ax.set_yscale(yscale)
    if yscale == 'log':
        ax.set_ylim(1e6, None)
    plt.xlabel('Time [s]')
    plt.ylabel('Integrated AXUV signal – all channels')
    plt.title('#' + shotno + ' ' + diagname.split('_')[0])
    plt.show()


def plot_integral_combined(shotno, array=None, yscale='linear',
                           start=2.3, end=2.4, axes=None, savename=None):
    """Plot the integrated channel sum for each diagnostic in *array* on one axes.

    :param shotno:   AUG discharge number (int or str)
    :param array:    list of diagnostic names; defaults to VERT
    :param yscale:   ``'linear'`` or ``'log'``
    :param start:    start of the time window [s]
    :param end:      end of the time window   [s]
    :param axes:     matplotlib Axes to plot on; if None a new figure is created
    :param savename: path to save the figure; if None and axes is None, plt.show()
                     is called instead.  Has no effect when *axes* is provided.
    """
    if array is None:
        array = VERT

    shotno = str(shotno)
    # Track whether this call owns the figure so we know whether to show/save
    created_fig = axes is None
    if created_fig:
        fig = plt.figure(figsize=[8, 5])
        axes = plt.gca()

    for diagname in array:
        signalnames, timesignal, indices, lenofsignals, f = read_data_base(
            diagname, start=start, end=end, shotno=shotno
        )
        datatoplot = np.zeros(lenofsignals)
        for signal in signalnames:
            datatoplot += f['/'.join([shotno, diagname, signal])][()][indices[0]:indices[1]]
        f.close()
        axes.plot(timesignal, datatoplot, label=diagname.split('_')[0])

    axes.set_yscale(yscale)
    if yscale == 'log':
        axes.set_ylim(1e6, None)
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Integrated AXUV signals – all channels')
    axes.legend()

    # Only save / show when this call created the figure
    if created_fig:
        if savename is not None:
            plt.savefig(savename, dpi=200)
            plt.close()
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Plasma current
# ---------------------------------------------------------------------------

def plot_current(shotno, start=2.3, end=2.4,
                 topgrad=None, tdip=None, tpeak=None,
                 plot=True, savename=None, title=None):
    """Plot the plasma current and its derivative; optionally locate or mark the
    current dip and peak times.

    Return behaviour depends on the arguments:

    * If *topgrad* is given **or** both *tdip* and *tpeak* are given:
        returns ``(diptime, peaktime)`` as floats.
    * Otherwise:
        returns ``(timesignal, datatoplot, derivative)`` as NumPy arrays.

    :param topgrad: time [s] near the steepest gradient; the nearest zero
                    crossings of the derivative on either side are used as
                    dip and peak times.
    :param tdip:    manually specify the current dip time [s]
    :param tpeak:   manually specify the current peak time [s]
    """
    shotno = str(shotno)
    fpc        = sf.SFREAD(int(shotno), 'FPC', experiment='AUGD')
    Ip         = fpc.getobject("IpiFP", cal=True)
    time_full  = fpc.gettimebase("IpiFP")
    indices    = calculate_indices(time_full, start, end)
    timesignal = time_full[indices[0]:indices[1]]
    datatoplot = Ip[indices[0]:indices[1]] / 1000
    derivative = np.gradient(datatoplot)

    if plot:
        fig = plt.figure(figsize=[8, 3])
        plt.plot(timesignal, datatoplot, color='black', zorder=4)
        plt.grid(axis='x', which='both')
        ax = plt.gca()
        ax.set_xlabel('Time [s]')

        if savename is None:
            ax2 = ax.twinx()
            ax2.plot(timesignal, derivative, color='red', zorder=5)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=0)
            ax2.set_ylabel(r'dI$_{p}$/dt [kA/s]', color='red')

    if topgrad is not None:
        roots = find_roots(timesignal, derivative)
        try:
            peaktime = roots[roots > topgrad].min()
            diptime  = roots[roots < topgrad].max()
        except Exception:
            print("No root found!")
            peaktime = start
            diptime  = end

        if plot:
            plt.axvline(x=diptime,  color='green', linestyle='--',
                        label='Dip: {:.8f} s'.format(diptime),  linewidth=1, zorder=2)
            plt.axvline(x=peaktime, color='blue',  linestyle='--',
                        label='Peak: {:.8f} s'.format(peaktime), linewidth=1, zorder=3)
            print('Current dip at {:.8f} s and peak at {:.8f} s'.format(diptime, peaktime))
            plt.legend()

    if tdip is not None and tpeak is not None:
        peaktime = tpeak
        diptime  = tdip
        if plot:
            plt.axvline(x=diptime,  color='green', linestyle='--',
                        label='Dip: {:.6f} s'.format(diptime),  linewidth=1, zorder=2)
            plt.axvline(x=peaktime, color='blue',  linestyle='--',
                        label='Peak: {:.6f} s'.format(peaktime), linewidth=1, zorder=3)
            print('Current dip and peak manually set')
            plt.legend()

    if plot:
        ax.set_ylim(None, None)
        plt.xlabel('Time [s]')
        ax.set_ylabel('Plasma current [kA]', color='black')
        plt.title(('#' + shotno + ' ' + title) if title else '#' + shotno)
        if savename is not None:
            plt.savefig(savename + '.png', dpi=150)
            plt.close(fig)
        else:
            plt.show()

    if topgrad is not None or (tdip is not None and tpeak is not None):
        return diptime, peaktime
    else:
        return timesignal, datatoplot, derivative


# ---------------------------------------------------------------------------
# Time-slice and downsampled plots
# ---------------------------------------------------------------------------

def plot_timeslice(shotno, diagname, when=2.32, signalrange=(0, 47)):
    """Plot channel signals as a line plot at a single time *when* [s]."""
    shotno   = str(shotno)
    filename = ROOTFOLDER + 'smoothed_data/' + shotno + '_sm.h5'

    with h5py.File(filename, 'r') as f:
        signalnames = list(f['/'.join([shotno, diagname])].keys())

        timesignalname = next((s for s in signalnames if '_time' in s), None)
        timesignal = f['/'.join([shotno, diagname, timesignalname])][()]
        signalnames.pop(signalnames.index(timesignalname))

        numofsignals = signalrange[1] - signalrange[0] + 1
        t_index = bisect.bisect_left(timesignal, when)
        data1d = np.zeros(numofsignals)
        for i in range(signalrange[0], signalrange[1] + 1):
            data1d[i] = f['/'.join([shotno, diagname, signalnames[i]])][()][t_index]

    plt.figure(figsize=[6, 4])
    plt.step(range(signalrange[0] + 1, signalrange[1] + 2), data1d, where='mid')
    plt.show()


# ---------------------------------------------------------------------------
# Poloidal radiation mapping
# ---------------------------------------------------------------------------

def radiation_poloidal(data3d, timesignal, time, shotno, sector,
                       multiplier=20, polygon=ELLIPSE,
                       interp=None, ax=None, plot=True,
                       save=False, given_name=None):
    """Interpolate and/or plot data from a (T × 48 × 48) intersection array.

    The function slices *data3d* at the requested *time*, filters the
    intersection points by *polygon*, and optionally interpolates them onto a
    regular grid.

    :param data3d:      (T × 48 × 48) array from ``ridge_filter`` or
                        ``generate_isect_data3d``
    :param timesignal:  time array corresponding to the first axis of *data3d*
    :param time:        time at which to evaluate [s]
    :param shotno:      AUG discharge number (int or str)
    :param sector:      tokamak sector (5 or 16)
    :param multiplier:  scatter dot-size scaling factor; pass ``'raw'`` for a
                        log-scaled size based on signal magnitude
    :param polygon:     shapely Polygon used as the spatial mask
    :param interp:      griddata method string (e.g. ``'linear'``), or None to
                        skip interpolation
    :param ax:          matplotlib Axes to draw on; a new figure is created when
                        both *ax* is None and *plot* is True
    :param plot:        if True, render the scatter plot
    :param save:        if True, save the figure to disk
    :param given_name:  filename prefix when saving

    :returns: ``(grid_interp, points, values, Rloc, zloc)``
    """
    imat      = _load_isect_matrix(sector)
    key       = 'S' + str(sector)

    if ax is None and plot:
        fig, ax = plt.subplots(dpi=150, figsize=(5, 6))
        fig.set_facecolor('black')

    timeindex = bisect.bisect_left(timesignal, time)

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    if polygon is None:
        return None, [0, 0], 0, None, None

    res    = _filter_by_polygon(isecs, polygon)
    points = res[:, :2]
    values = res[:, 2]

    if interp is not None:
        grid_R, grid_z = np.meshgrid(
            np.linspace(1, 2.2, 600), np.linspace(-1, 1.1, 1100), indexing='ij'
        )
        grid_interp = griddata(points, values, (grid_R, grid_z), method=interp)
        valueindex  = np.unravel_index(np.nanargmax(grid_interp), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        zloc = grid_z[0, valueindex[1]]
    else:
        grid_interp = None
        maxrowindex = np.argmax(res[:, 2])
        Rloc = res[maxrowindex, 0]
        zloc = res[maxrowindex, 1]

    if ax is not None or plot:
        ax.set_aspect(1)
        gc_d = sf.getgc()
        for gc in gc_d.values():
            ax.plot(gc.r, gc.z, lw=0.5, color='k')

        plot_qsurfaces(shotno, time, axes=ax)
        ax.plot([1.5, 2.275708], [-0.153213,  0.308526], lw=0.5, ls='--', c='grey', label='SPI1')
        ax.plot([1.5, 2.295673], [-0.071579,  0.324483], lw=0.5, ls='-',  c='grey', label='SPI2')
        ax.plot([1.5, 2.271826], [ 0.0975929, 0.361589], lw=0.5, ls='-.', c='grey', label='SPI3')

        if multiplier == 'raw':
            scale_array = np.zeros(len(res[:, 2]))
            for i in range(len(scale_array)):
                if res[i, 2] is None or res[i, 2] < 1e11:
                    scale_array[i] = 0
                else:
                    scale_array[i] = ((np.log10(res[i, 2]) - 11) ** 3) * 4
            scale_array[scale_array < 1.5] = 0
            res[:, 2] = res[:, 2] / 1e11
            scatter = ax.scatter(res[:, 0], res[:, 1],
                                 s=scale_array, c=scale_array,
                                 cmap='inferno_r', zorder=10)
        else:
            scatter = ax.scatter(res[:, 0], res[:, 1],
                                 s=res[:, 2] * multiplier, c=res[:, 2],
                                 cmap='inferno_r', zorder=10)

        ax.plot(Rloc, zloc, '1', mew=2, ms=20, c='lime')
        ax.set_xlim(1, 2.6)
        ax.set_xlabel('R [m]')
        ax.set_ylim(-1.3, 1.3)
        ax.set_ylabel('z [m]')
        ax.set_title('#{} S-{} @ {:.6f} s'.format(shotno, sector, time))
        ax.legend(facecolor='white', markerscale=10, fontsize='small',
                  labelcolor='k', framealpha=1, loc='lower right')

        if plot:
            plt.colorbar(scatter, anchor=(0.0, 0.75), shrink=0.5)
            if save:
                savename_full = '_'.join([given_name, key, str(shotno)])
                plt.savefig(savename_full + '.png', dpi=150)
                plt.close(fig)
            else:
                plt.show()

    return grid_interp, points, values, Rloc, zloc


def radiation_inside_q2surface(shotno, start, end, sector=16, use_sqrt=False):
    """Compute the integrated intersection signal inside q=2 and inside the LCFS.

    The two original functions ``radiation_inside_q2surface`` and
    ``radiation_inside_q2surface_2`` have been merged here.  Pass
    ``use_sqrt=True`` to take the element-wise square root of the
    intersection product before integrating (former ``_2`` behaviour).

    :param shotno:    AUG discharge number (int or str)
    :param start:     start of the time window [s]
    :param end:       end of the time window   [s]
    :param sector:    tokamak sector (5 or 16)
    :param use_sqrt:  if True, apply ``np.sqrt`` to the product array
    :returns:         ``(q2sum, totalsum, timesignal)``
    """
    # Defaults (sector 16); overridden below for sector 5
    vertdiagname  = D16
    horizdiagname = DHT
    vertrange     = VERT_S16_RANGE
    horizrange    = HORIZ_S16_RANGE

    if sector == 5:
        vertdiagname  = DVC
        horizdiagname = DHC
        vertrange     = VERT_S5_RANGE
        horizrange    = HORIZ_S5_RANGE
    elif sector == 16:
        pass  # already set above

    _, _, datavert  = ridge_filter(shotno, vertdiagname,  start=start, end=end, signalrange=vertrange)
    _, timesignalh, datahoriz = ridge_filter(shotno, horizdiagname, start=start, end=end,
                                             signalrange=horizrange)

    raw = generate_isect_data3d(vertrange, horizrange, datavert, datahoriz)
    multiplied = np.sqrt(raw) if use_sqrt else raw

    q2poly, sep_poly, equ = plot_qsurfaces(shotno, start, axes=None, diag="EQH")

    q2sum    = np.zeros(len(timesignalh))
    totalsum = np.zeros(len(timesignalh))

    for i, thistime in enumerate(timesignalh):
        q2poly, sep_poly, _ = plot_qsurfaces(equ, thistime, axes=None)
        _, _, valuesq2, _, _    = radiation_poloidal(multiplied, timesignalh, time=thistime,
                                                     shotno=shotno, sector=sector,
                                                     save=False, plot=False,
                                                     interp=None, polygon=q2poly)
        _, _, valuestotal, _, _ = radiation_poloidal(multiplied, timesignalh, time=thistime,
                                                     shotno=shotno, sector=sector,
                                                     save=False, plot=False,
                                                     interp=None, polygon=sep_poly)
        q2sum[i]    = np.sum(valuesq2)
        totalsum[i] = np.sum(valuestotal)

    return q2sum, totalsum, timesignalh


# ---------------------------------------------------------------------------
# Overview and summary plots
# ---------------------------------------------------------------------------

def plot_overview_1(shotno, start=2.3, end=2.4, vmin=1e4, vmax=1e8,
                    save=False, vtimes=None, title=None, sector=16, plot=False):
    """Plot (optionally) the raw 2-D camera data, plasma current, integrated
    vertical signals, and the radiation fraction inside q=2; always saves the
    q=2 integral data to an HDF5 file.

    :param shotno:  AUG discharge number (int or str)
    :param start:   start of the time window [s]
    :param end:     end of the time window   [s]
    :param vmin:    lower colour bound for the pcolormesh
    :param vmax:    upper colour bound for the pcolormesh
    :param save:    if True, save the overview figure to disk
    :param vtimes:  2-element tuple ``(diptime, peaktime)``
    :param title:   optional figure title override
    :param sector:  tokamak sector (5 or 16)
    :param plot:    if True, produce the full overview figure
    """
    shotno = str(shotno)
    diag   = D16 if sector == 16 else DVC

    signalnames, timesignal, indices, lenofsignals, f = read_data_base(
        diag, start=start, end=end, shotno=shotno
    )
    numofsignals = len(signalnames)
    tickrange    = np.arange(2, 49, 2)
    plotrange    = range(1, 49)
    data2d       = np.zeros([numofsignals, lenofsignals])

    for i in range(numofsignals):
        data2d[i, :] = f['/'.join([shotno, diag, signalnames[i]])][()][indices[0]:indices[1]]
        print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')

    if plot:
        fig     = plt.figure(figsize=[16, 9])
        if title is not None:
            fig.suptitle(title)

        subfigs      = fig.subfigures(1, 2, wspace=0.05)
        subfigsnest  = subfigs[0].subfigures(2, 1, height_ratios=[1, 0.4])
        axsnest0     = subfigsnest[0].subplots(1, 1)

        im   = axsnest0.pcolormesh(timesignal, plotrange, data2d,
                                   norm='log', vmin=vmin, vmax=vmax, zorder=1)
        cbar = plt.colorbar(im, ax=axsnest0)
        cbar.set_label('Diode signals [a. u.]')
        axsnest0.set_facecolor('black')
        axsnest0.set_yticks(tickrange)
        axsnest0.xaxis.grid(True, linestyle='--', zorder=5)
        axsnest0.set_ylabel('Channel number')

        subfigsnest[1].suptitle('Plasma current [kA]')
        axsnest1 = subfigsnest[1].subplots(1, 1)
        axsRight = subfigs[1].subplots(2, 1)

        fpc          = sf.SFREAD(int(shotno), 'FPC', experiment='AUGD')
        Ip           = fpc.getobject("IpiFP", cal=True)
        Ip_timefull  = fpc.gettimebase("IpiFP")
        Ip_indices   = calculate_indices(Ip_timefull, start, end)
        Ip_time      = Ip_timefull[Ip_indices[0]:Ip_indices[1]]
        Ip_data      = Ip[Ip_indices[0]:Ip_indices[1]] / 1000

        axsnest1.plot(Ip_time, Ip_data)

        if vtimes is not None:
            diptime, peaktime = vtimes
            for axis in [axsnest0, axsnest1, axsRight[0], axsRight[1]]:
                axis.axvline(x=diptime,  color='green', linestyle='--',
                             label='Dip: {:.6f} s'.format(diptime),  linewidth=1, zorder=10)
                axis.axvline(x=peaktime, color='blue',  linestyle='--',
                             label='Peak: {:.6f} s'.format(peaktime), linewidth=1, zorder=11)
            axsnest1.legend(loc='lower left')

        axsnest1.grid()
        axsnest1.set_xlabel('Time [s]')

        plot_integral_combined(shotno, array=VERT, yscale='linear',
                               start=start, end=end, axes=axsRight[0])

    q2sum, totalsum, ts = radiation_inside_q2surface(
        shotno=shotno, start=start, end=end, sector=sector
    )

    fw = h5py.File('data_export/' + shotno + '_' + str(sector) + '_q2sum.h5', 'w')
    fw.create_dataset('q2sum',  data=q2sum)
    fw.create_dataset('total',  data=totalsum)
    fw.create_dataset('time',   data=ts)
    fw.close()

    if plot:
        axsRight[1].plot(ts, q2sum,    label='q2')
        axsRight[1].plot(ts, totalsum, label='tot')
        axsRight[1].legend(loc='upper left')
        axsRight[1].set_ylabel('Multiplied raw signals')
        ax2 = axsRight[1].twinx()
        ax2.plot(ts, q2sum / totalsum, c='green')
        ax2.set_ylabel('inside q=2 / total', c='green')
        ax2.set_ylim(0, 1)
        axsRight[1].set_xlim(ts[0], ts[-1])

        if save:
            plt.savefig('spec_plots/overviews/' + shotno + '.png', dpi=300)
            plt.close(fig)
            print('Saved ' + shotno)
        else:
            plt.show()

    f.close()


def save_sqrt(shotno, start=2.3, end=2.4, sector=16):
    """Compute the sqrt-weighted q=2 integral and save it to an HDF5 file.

    This is identical to ``plot_overview_1`` with ``use_sqrt=True`` in the
    radiation integral, but without producing any plot.
    """
    shotno = str(shotno)

    q2sum, totalsum, ts = radiation_inside_q2surface(
        shotno=shotno, start=start, end=end, sector=sector, use_sqrt=True
    )

    fw = h5py.File('data_export/' + shotno + '_' + str(sector) + '_q2sum_sqrt.h5', 'w')
    fw.create_dataset('q2sum',  data=q2sum)
    fw.create_dataset('total',  data=totalsum)
    fw.create_dataset('time',   data=ts)
    fw.close()


# ---------------------------------------------------------------------------
# LOS-sum helpers
# ---------------------------------------------------------------------------

def sum_LOSs(shotno, start, end, diagname):
    """Sum signal values for the LOSs crossing the q=2 surface and the LCFS.

    The crossing LOSs are determined once at *start* using the equilibrium and
    held fixed for the whole time window (see commented-out loop below for a
    time-varying version).

    :returns: ``(edge_rad, core_rad, timesignal)``
    """
    shotno   = str(shotno)
    filename = ROOTFOLDER + 'smoothed_data/' + shotno + '_sm.h5'

    f           = h5py.File(filename, 'r')
    signalnames = list(f['/'.join([shotno, diagname])].keys())

    _, crq2, crsep = get_intersecting_LOSs(shotno, start, diag1=diagname, get_data=True)

    timesignalname = next((s for s in signalnames if '_time' in s), None)
    timesignal     = f['/'.join([shotno, diagname, timesignalname])][()]
    signalnames.pop(signalnames.index(timesignalname))

    ti_start    = bisect.bisect_left(timesignal, start)
    ti_end      = bisect.bisect_left(timesignal, end) + 1
    timesignal  = timesignal[ti_start:ti_end]
    lenofsignals = ti_end - ti_start

    core_rad = np.zeros(lenofsignals)
    edge_rad = np.zeros(lenofsignals)

    print(lenofsignals)

    for signal in crsep:
        edge_rad += f['/'.join([shotno, diagname, signal.signalname + '_data'])][()][ti_start:ti_end]
    for signal in crq2:
        core_rad += f['/'.join([shotno, diagname, signal.signalname + '_data'])][()][ti_start:ti_end]

    # Time-varying version (slow – equilibrium queried at every step):
    # for i in range(lenofsignals):
    #     _, crq2, crsep = get_intersecting_LOSs(shotno, timesignal[i], diag1=diagname,
    #                                             get_data=True, equ=equ)
    #     for signal in crsep:
    #         edge_rad[i] += f[...][()][i + ti_start]
    #     for signal in crq2:
    #         core_rad[i] += f[...][()][i + ti_start]
    #     print(str(i + 1) + ' step  ', end='\r')

    f.close()
    return edge_rad, core_rad, timesignal


def save_LOS_sums(shotno, start, end, diptime, peaktime, title, save=True):
    """Compute and plot/save LOS sums for D16 and DHT diagnostics.

    For each diagnostic, two time traces are shown (edge and core LOSs) plus
    their cumulative sums on a twin axis.  Data are also written to HDF5.
    """
    f = h5py.File('data_export/LOS_sums/' + str(shotno) + '.h5', 'w')

    for diag in [D16, DHT]:
        diag_short = diag.split('_')[0]
        edge_rad, core_rad, timesignal = sum_LOSs(shotno, start, end, diag)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        ax.plot(timesignal, edge_rad, label='Sum of edge LOSs')
        ax.plot(timesignal, core_rad, label='Sum of core LOSs')
        ax.axvline(diptime,  ls='--', lw=1, c='green', label='Current dip')
        ax.axvline(peaktime, ls='--', lw=1, c='blue',  label='Current peak')
        ax.set_ylabel('Sum of LOS values in each timestep [a. u.]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper left')
        ax.set_ylim(0, None)
        ax.set_xlim(timesignal[0], None)
        ax.grid()

        ax2 = ax.twinx()
        ax2.set_title(title + ' ' + diag_short)
        ax2.set_xlim(timesignal[0], None)
        ax2.set_ylabel('Cumulative sum of LOS values [a. u.]')
        ax2.plot(timesignal, np.cumsum(edge_rad), label='Cumulative sum – edge', c='green')
        ax2.plot(timesignal, np.cumsum(core_rad), label='Cumulative sum – core', c='red')
        ax2.legend(loc='center left')
        ax2.set_ylim(0, None)

        if save:
            plt.savefig('spec_plots/LOS sums/' + str(shotno) + '_' + diag_short + '.png', dpi=150)
            plt.close()
        else:
            plt.show()

        f.create_dataset(diag_short + '_core', data=core_rad)
        f.create_dataset(diag_short + '_edge', data=edge_rad)
        f.create_dataset(diag_short + '_time', data=timesignal)

    f.close()
