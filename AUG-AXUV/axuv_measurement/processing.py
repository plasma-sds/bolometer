"""
axuv/processing.py  –  signal downsampling, interpolation, repair, and the
                        Sato ridge-filter pipeline.
"""
import bisect
import numpy as np
import skimage.filters

from .config import NEWAXUV, DHT
from .io import read_data_base


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def find_roots(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the x-locations where *y* crosses zero (linear interpolation)."""
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def upsample_interpolate(templist: list, timesignals: list):
    """Upsample the shorter of two 2-D data arrays to match the longer one.

    Used by ``plot_diff()`` to align data from two diagnostics that may have
    different time resolutions before computing the difference.

    :param templist:    2-element list of (N_channels × T) arrays
    :param timesignals: 2-element list of the corresponding time arrays
    :returns:           ``(tobesubtracted, timesignal)`` where *tobesubtracted*
                        is the pair of aligned arrays and *timesignal* belongs
                        to whichever array has the higher time resolution.
    """
    dims1 = templist[0].shape
    dims2 = templist[1].shape
    tobesubtracted = []

    if dims1[1] < dims2[1]:
        # Upsample array 0 to match array 1
        tempdata = np.zeros([dims2[0], dims2[1]])
        for row in range(dims1[0]):
            tempdata[row, :] = np.interp(
                np.linspace(0, 10, dims2[1]),
                np.linspace(0, 10, dims1[1]),
                templist[0][row, :],
            )
        tobesubtracted.append(tempdata)
        tobesubtracted.append(templist[1])
        timesignal = timesignals[1]

    elif dims2[1] < dims1[1]:
        # Upsample array 1 to match array 0
        tempdata = np.zeros([dims1[0], dims1[1]])
        for row in range(dims1[0]):
            tempdata[row, :] = np.interp(
                np.linspace(0, 10, dims1[1]),
                np.linspace(0, 10, dims2[1]),
                templist[1][row, :],
            )
        tobesubtracted.append(templist[0])
        tobesubtracted.append(tempdata)
        timesignal = timesignals[0]

    else:
        tobesubtracted.append(templist[0])
        tobesubtracted.append(templist[1])
        timesignal = timesignals[0]

    return tobesubtracted, timesignal


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def downsample(shotno, diagname: str,
               start: float = 2.31, end: float = 2.32,
               signalrange: tuple = (0, 47)):
    """Downsample time signal and channel data to a uniform 1×10⁻⁵ s resolution.

    The downsampling factor differs by diagnostic so that all cameras end up
    on the same time grid after this step.

    :param shotno:      AUG discharge number (int or str)
    :param diagname:    custom diagnostic name (see axuv/config.py)
    :param start:       start of the time window [s]
    :param end:         end of the time window   [s]
    :param signalrange: (first_channel_index, last_channel_index), inclusive
    :returns:           ``(downsampled_data, downsampled_time)``
                        data shape is (T_down, N_channels)
    """
    # Factor chosen so that all diagnostics reach the same 1e-5 s resolution
    factor = 10 if any(x == diagname for x in NEWAXUV) else 4

    shotno = str(shotno)
    signalnames, timesignal, indices, _, f = read_data_base(
        diagname, start=start, end=end, shotno=shotno
    )

    newend = timesignal.shape[0] - (timesignal.shape[0] % factor)
    timesignal = timesignal[:newend]
    downsampled_time = timesignal.reshape(-1, factor).mean(axis=1)

    numofsignals = signalrange[1] - signalrange[0] + 1
    downsampled_data = np.zeros((downsampled_time.shape[0], numofsignals))

    for i, channel in enumerate(range(signalrange[0], signalrange[1])):
        tempdata = f['/'.join([shotno, diagname, signalnames[channel]])][()][indices[0]:indices[1]]
        downsampled_data[:, i] = tempdata[:newend].reshape(-1, factor).mean(axis=1)

    return downsampled_data, downsampled_time


# ---------------------------------------------------------------------------
# Signal repair
# ---------------------------------------------------------------------------

def repair_2d_data(data: np.ndarray, add_indices=None) -> np.ndarray:
    """Linearly interpolate dead / missing channels in a 2-D data array.

    A channel is considered missing when all its time-series values are below
    1×10⁴.  Missing channels are replaced by the average of their immediate
    neighbours; edge channels copy the single available neighbour.

    .. warning::
        Will not work correctly if the *first* channels are missing and their
        two neighbours are also missing.

    :param data:        (T × N_channels) array, modified in-place and returned
    :param add_indices: optional list of additional channel indices to repair
                        regardless of the automatic detection threshold
    :returns:           the repaired array (same object as *data*)
    """
    booleans  = ~np.all(data < 1e4, axis=0)
    indices   = list(np.where(~booleans)[0])
    if add_indices is not None:
        indices.extend(add_indices)

    for i in indices:
        if i - 1 < 0:
            data[:, i] = data[:, i + 1]
        elif (i + 1) > (data.shape[1] - 1) or np.all(data[:, i + 1] < 1e4):
            data[:, i] = data[:, i - 1]
        else:
            data[:, i] = (data[:, i - 1] + data[:, i + 1]) / 2

    return data


# ---------------------------------------------------------------------------
# Ridge filter pipeline
# ---------------------------------------------------------------------------

def ridge_filter(shotno, diagname: str,
                 start: float = 2.31, end: float = 2.32,
                 signalrange: tuple = (0, 47),
                 sigmas: list = None,
                 normalize: bool = True,
                 plot: bool = False,
                 repair: bool = True,
                 vmin: float = 1e4):
    """Downsample, optionally repair, then apply the Sato ridge-following algorithm.

    Returns the filtered (and optionally normalised) data together with the
    downsampled time array and the raw downsampled data before filtering.

    :param shotno:      AUG discharge number (int or str)
    :param diagname:    custom diagnostic name (see axuv/config.py)
    :param start:       start of the time window [s]
    :param end:         end of the time window   [s]
    :param signalrange: (first_channel_index, last_channel_index), inclusive
    :param sigmas:      list of scales for the Sato filter (default [1, 2]);
                        larger values smooth and spread the result more
    :param normalize:   if True, normalise the output so that each timestep's
                        L² norm equals 1; if False, normalise to the global max
    :param plot:        if True, show a side-by-side comparison of filtered vs
                        raw data
    :param repair:      if True, interpolate missing channels before filtering;
                        if False, strip them out, filter, then re-insert zeros
    :param vmin:        lower colour-map bound for the raw-data panel when
                        *plot* is True
    :returns:           ``(data, downsampled_time, raw_data)``
    """
    if sigmas is None:
        sigmas = [1, 2]

    downsampled_data, downsampled_time = downsample(
        shotno, diagname, start=start, end=end, signalrange=signalrange
    )

    # Detect low-signal (dead) channels
    booleans = ~np.all(downsampled_data < 1e4, axis=0)
    bad_indices = np.where(~booleans)

    add_indices = None
    if diagname == DHT:
        add_indices = [16]

    if repair:
        tobefiltered = repair_2d_data(downsampled_data, add_indices=add_indices)
    else:
        tobefiltered = downsampled_data[:, booleans]

    # Sato ridge-following algorithm
    array = skimage.filters.sato(tobefiltered, sigmas=sigmas, black_ridges=False)

    # Normalise
    if normalize:
        row_norms = np.linalg.norm(array, axis=1)
        data = array / row_norms[:, np.newaxis]
    else:
        data = array / np.max(array)

    # Re-insert zeroed columns for the stripped channels when repair=False
    if not repair:
        for index in bad_indices[0]:
            data = np.insert(data, index, 0, axis=1)

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=[10, 3])
        axes[0].pcolormesh(
            downsampled_time, range(downsampled_data.shape[1]), data.T,
            shading='nearest', norm='linear',
        )
        axes[1].pcolormesh(
            downsampled_time, range(downsampled_data.shape[1]), downsampled_data.T,
            shading='nearest', norm='log', vmin=vmin, vmax=1e8,
        )
        plt.show()

    return data, downsampled_time, tobefiltered