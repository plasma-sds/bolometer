"""
axuv/io.py  –  AUG shotfile I/O, data extraction and Gaussian signal smoothing
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

try:
    import aug_sfutils as sf
except ImportError:
    print("aug_sfutils not found. Some functionality will be limited.")

from axuv_measurement.config import SIGNAL_NAMES, SIG_KEY_LIST


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_AXUV_signals(shot: int, camera: str, tbeg: float=2.0, tend: float=2.5, gaussian_sigma: int=10,
                     offset_window: float | None = 1.0):
    """
    Goes through all signals from one AXUV :camera: in one :shot:
    Takes the shortened timesignal once, set by :tbeg: and :tend:
    Subtracts the dark offset of each channel, the mean of the calibrated signal over the
    last :offset_window: seconds of the shotfile timebase, which lies after the end of the plasma.
    None skips the subtraction.
    Takes the AXUV signals and smooths them with a gaussian filter with :gaussian_sigma:
    """
    if camera not in SIG_KEY_LIST:
        raise ValueError(f"'{camera}' is not a valid option. Allowed values: {SIG_KEY_LIST}")

    diag = SIGNAL_NAMES[camera][0]

    num_of_signals: int = len(SIGNAL_NAMES[camera][1])
    signal = sf.SFREAD(shot, diag, experiment='AUGD')
    time = signal.gettimebase(SIGNAL_NAMES[camera][1][0], tbeg=tbeg, tend=tend, cal=True)
    signallength = len(time)

    if not signallength:
        raise ValueError('signallength is not set')

    data_out = np.zeros((num_of_signals, signallength))

    if offset_window is not None:
        time_end = float(signal.gettimebase(SIGNAL_NAMES[camera][1][0])[-1])
        offset_tbeg = time_end - offset_window
        if offset_tbeg < tend:
            raise ValueError(
                f"Offset window {offset_tbeg:.3f}..{time_end:.3f} s overlaps the requested data "
                f"{tbeg:.3f}..{tend:.3f} s; it has to lie after the plasma."
            )

    for i, sig in enumerate(SIGNAL_NAMES[camera][1]):
        try:
            data_in = signal.getobject(sig, tbeg=tbeg, tend=tend, cal=True)
            if offset_window is not None:
                offset = np.mean(signal.getobject(sig, tbeg=offset_tbeg, tend=time_end, cal=True))
                data_in = data_in - offset
            data_out[i, :] = gaussian_filter1d(input=data_in, sigma=gaussian_sigma)
            print('OK', end="\r")

        except Exception as e:
            print(f"{e}")
            print(f'\033[31mWARNING! Could not get data for {sig}! Substituted by zeros\033[0m')
            data_out[i, :] = np.zeros([signallength])

    return data_out, time