"""
axuv/io.py  –  HDF5 I/O, data extraction utilities, and calibration routines.

NOTE: axuv_to_hdf() and calibrate_and_smooth() are due for a logic rewrite
"""
import os
import h5py
import bisect
import numpy as np
import aug_sfutils as sf
from scipy.ndimage import gaussian_filter1d

from .config import ROOTFOLDER, SIGNAL_NAMES, SIG_KEY_LIST, NEWAXUV


# ---------------------------------------------------------------------------
# Verbatim – do not change
# ---------------------------------------------------------------------------

def axuv_to_hdf(shot, filewrite=None, starttime=2):
    """Extracts AXUV data from AUG shotfiles into custom HDF5 files
    Creates the following HDF5 hierarchy from the AUG shotfiles:
    <shotnumber>/<diagnostic_name>/<signalname>_<"data" or "time">

    :param shot: int or str, mandatory
        AUG discharge number
    :param filewrite: str, optional, default=None
        filename for the output
        if None, the naming is automatic
    :param starttime: float, optional, default=2
        from which point in time to save the data
        SPI usually triggers around 2.3 seconds
    """
    shot = str(shot)
    if filewrite is None:
        f = h5py.File(ROOTFOLDER + 'data_export/' + shot + '_AXUV.h5', 'w')
    else:
        f = h5py.File(ROOTFOLDER + filewrite, 'w')

    # Load equilibrium to determine the end of plasma time
    equ = sf.EQU(int(shot))
    # Add 1.5 second to be able to adjust the zero-level of the signals when there is no plasma
    endtime = equ.time[-1] + 1.5 

    # Go through all signals from all diagnostics, take the shortened timesignal once
    # Save timesignal and all data signals to HDF5 starting from :starttime:
    for key in SIG_KEY_LIST:
        diag = SIGNAL_NAMES[key][0]
        time_taken = False
        signal = sf.SFREAD(int(shot), diag, experiment='AUGD')
        for sig in SIGNAL_NAMES[key][1]:
            if time_taken == False:
                timefull = signal.gettimebase(sig)
                # get the indices corresponding to the requested start and end times
                endindex = bisect.bisect_left(timefull, endtime)
                startindex = bisect.bisect_left(timefull, starttime)
                time = timefull[startindex:endindex]
                time_taken = True
                signallength = len(time)
                hierarchy_t = '/'.join([shot, key, sig + '_time']) 
                try:
                    f.create_dataset(hierarchy_t, data=time)
                except:
                    print('\033[31mWARNING! Could not create dataset for ' + hierarchy_t + '\033[0m')

            if key in NEWAXUV:
                try:
                    datafull = signal.getobject(sig, cal=False)
                    data = datafull[startindex:endindex]
                except:
                    print('\033[31mWARNING! Could not get data! Substituted by zeros\033[0m')
                    data = np.zeros([signallength])
            else:
                try:
                    datafull = signal.getobject(sig, cal=True)
                    data = datafull[startindex:endindex]
                except:
                    print('\033[31mWARNING! Uncalibrated data!\033[0m')
                    try:
                        datafull = signal.getobject(sig)
                        data = datafull[startindex:endindex]
                    except:
                        print('\033[31mWARNING! Could not get data! Substituted by zeros\033[0m')
                        data = np.zeros([signallength])

            hierarchy_d = '/'.join([shot, key, sig + '_data']) 
            try:
                f.create_dataset(hierarchy_d, data=data)
                print('\033[32mCreated dataset for ' + hierarchy_d + '\033[0m')

            except:
                f.create_dataset(hierarchy_d, data=np.zeros([signallength]))
                print('\033[31mWARNING! Could not create dataset for ' + hierarchy_d 
                      + '\033[0m\nValues have been substituted by zeros')

    f.close()


def calibrate_and_smooth(shotno: int | str, fileread=None, filewrite=None, smooth: bool=True, sigma: int=1):
    """Moves all signals by an offset based on averaging when there is no plasma
    Calibrates gains based on the CSV gains files generated from #40989 flat-top
    Smooths both the 'new' and the 'old' signals with Gaussian smoothing
    Cuts off 1 second at the end then saves the smoothed data and the timesignals into a new HDF5 file

    Can be used from #40342 until #41307

    :param shotno: int or str, mandatory
        AUG discharge number
    :param fileread: str, optional
        filename to serve as input
    :param filewrite: str, optional
        filename for the output
    :param smooth: bool, optional
        if True, smoothing is done via Gaussian filter with :sigma:
    :param sigma:
    """

    if type(shotno) is not str:
        shotno: str = str(shotno)

    if fileread is None:
        fileread: str = 'data_export/' + shotno + '_AXUV.h5'
    if filewrite is None and smooth:
        filewrite: str = 'smoothed_data/' + shotno + '_sm.h5'

    with h5py.File(fileread, 'r') as fr, h5py.File(filewrite, 'w') as fw:
        print('\nDischarge #' + shotno + '\n---- READING SHOTFILES AND CALIBRATING ----')
        diagnamelist = list(fr[shotno].keys())
        for j, diagname in enumerate(diagnamelist):
            if diagname in NEWAXUV:
                gains = np.genfromtxt('calibration/' + diagname + '.csv', delimiter=',')

            signalnames = list(fr['/'.join([shotno, diagname])].keys())
            
            for i in range(len(signalnames)):
                if '_time' in signalnames[i]:
                    index = i
                    timesignalname = signalnames[i]
                    break

            hierarchy_t = '/'.join([shotno, diagname, timesignalname])
            timefull = fr[hierarchy_t][()]
            # Here is the cutting of the last second - we do not need data from there anymore
            endindex = bisect.bisect_left(timefull, timefull[-1] - 1)
            time = timefull[:endindex]
            fw.create_dataset(hierarchy_t, data=time)
            signalnames.pop(index)
            numofsignals = len(signalnames)
            print('Working on {} camera data'.format(diagname))
            
            for i, signal in enumerate(signalnames):
                hierarchy = '/'.join([shotno, diagname, signal])
                tempdata = fr[hierarchy][()][endindex:]
                offset = np.average(tempdata)
                data = fr[hierarchy][()][:endindex] - offset

                if smooth:
                    if diagname in NEWAXUV:
                        data = gaussian_filter1d(data * gains[i], sigma=1)
                    else:
                        data = gaussian_filter1d(data, sigma=1)

                fw.create_dataset(hierarchy, data=data)
                print('{}/{} signals calibrated and written to HDF5'.format(i + 1, numofsignals), end='\r')
                
            print('\n\033[34m{}/{} cameras cross-calibrated\033[0m'.format(j + 1, len(diagnamelist)))


# ---------------------------------------------------------------------------
# Other I/O utilities
# ---------------------------------------------------------------------------

def process_signals(shot):
    """Data acquisition and processing pipeline.

    Calls axuv_to_hdf() then calibrate_and_smooth(), and creates a per-shot
    folder under plots/ for figures produced from this data.

    :param shot: int or str, mandatory
        AUG discharge number
    """
    axuv_to_hdf(shot)
    calibrate_and_smooth(shot)
    shot = str(shot)
    if not os.path.exists('plots/' + shot):
        os.mkdir('plots/' + shot)


def calculate_indices(timegrid, start, end):
    """Find the indices in *timegrid* corresponding to *start* and *end* [seconds].

    Prints a warning and falls back to 0 / len(timegrid) if a lookup fails.

    :param timegrid: array-like – the time base to search
    :param start:    float – start time [s]
    :param end:      float – end time   [s]
    :returns:        [startindex, endindex]
    """
    try:
        i = bisect.bisect_left(timegrid, start)
    except Exception:
        print('Problem with start time, time base from {} to {} seconds.'.format(
            timegrid[0], timegrid[-1]))
        print('Defaulting start to index 0')
        i = 0

    try:
        j = bisect.bisect_left(timegrid, end)
    except Exception:
        print('Problem with end time, time base from {} to {} seconds.'.format(
            timegrid[0], timegrid[-1]))
        print('Defaulting end to index -1')
        j = len(timegrid)

    return [i, j]


def read_data_base(diagname, start, end, shotno, mode='shotno', h5file=None):
    """Custom HDF5 file reader and time-signal finder.

    Extracts data related to a specific diagnostic either by opening a file
    (mode='shotno') or by reading an already-open h5py.File instance
    (mode='h5file').  In 'h5file' mode the caller retains ownership of the
    file handle – this function will never close it.

    :param diagname: str – custom diagnostic name (see axuv/config.py)
    :param start:    float – start of the time window [s]
    :param end:      float – end of the time window   [s]
    :param shotno:   int or str – AUG discharge number
    :param mode:     'shotno' (default) opens a file; 'h5file' uses *h5file*
    :param h5file:   open h5py.File – required when mode='h5file'

    :returns: (signalnames, timesignal, indices, lenofsignals, f)
              or an integer error code on failure:
                1 – file not found
                2 – diagnostic group not found
                3 – no time signal found
                4 – problem reading time signal
                5 – invalid mode
                6 – invalid h5py.File instance
    """
    shotno = str(shotno)
    owns_file = False  # True only when this function opened the file itself

    if mode == 'shotno':
        try:
            filename = ROOTFOLDER + 'smoothed_data/' + shotno + '_sm.h5'
            f = h5py.File(filename, 'r')
            owns_file = True
        except Exception:
            print('\033[31mNo cross-calibrated file\033[0m')
            return 1
    elif mode == 'h5file':
        f = h5file
    else:
        print("Choose a valid :mode: parameter, either 'shotno' or 'h5file'")
        return 5

    if not isinstance(f, h5py.File):
        print("No valid h5py.File instance")
        return 6

    try:
        signalnames = list(f['/'.join([shotno, diagname])].keys())
    except Exception as e:
        print(e)
        if owns_file:
            f.close()
        print('\033[31mNo cross-calibrated diagnostic\033[0m')
        return 2

    timesignalname = None
    for signal in signalnames:
        if '_time' in signal:
            timesignalname = signal

    if timesignalname is None:
        print('\033[31mCould not find "_time" in any of the signal names\033[0m')
        if owns_file:
            f.close()
        return 3

    try:
        fulltimesignal = f['/'.join([shotno, diagname, timesignalname])][()]
        indices = calculate_indices(fulltimesignal, start, end)
        timesignal = fulltimesignal[indices[0]:indices[1]]
        lenofsignals = len(timesignal)
        signalnames.pop(signalnames.index(timesignalname))
    except Exception as e:
        print('\033[31mProblem with the time signal\033[0m\n{}'.format(e))
        if owns_file:
            f.close()
        return 4

    return signalnames, timesignal, indices, lenofsignals, f