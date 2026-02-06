import os
import h5py
import copy
import bisect
import skimage
import shapely
import shapely.ops
import datetime
import matplotlib
import numpy as np
import aug_sfutils as sf
import shapely.affinity as affinity
import shapely.geometry as geom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from csv import writer
from copy import deepcopy
from scipy.signal import savgol_filter
from scipy.interpolate import griddata


# Update default matplotlib parameters
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({'font.size': 16, "figure.dpi" : 150})
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams['lines.linewidth'] = 2

ROOTFOLDER = "/shares/departments/AUG/users/lefer/AXUV/"

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

# Dictionary of AXUV signal names we need for SPI experiment data analysis based on AUG shotfiles:
SIGNAL_NAMES = {"DVC_S5_vert": ["XVR", ['S0L0A00','S0L0A01','S0L0A02','S0L0A03','S0L0A04','S0L0A05','S0L0A06','S0L0A07',
                                        'S0L0A08','S0L0A09','S0L0A10','S0L0A11','S0L0A12','S0L0A13','S0L0A14','S0L0A15',
                                        'S0L1A00','S0L1A01','S0L1A02','S0L1A03','S0L1A04','S0L1A05','S0L1A06','S0L1A07',
                                        'S0L1A08','S0L1A09','S0L1A10','S0L1A11','S0L1A12','S0L1A13','S0L1A14','S0L1A15',
                                        'S0L2A00','S0L2A01','S0L2A02','S0L2A03','S0L2A04','S0L2A05','S0L2A06','S0L2A07',
                                        'S0L2A08','S0L2A09','S0L2A10','S0L2A11','S0L2A12','S0L2A13','S0L2A14','S0L2A15']],
                "DHC_S5_horiz": ["XVR", ['S1L0A00','S1L0A01','S1L0A02','S1L0A03','S1L0A04','S1L0A05','S1L0A06','S1L0A07',
                                         'S1L0A08','S1L0A09','S1L0A10','S1L0A11','S1L0A12','S1L0A13','S1L0A14','S1L0A15',
                                         'S1L1A00','S1L1A01','S1L1A02','S1L1A03','S1L1A04','S1L1A05','S1L1A06','S1L1A07',
                                         'S1L1A08','S1L1A09','S1L1A10','S1L1A11','S1L1A12','S1L1A13','S1L1A14','S1L1A15',
                                         'S1L2A00','S1L2A01','S1L2A02','S1L2A03','S1L2A04','S1L2A05','S1L2A06','S1L2A07',
                                         'S1L2A08','S1L2A09','S1L2A10','S1L2A11','S1L2A12','S1L2A13','S1L2A14','S1L2A15']],
                "D01_S1_vert": ["XVU", ['S6L0A00','S6L0A01','S6L0A02','S6L0A03','S6L0A04','S6L0A05','S6L0A06','S6L0A07',
                                        'S6L0A08','S6L0A09','S6L0A10','S6L0A11','S6L0A12','S6L0A13','S6L0A14','S6L0A15',
                                        'S6L0A16','S6L0A17','S6L0A18','S6L0A19','S6L0A20','S6L0A21','S6L0A22','S6L0A23',
                                        'S6L0A24','S6L0A25','S6L0A26','S6L0A27','S6L0A28','S6L0A29','S6L0A30','S6L0A31',
                                        'S6L1A00','S6L1A01','S6L1A02','S6L1A03','S6L1A04','S6L1A05','S6L1A06','S6L1A07',
                                        'S6L1A08','S6L1A09','S6L1A10','S6L1A11','S6L1A12','S6L1A13','S6L1A14','S6L1A15']],
                "D16_S16_vert": ["XVU", ['S6L1A16','S6L1A17','S6L1A18','S6L1A19','S6L1A20','S6L1A21','S6L1A22','S6L1A23',
                                         'S6L1A24','S6L1A25','S6L1A26','S6L1A27','S6L1A28','S6L1A29','S6L1A30','S6L1A31',
                                         'S6L2A00','S6L2A01','S6L2A02','S6L2A03','S6L2A04','S6L2A05','S6L2A06','S6L2A07',
                                         'S6L2A08','S6L2A09','S6L2A10','S6L2A11','S6L2A12','S6L2A13','S6L2A14','S6L2A15',
                                         'S6L2A16','S6L2A17','S6L2A18','S6L2A19','S6L2A20','S6L2A21','S6L2A22','S6L2A23',
                                         'S6L2A24','S6L2A25','S6L2A26','S6L2A27','S6L2A28','S6L2A29','S6L2A30','S6L2A31']],
                "D13_S13_vert": ["XVS", ['S3L0A00','S3L0A01','S3L0A02','S3L0A03','S3L0A04','S3L0A05','S3L0A06','S3L0A07',
                                         'S3L0A08','S3L0A09','S3L0A10','S3L0A11','S3L0A12','S3L0A13','S3L0A14','S3L0A15',
                                         'S3L1A00','S3L1A01','S3L1A02','S3L1A03','S3L1A04','S3L1A05','S3L1A06','S3L1A07',
                                         'S3L1A08','S3L1A09','S3L1A10','S3L1A11','S3L1A12','S3L1A13','S3L1A14','S3L1A15',
                                         'S3L2A00','S3L2A01','S3L2A02','S3L2A03','S3L2A04','S3L2A05','S3L2A06','S3L2A07',
                                         'S3L2A08','S3L2A09','S3L2A10','S3L2A11','S3L2A12','S3L2A13','S3L2A14','S3L2A15']],
                "DHT_S16_horiz": ["XVU", ['S7L1A16','S7L1A17','S7L1A18','S7L1A19','S7L1A20','S7L1A21','S7L1A22','S7L1A23',
                                          'S7L1A24','S7L1A25','S7L1A26','S7L1A27','S7L1A28','S7L1A29','S7L1A30','S7L1A31',
                                          'S7L2A00','S7L2A01','S7L2A02','S7L2A03','S7L2A04','S7L2A05','S7L2A06','S7L2A07',
                                          'S7L2A08','S7L2A09','S7L2A10','S7L2A11','S7L2A12','S7L2A13','S7L2A14','S7L2A15',
                                          'S7L2A16','S7L2A17','S7L2A18','S7L2A19','S7L2A20','S7L2A21','S7L2A22','S7L2A23',
                                          'S7L2A24','S7L2A25','S7L2A26','S7L2A27','S7L2A28','S7L2A29','S7L2A30','S7L2A31']],
                "D15_S15_vert": ["XVU", ['S7L0A00','S7L0A01','S7L0A02','S7L0A03','S7L0A04','S7L0A05','S7L0A06','S7L0A07',
                                         'S7L0A08','S7L0A09','S7L0A10','S7L0A11','S7L0A12','S7L0A13','S7L0A14','S7L0A15',
                                         'S7L0A16','S7L0A17','S7L0A18','S7L0A19','S7L0A20','S7L0A21','S7L0A22','S7L0A23',
                                         'S7L0A24','S7L0A25','S7L0A26','S7L0A27','S7L0A28','S7L0A29','S7L0A30','S7L0A31',
                                         'S7L1A00','S7L1A01','S7L1A02','S7L1A03','S7L1A04','S7L1A05','S7L1A06','S7L1A07',
                                         'S7L1A08','S7L1A09','S7L1A10','S7L1A11','S7L1A12','S7L1A13','S7L1A14','S7L1A15']]
               }

# All the keys of the dictionary
SIG_KEY_LIST = list(SIGNAL_NAMES.keys())

# Diagnostic name aliases
D15 = "D15_S15_vert"    # Sector 15 vertical - some channels are problematic, mostly not used - clockwise from S16
D16 = "D16_S16_vert"    # Sector 16 vertical - at SPI location - poloidal cross section is mappable in 2D
D01 = "D01_S1_vert"     # Sector 01 vertical - counter clockwise neighbor of S16
DVC = "DVC_S5_vert"     # Sector 05 vertical - poloidal cross section is mappable in 2D
D13 = "D13_S13_vert"    # Sector 13 vertical
DHC = "DHC_S5_horiz"    # Sector 05 horizontal - poloidal cross section is mappable in 2D
DHT = "DHT_S16_horiz"   # Sector 16 horizontal - poloidal cross section is mappable in 2D

# Categorization of diagnostics based on orientation
VERT = [D16, D01, DVC, D13]
HORIZ = [DHT, DHC]
DIAGNAMES_IN_S5_AND_S16 = [D16, DHT, DVC, DHC]
# These diagnostics have higher time resolution
NEWAXUV = [D16, D01, DHT]

# Default channel ranges to consider when evaluating data in poliodal cross section
VERT_S5_RANGE = [0, 47]
HORIZ_S5_RANGE = [3, 47]
VERT_S16_RANGE = [0, 29]
HORIZ_S16_RANGE = [3, 44]

# Ellipse used for masking the data in the poloidal cross section
ELLIPSE_U = 1.59     # x-position of the center
ELLIPSE_V = 0.11     # y-position of the center
ELLIPSE_A = 0.58     # radius on the x-axis
ELLIPSE_B = 1.03     # radius on the y-axis
ELLIPSE_T = np.linspace(0, 2*np.pi, 100)
ELLIPSE_R = (ELLIPSE_A * ELLIPSE_B) / np.sqrt((ELLIPSE_B * np.cos(ELLIPSE_T))**2 + (ELLIPSE_A * np.sin(ELLIPSE_T))**2)
ELLIPSE_XY = np.stack([ELLIPSE_U + ELLIPSE_R * np.cos(ELLIPSE_T), ELLIPSE_V + ELLIPSE_R * np.sin(ELLIPSE_T)], 1)
ELLIPSE = geom.Polygon(ELLIPSE_XY)

# SPI pellet database and corresponding column names
try:
    SPI_DB = np.genfromtxt(ROOTFOLDER + "CSVs/pellets.csv", delimiter=",")
except:
    print('\033[31mWARNING! Could not read pellets.csv')
SPI_COLUMN_NAMES = ["#", " Ne% ", " GT-", " SpeedA ", " first light ", " Delay ",
                     " SpeedM ", " SpeedML ", " SpeedMU ", " dSpeedM ", " Diameter ", " Mode "] 

# Current dip and peak times database
try:
    #CURRENT_DB = np.genfromtxt("CSVs/currentdipandpeak.csv", delimiter=",")
    CURRENT_DB = np.genfromtxt(ROOTFOLDER + "CSVs/startofincline.csv", delimiter=",")
except:
    print('\033[31mWARNING! Could not read "current" database')

# Neon amount database from desublimation
try:
    DESUBLIM_DB = np.genfromtxt(ROOTFOLDER + "CSVs/desublimation.csv", delimiter=",")
except:
    print('\033[31mWARNING! Could not read desublimation.csv')

# Comprehensive pellet database
try:
    #FULL_DB = np.genfromtxt("CSVs/comprehensive_db_new.csv", delimiter=",", skip_header=1)
    FULL_DB = np.genfromtxt(ROOTFOLDER + "CSVs/db_with_sqrt.csv", delimiter=",", skip_header=1)
except:
    print('\033[31mWARNING! Could not read comprehensive database')

# AXUV lines of sights database
try:
    LOS_DB = np.genfromtxt(ROOTFOLDER + "CSVs/AXUV_LOSs.csv", delimiter=",", names=True, 
                           dtype="S3,i8,i8,i8,S7,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8")
except:
    print('\033[31mWARNING! Could not read AXUV LOS file')


class LOS:
    def __init__(self, signalname):
        self.dbi = np.where(LOS_DB["RAW"] == bytes(signalname, 'utf-8'))[0][0]
        diagname = LOS_DB["Cam"][self.dbi].decode('utf-8')
        for diag in DIAGNAMES_IN_S5_AND_S16:
            if diagname in diag:
                self.diagname = diag
        self.signalname = signalname
        self.Rstart = LOS_DB["R_start"][self.dbi]
        self.zstart = LOS_DB["z_start"][self.dbi]
        self.Rend = LOS_DB["R_end"][self.dbi]
        self.zend = LOS_DB["z_end"][self.dbi]
        self.midpoint_R = (self.Rstart + self.Rend) / 2
        self.length = LOS_DB["length"][self.dbi]
        self.vector = [(self.Rend - self.Rstart) ** 2, (self.zend - self.zstart) ** 2] / (self.length ** 2)
        self.startpoint = geom.Point(self.Rstart, self.Rend)
        line = geom.LineString([(self.Rstart, self.zstart), (self.Rend, self.zend)])
        self.line_0 = line
        self.line = affinity.scale(line, xfact=2, yfact=2)
        # # self.total_volume = (self.line.length ** 3) * (2.8/45) * (7/45) * (1/3)
        # Pinholes 2x0.8 mm?

    def intersects(self, polygon):
        return shapely.intersection(self.line, polygon)

    def plot(self, color=None, lw=.5):
        plt.plot([self.Rstart, self.Rend], [self.zstart, self.zend], lw=lw, c=color)

    def load_data(self, file, shotno, ids=None, downsample=True):
        if ids is not None:
            data = file['/'.join([shotno, self.diagname, self.signalname + "_data"])][()][ids[0]:ids[1]]
        else:
            data = file['/'.join([shotno, self.diagname, self.signalname + "_data"])][()]
        
        if downsample:
            if any(x == self.diagname for x in NEWAXUV):
                factor = 10
            else:
                factor = 4

            newend = len(data) - (len(data) % factor)
            self.data = data[:newend].reshape(-1, factor).mean(axis=1)
        else:
            self.data = data

        self.mask = np.ones(len(self.data))

    def set_data(self, data):
        self.data = data

    def mask(self, values):
        self.mask = values


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
        f = h5py.File('data_export/' + shot + '_AXUV.h5', 'w')
    else:
        f = h5py.File(filewrite, 'w')

    # Load equilibrium to determine the end of plasma time
    equ = sf.EQU(int(shot))
    # Add 1.5 second to be able to adjust the zero-level of the signals when there is no plasma
    endtime = equ.time[-1] + 1.5

    # Go through all signals from all diagnostics, take the shortened timesignal once
    # Save timesignal and all data signals to HDF5 starting from :starttime:
    for key in SIG_KEY_LIST:
        diag = SIGNAL_NAMES[key][0]
        time_taken = False
        for sig in SIGNAL_NAMES[key][1]:
            signal = sf.SFREAD(int(shot), diag, experiment='AUGD')
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
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE + ERASE_LINE 
                      + CURSOR_UP_ONE + ERASE_LINE, end= '\r')
                print('\033[32mCreated dataset for ' + hierarchy_d + '\033[0m')

            except:
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE + ERASE_LINE, end= '\r')
                f.create_dataset(hierarchy_d, data=np.zeros([signallength]))
                print('\033[31mWARNING! Could not create dataset for ' + hierarchy_d 
                      + '\033[0m\nValues have been substituted by zeros')

    f.close()

def moving_average(array, window_size=3):
    """Calculates moving average of array with given window size
    Window size must be a positive odd integer"""

    if not window_size % 2 and window_size < 0:
        raise ValueError("Window size must be a positive odd integer")

    length = len(array)
    displacement = window_size // 2

    # Pad the array with zeros on both sides
    padded_arr = np.zeros(length + window_size - 1)
    padded_arr[displacement:-displacement] = arr

    # Initialize an empty array to store moving averages
    # Same length as initial array
    moving_averages = np.zeros(length)

    # Loop through the padded array to consider every window
    for i in range(length):
        # Calculate the average of current window
        window_average = np.sum(padded_arr[i:i+window_size]) / window_size
        
        # Store the average of current window in moving average array
        moving_averages[i] = window_average

return moving_averages
    
def calibrate_and_smooth(shotno, fileread=None, filewrite=None, savgol=True):
    """Moves all signals by an offset based on averaging when there is no plasma
    Calibrates gains based on the CSV gains files generated from #40989 flat-top
    Smooths both the 'new' and the 'old' signals with different Savitzky-Golay filters or with moving average
    Cuts off 1 second at the end then saves the smoothed data and the timesignals into a new HDF5 file

    Can be used from #40342 until #41307

    :param shotno: int or str, mandatory
        AUG discharge number
    :param fileread: str, optional
        filename to serve as input
    :param filewrite: str, optional
        filename for the output
    :param savgol: bool, optional
        if True, smoothing is done via Savitzky-Golay filters
        if False, smoothing is done by taking the moving average
    """

    shotno = str(shotno)
    if fileread is None:
        fileread = 'data_export/' + shotno + '_AXUV.h5'
    if filewrite is None and savgol:
        filewrite = 'smoothed_data/' + shotno + '_sm.h5'
    elif filewrite is None and not savgol:
        filewrite = 'ma_data/' + shotno + '_ma.h5'
    fr = h5py.File(fileread, 'r')
    fw = h5py.File(filewrite, 'w')

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

            if savgol:
                if diagname in NEWAXUV:
                    data = savgol_filter(data * gains[i], 51, 5)
                else:
                    data = savgol_filter(data, 31, 7)
            else:
                if diagname in NEWAXUV:
                    data = moving_average(data * gains[i], 9)
                else:
                    data = savgol_filter(data, 7)

            fw.create_dataset(hierarchy, data=data)
            print('{}/{} signals calibrated and written to HDF5'.format(i + 1, numofsignals), end='\r')
            
        print('\n\033[34m{}/{} cameras cross-calibrated\033[0m'.format(j + 1, len(diagnamelist)))
    
    fr.close()
    fw.close()
    
def process_signals(shot):
    """Data acqusition and processing 
    Executes axuv_to_hdf() and calibrate_and_smooth()
    Also creates a folder under plots/ for the figures created from this data

    :param shot: int or str, mandatory
        AUG discharge number
    """
    axuv_to_hdf(shot)
    calibrate_and_smooth(shot)
    shot = str(shot)
    if not os.path.exists('plots/' + shot): 
        os.mkdir('plots/' + shot)

def calculate_indices(timegrid, start, end):
    """Finds the indexes in :timegrid: corresponding to :start: and :end: given in seconds
    Prints warning if there is a problem, and sets 0 and -1 indices as default in this case

    :return indices: tuple of startindex and endindex
    """
    try:
        i = bisect.bisect_left(timegrid, start)
    except:
        print('Problem with start time, time base from {} to {} seconds.'.format(timegrid[0], timegrid[-1]))
        print('Defaulting start to index 0')
        i = 0
        
    try:
        j = bisect.bisect_left(timegrid, end)
    except:
        print('Problem with end time, time base from {} to {} seconds.'.format(timegrid[0], timegrid[-1]))
        print('Defaulting end to index -1')
        j = len(timegrid)
    
    indices = [i, j]
    return indices

def read_data_base(diagname, start, end, shotno, mode='shotno', h5file=None):
    """Custom HDF5 file reader, and time signal finder
    Extracts data related to a specific dignostic 
        - either by opening a file or by reading an opened h5py.File instance
    Gives feedback on some exceptions

    :param diagname: str, mandatory
        custom diagnostic name - see at the beggining of axuv_scripts.py
    :param start: float, mandatory
        the experiment time from when to start data loading
    :param end: float, mandatory
        the experiment time until when to load data
    :param shotno: int or str, mandatory
        AUG discharge number
    :param mode: either 'shotno' or 'h5file'
        'shotno' - must provide :shotno: too - opens and reads an HDF5 filr
        'h5file' - must provide :h5file: too - reads an already open h5py.File instance
    :param h5file: h5py.File, optional
        open custom HDF5 file - mandatory if mode is 'h5file'

    :return signalnames: list of strings 
        signal names of the specified diagnostic
    :return numofsignals: int
        number of signals or channels
    :return timesignal: np.arr
        the time signal
    :return indices: tuple
        (startindex, endindex)
    :return lenofsignals: int
        length of signals along the time axis
    :return f: h5py.File
        the opened custom HDF5 file containing the data
    """
    if mode == 'shotno':
        try:
            filename = ROOTFOLDER + 'smoothed_data/' + shotno + '_sm.h5'    
            f = h5py.File(filename, 'r')
        except:
            f.close()
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
        f.close()
        print('\033[31mNo cross-calibrated diagnostic\033[0m')
        return 2
    
    timesignalname = None
    for signal in signalnames:
        if '_time' in signal: timesignalname = signal
    
    if timesignalname is not None:
        try:
            fulltimesignal = f['/'.join([shotno, diagname, timesignalname])][()]
            indices = calculate_indices(fulltimesignal, start, end)
            timesignal = f['/'.join([shotno, diagname, timesignalname])][()][indices[0]:indices[1]]
            lenofsignals = len(timesignal)
            index = signalnames.index(timesignalname)
            signalnames.pop(index)
        except Exception as e:
            print('\033[31mProblem with the time signal\033[0m\n{}'.format(e))
            return 4
    else:
        print('\033[31mCould not find "_time" in any of the signal names\033[0m')
        return 3
        
    return signalnames, timesignal, indices, lenofsignals, f  
                
def plot_diag_non_mapped(shotno, diagname, start=2.3, end=2.4, norm='log', vmin=1e4, vmax=1e8,
                         save=False, vtimes=None, current=False, signalrange=[0, 47], hline=None):
    """Plots diagnostic data in 2D without mapping it to the R coordinate of AUG
    X axis is time, Y axis is channel number of the diagnostic
    Uses matplotlib.pyplot.pcolormesh
    Optionally plots the plasma current in an axis below the pcolormesh

    :param shotno: int or str, mandatory
        AUG discharge number
    :param diagname: str, mandatory
        custom diagnostic name - see at the beggining of axuv_scripts.py
    :param start: float, optional
        the experiment time from when to start data loading
    :param end: float, optional
        the experiment time until when to load data
    :param norm: str, 'linear' or 'log', optional
        for the pcolormesh plot normalization
    :param vmin: float, optional
        for pcolormesh
    :param vmax: float, optional
        for pcolormesh
    :param save: bool, optional
        to save or not to save
    :param vtimes: 2-element tuple or np.arr, optional
        plot vertical lines at these time coordinates
        for the determined current dip and peak times
    :param current: bool, optional
        whether or not to plot the plasma current
    :param signalrange: 2-element list of integers, optional
        range of channel INDICES (channel number - 1) to plot, provide the first and last index
    """
    shotno = str(shotno)
    
    signalnames, timesignal, indices, lenofsignals, f = read_data_base(diagname, start=start, end=end, shotno=shotno)
    numofsignals = signalrange[1] - signalrange[0] + 1
    tickrange = np.arange(signalrange[0] + 1, signalrange[1] + 2, 4)
    plotrange = range(signalrange[0] + 1, signalrange[1] + 2)
    
    data2d = np.zeros([numofsignals, lenofsignals])
    
    for i in range(numofsignals):
        data2d[i, :] = f['/'.join([shotno, diagname, signalnames[i+signalrange[0]]])][()][indices[0]:indices[1]]
        print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')
    
    if current:
        fig = plt.figure(figsize=[8,6])
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
        ax1, ax2 = gs.subplots(sharex=True)
        fig.suptitle('#' + shotno + ' ' + diagname.split('_')[0] + ' signals with plasma current')
        im = ax1.pcolormesh(timesignal, plotrange, data2d, norm=norm, 
                            vmin=vmin, vmax=vmax, zorder=1)
        
        shotno = str(shotno)
        fpc = sf.SFREAD(int(shotno), 'FPC', experiment='AUGD')
        Ip = fpc.getobject("IpiFP", cal=True)
        time = fpc.gettimebase("IpiFP")
        indices = calculate_indices(time, start, end)
        timesignal = time[indices[0]:indices[1]]
        datatoplot = Ip[indices[0]:indices[1]]/1000
        ax2.plot(timesignal, datatoplot)
        ax2.set_ylabel(r'I$_p$ [kA]')
        if hline is not None:
            ax1.axhline(hline, zorder=20)
        if vtimes is not None:
            (diptime, peaktime) = vtimes
            for axis in [ax1, ax2]:
                axis.axvline(x=diptime, color='green', linestyle='--', label='Dip: {:.6f} s'.format(diptime),
                            linewidth=1, zorder=2)
                axis.axvline(x=peaktime, color='blue', linestyle='--', label='Peak: {:.6f} s'.format(peaktime),
                            linewidth=1, zorder=3)
            ax2.legend(loc='lower left')
        
        cbar = plt.colorbar(im, ax=ax1)
        ax1.set_facecolor('black')
        ax1.set_yticks(tickrange)
        ax1.xaxis.grid(True, linestyle='--', zorder=5)
        ax2.grid()
        cbar.set_label('Diode signals [a. u.]')
        ax2.set_xlabel('Time [s]')
        ax1.set_ylabel('Channel number of camera')
    else:
        fig = plt.figure(figsize=[8,4.5])
        plot = plt.pcolormesh(timesignal, plotrange, data2d, norm=norm,
                              vmin=vmin, vmax=vmax, zorder=1)
        cbar = plt.colorbar()
        axes = plt.gca()
        axes.set_facecolor('black')
        if hline is not None:
            axes.axhline(hline, zorder=20)
        plt.yticks(tickrange)
        axes.xaxis.grid(True, linestyle='--', zorder=5)
        cbar.set_label('Diode signals [a. u.]')
        plt.xlabel('Time [s]')
        plt.ylabel('Channel number of camera')
        plt.title('#' + shotno + ' ' + diagname.split('_')[0])
    
        if vtimes is not None:
            plt.axvline(x=vtimes[0], color='green', linestyle='--', label=r'I$_p$ dip',
                        linewidth=1, zorder=2)
            plt.axvline(x=vtimes[1], color='blue', linestyle='--', label=r'I$_p$ peak',
                        linewidth=1, zorder=3)
            plt.legend(loc='upper left')

    if save:
        if not os.path.exists('plots/' + shotno): os.mkdir('plots/' + shotno)
        savename = '_'.join([diagname, str(start), str(end), str(signalrange[0]) + "-" + str(signalrange[1]),
                             'Ip', str(bool(current))])
        plt.savefig('plots/' + shotno + '/' + savename + ".png", dpi=150)
        plt.close(fig)
        print('Saved as ' + 'plots/' + shotno + '/' + savename + ".png")
    else:
        plt.show()
    
    f.close()

def upsample_interpolate(templist, timesignals):
    """Used for plot_diff(), returns a 2-element list of 2D arrays
    """
    dims1 = templist[0].shape
    dims2 = templist[1].shape
    tobesubtracted = []
    if dims1[1] < dims2[1]:
        tempdata = np.zeros([dims2[0], dims2[1]])
        for row in range(dims1[0]):
            tempdata[row, :] = np.interp(np.linspace(0, 10, dims2[1]), 
                                         np.linspace(0, 10, dims1[1]), templist[0][row, :])
            
        tobesubtracted.append(tempdata)
        tobesubtracted.append(templist[1])
        timesignal = timesignals[1]
    elif dims2[1] < dims1[1]:
        tempdata = np.zeros([dims1[0], dims1[1]])
        for row in range(dims1[0]):
            tempdata[row, :] = np.interp(np.linspace(0, 10, dims1[1]), 
                                         np.linspace(0, 10, dims2[1]), templist[1][row, :])

        tobesubtracted.append(templist[0])
        tobesubtracted.append(tempdata)
        timesignal = timesignals[0]
    else:
        tobesubtracted.append(templist[0])
        tobesubtracted.append(templist[1])
        timesignal = timesignals[0]
    
    return tobesubtracted

def plot_diff(shotno, diagname1, diagname2, start=2.3, end=2.4, faultychannels='repair',
              linthresh=1e5, save=False, vtimes=None, fname=None):
    """Plots the difference of two diagnostics in 2D with symmetric logarithmic normalization
    Optionally linearly interploates missing channel data based on neighbor channels
    Subtracts the signal of :diagname2: from :diagname1:
    Optionally plots the plasma current in an axis below the pcolormesh
    Returns the two data arrays which were subtracted and the timesignal

    :param shotno: int or str, mandatory
        AUG discharge number
    :param diagname1: str, mandatory
        custom diagnostic name - see at the beggining of axuv_scripts.py
    :param diagname2: str, mandatory
        custom diagnostic name - see at the beggining of axuv_scripts.py
    :param start: float, optional
        the experiment time from when to start data loading
    :param end: float, optional
        the experiment time until when to load data
    :param faultychannels: list of integers or 'repair', optional
        if None - all channels of diag2 will be subtracted from the channels of diag1
        if list - provide a list of channels where at least one diagnostic is missing the signal
                  and the function will set the resulting data to zero for those channels
        if 'repair' - linearly interpolate the missing channels for both diagnostics and subtract after
    :param save: bool, optional
        to save or not to save
    :param vtimes: 2-element tuple or np.arr, optional
        plot vertical lines at these time coordinates
        for the determined current dip and peak times
    :param current: bool, optional
        whether or not to plot the plasma current

    :return tobesubtracted: 2 element list of np.ndarrays
        1st element is the data from diagname1, 2nd is from diagname2
    :return timesignal: np.ndarray
        timesignal corresponding to the data
    """
    shotno = str(shotno)
    if not os.path.exists('plots/' + shotno): os.mkdir('plots/' + shotno)
    
    try:
        if fname is None:
            filename = 'smoothed_data/' + shotno + '_sm.h5'
        else:
            filename = fname  
        f = h5py.File(filename, 'r')
    except:
        f.close()
        print('\033[31mNo cross-calibrated file\033[0m')
        return 1
    
    datasignals = []
    timesignals = []
    for diagname in [diagname1, diagname2]:
        signalnames, timesignal, indices, lenofsignals, _ = read_data_base(diagname, start=start, end=end, 
                                                                           shotno=shotno, mode='h5file', h5file=f)
        timesignals.append(timesignal)
        numofsignals = len(signalnames)
        data2d = np.zeros([numofsignals, lenofsignals])

        for i in range(numofsignals):
            if isinstance(faultychannels, list) and (i in faultychannels):
                data2d[i, :] = np.zeros([lenofsignals])
                print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')
            else:
                data2d[i, :] = f['/'.join([shotno, diagname, signalnames[i]])][()][indices[0]:indices[1]]
                print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')
                
        datasignals.append(data2d)

    f.close()

    # Optionally linearly interpolate the missing signals - starts from channel 1
    # If one of the neighboring channels is also missing, copies the other neighbor
    # IMPORTANT: Will not work for the first channel(s) if Channel 2 AND 3 (AND 4, ...) are all missing 
    if faultychannels == 'repair':
        templist = []
        for data_r in datasignals:
            booleans = ~np.all(data_r < 1e4, axis=1)
            indices = np.where(~booleans)
            for i in indices[0]:
                if i-1 < 0:
                    if np.all(data_r[i+1, :] < 1e4):
                        data_r[i, :] = data_r[i+2, :]
                    else:
                        data_r[i, :] = data_r[i+1, :]
                elif (i+1) > (numofsignals-1) or np.all(data_r[i+1, :] < 1e4):
                    data_r[i, :] = data_r[i-1, :]
                else:
                    data_r[i, :] = (data_r[i-1, :] + data_r[i+1, :]) / 2
            templist.append(data_r)
    else:
        templist = datasignals

    # check if both 2D arrays have the same dimensions, if not, upsample and interpolate the smaller        
    tobesubtracted = upsample_interpolate(templist, timesignals)

    fig = plt.figure(figsize=[8,5])
    datatoplot = tobesubtracted[0] - tobesubtracted[1]
    vmax = max([abs(datatoplot.min()), datatoplot.max()])  
    plot = plt.pcolormesh(timesignal, range(numofsignals + 1)[1:], datatoplot, 
                            norm=matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
                            cmap=plt.colormaps['RdBu'])

    cbar = plt.colorbar()
    axes = plt.gca()
    plt.yticks(np.arange(2, numofsignals + 1, 2))
    axes.xaxis.grid(True, color='black', linestyle='--', zorder=5)
    plt.xlabel('Time [s]')
    plt.ylabel('Channel number of camera')
    diagshort1 = diagname1.split("_")[0]
    diagshort2 = diagname2.split("_")[0]
    cbar.set_label('<- ' + diagshort2 + ' higher signal    |    ' + diagshort1 + ' higher signal ->')
    plt.title('#' + shotno + ' ' + diagshort2 + ' subtracted from ' + diagshort1)
    
    if vtimes is not None:
        plt.axvline(x=vtimes[0], color='green', linestyle='--', label=r'I$_p$ dip',
                    linewidth=1, zorder=2)
        plt.axvline(x=vtimes[1], color='blue', linestyle='--', label=r'I$_p$ peak',
                    linewidth=1, zorder=3)
        plt.legend(loc='upper left')
    
    if save:
        savename = '_'.join([diagshort1, 'minus', diagshort2, str(start), str(end), str(numofsignals)])
        if os.path.exists('plots/' + shotno + '/' + savename + ".png"):
            now = datetime.datetime.now()
            date_time = now.strftime("_%Y-%m-%d_%Hh-%Mm-%Ss")
            savename += date_time
        plt.savefig('plots/' + shotno + '/' + savename + ".png", dpi=150)
        plt.close(fig)
        print('Saved as ' + 'plots/' + shotno + '/' + savename + ".png")
    else:
        plt.show()
    
    return tobesubtracted, timesignal

def plot_1D(shotno, diagname, channel, start=2.3, end=2.4):
    """Plots one :channel: (input is the index) as a function of time"""
    shotno = str(shotno)
    signalnames, timesignal, indices, _, f = read_data_base(diagname, start=start, end=end, shotno=shotno)
    
    datatoplot = f['/'.join([shotno, diagname, signalnames[channel-1]])][()][indices[0]:indices[1]]
    
    fig = plt.figure(figsize=[8,5])
    plot = plt.plot(timesignal, datatoplot)
    axes = plt.gca()
    # axes.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    axes.set_ylim(None, None)
    plt.xlabel('Time [s]')
    plt.title(diagname + ' channel #' + str(channel) + '\n' + signalnames[channel-1])
    plt.show()
    f.close()

def plot_full_integral(shotno, diagname, yscale='linear', start=2.3, end=2.4):
    """Plots the sum of all channel signals of :diagname: as a function of time"""
    shotno = str(shotno)
    signalnames, timesignal, indices, lenofsignals, f = read_data_base(diagname, start=start, 
                                                                       end=end, shotno=shotno)
    
    datatoplot = np.zeros(lenofsignals)
    for signal in signalnames:
        datatoplot[:] += f['/'.join([shotno, diagname, signal])][()][indices[0]:indices[1]]
    f.close()

    fig = plt.figure(figsize=[8,5])
    plot = plt.plot(timesignal, datatoplot)

    axes = plt.gca()
    axes.set_yscale(yscale)
    if yscale == 'log':
        axes.set_ylim(1e6, None)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Integrated AXUV signal - all channels')
    plt.title('#' + shotno + ' ' + diagname.split('_')[0])
    plt.show()
    
def plot_integral_combined(shotno, array=VERT, yscale='linear', start=2.3, end=2.4, axes=None, savename=None):
    """Plots the integrals of diags in :array: in one plot with labels"""
    shotno = str(shotno)
    if axes is None:
        fig = plt.figure(figsize=[8,5])
        axes = plt.gca()
    
    for diagname in array:
        signalnames, timesignal, indices, lenofsignals, f = read_data_base(diagname, start=start, 
                                                                           end=end, shotno=shotno)
        datatoplot = np.zeros(lenofsignals)
        for signal in signalnames:
            datatoplot[:] += f['/'.join([shotno, diagname, signal])][()][indices[0]:indices[1]]

        axes.plot(timesignal, datatoplot, label=diagname.split('_')[0])
    
    f.close()
    axes.set_yscale(yscale)
    if yscale == 'log':
        axes.set_ylim(1e6, None)
        
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Integrated AXUV signals - all channels')
    axes.legend()
    

    plt.savefig(savename, dpi=200)
    plt.close()
    # if axes is None:
    #     if savename is None:
    #         plt.title('#' + shotno )
    #         plt.show()
    #     else:
            
def find_roots(x,y):
    """Finds the x locations where y has roots"""
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
    
def plot_current(shotno, start=2.3, end=2.4, topgrad=None, tdip=None, tpeak=None, plot=True, savename=None,
                 title=None):
    """Plots the total plasma current from AUG shotfile and it's derivative
    If :topgrad: (time in seconds) is not None, finds the first root of the derivative to the left and to the right
        and returns them as the current dip time and the current peak time
    If :tdip: and :tpeak: are set, they are assumed to be the current dip and peak times and will be returned
    """
    shotno = str(shotno)
    fpc = sf.SFREAD(int(shotno), 'FPC', experiment='AUGD')
    Ip = fpc.getobject("IpiFP", cal=True)
    time = fpc.gettimebase("IpiFP")
    indices = calculate_indices(time, start, end)
    timesignal = time[indices[0]:indices[1]]
    datatoplot = Ip[indices[0]:indices[1]]/1000
    derivative = np.gradient(datatoplot)
    
    if plot:
        fig = plt.figure(figsize=[8,3])
        plt.plot(timesignal, datatoplot, color='black', zorder=4)
        plt.grid(axis='x', which='both')
        axes = plt.gca()
        axes.set_xlabel('Time [s]')
        
        if savename is None:
            ax2 = axes.twinx()
            ax2.plot(timesignal, derivative, color='red', zorder=5)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=0)
            ax2.set_ylabel(r'dI$_{p}$/dt [kA/s]', color='red')
    
    if topgrad is not None:
        roots = find_roots(timesignal, derivative)
        try:
            peaktime = roots[roots > topgrad].min()
            diptime = roots[roots < topgrad].max()
        except:
            print("No root found!")
            peaktime = start
            diptime = end

        if plot:
            plt.axvline(x=diptime, color='green', linestyle='--', label='Dip: {:.8f} s'.format(diptime),
                        linewidth=1, zorder=2)
            plt.axvline(x=peaktime, color='blue', linestyle='--', label='Peak: {:.8f} s'.format(peaktime),
                        linewidth=1, zorder=3)
            
            print('Current dip at {:.8f} s and peak at {:.8f} s'.format(diptime, peaktime))
            plt.legend()
        
    if tdip is not None and tpeak is not None:
        peaktime = tpeak
        diptime = tdip
        
        if plot:
            plt.axvline(x=diptime, color='green', linestyle='--', label='Dip: {:.6f} s'.format(diptime),
                        linewidth=1, zorder=2)
            plt.axvline(x=peaktime, color='blue', linestyle='--', label='Peak: {:.6f} s'.format(peaktime),
                        linewidth=1, zorder=3)
        
            print('Current dip and peak manually set')
            plt.legend()
        
    if plot:
        axes.set_ylim(None, None)
        plt.xlabel('Time [s]')
        axes.set_ylabel('Plasma current [kA]', color='black')
        if title is not None:
            plt.title('#' + shotno + ' ' + title)
        else:
            plt.title('#' + shotno)
        if savename is not None:
            plt.savefig(savename + ".png", dpi=150)
            plt.close(fig)
        else:
            plt.show()
    
    if topgrad is not None or (tdip is not None and tpeak is not None):
        return diptime, peaktime
    else:
        return timesignal, datatoplot, derivative
    
def plot_timeslice(shotno, diagname, when=2.32, signalrange=[0, 47]):
    """Plots channel signals as line plot at a given time :when: - X axis is the channel numbers"""
    shotno = str(shotno)
    filename = 'smoothed_data/' + shotno + '_sm.h5'    
    f = h5py.File(filename, 'r')
    signalnames = list(f['/'.join([shotno, diagname])].keys())

    timesignalname = None
    for signal in signalnames:
        if '_time' in signal: timesignalname = signal
    
    timesignal = f['/'.join([shotno, diagname, timesignalname])][()]
    tsindex = signalnames.index(timesignalname)
    signalnames.pop(tsindex)
    numofsignals = signalrange[1] - signalrange[0] + 1
    index = bisect.bisect_left(timesignal, when)
    data1d = np.zeros(numofsignals)

    for i in range(signalrange[0], signalrange[1] + 1):
        data1d[i] = f['/'.join([shotno, diagname, signalnames[i]])][()][index]
    f.close()

    fig = plt.figure(figsize=[6,4])
    plt.step(range(signalrange[0] + 1, signalrange[1] + 2), data1d, where='mid')
    plt.show()

def downsample(shotno, diagname, start=2.31, end=2.32, signalrange=[0, 47]):
    """Downsamples timesignal and data to have 1e-5 s time resolution
    Returns downsampled data and time as tuple
    """
    # Downsampling factor depends on the dignostic, 
    # this way we will get the same time resolution for each, which is 1e-5 s
    if any(x == diagname for x in NEWAXUV):
        factor = 10
    else:
        factor = 4

    # read data from calibrated and smoothed HDF5 output
    shotno = str(shotno)
    signalnames, timesignal, indices, _, f = read_data_base(diagname, start=start, end=end, shotno=shotno)
    newend = timesignal.shape[0] - (timesignal.shape[0] % factor)
    timesignal = timesignal[:newend]
    # Downsampling first for the time, then for the signals
    downsampled_time = timesignal.reshape(-1, factor).mean(axis=1)

    numofsignals = signalrange[1] - signalrange[0] + 1
    dims = (downsampled_time.shape[0], numofsignals)
    downsampled_data = np.zeros(dims)

    for i, channel in enumerate(range(signalrange[0], signalrange[1])):
        tempdata = f['/'.join([shotno, diagname, signalnames[channel]])][()][indices[0]:indices[1]]
        downsampled_data[:, i] = tempdata[:newend].reshape(-1, factor).mean(axis=1)

    return downsampled_data, downsampled_time

def repair_2d_data(data, add_indices=None):
    booleans = ~np.all(data < 1e4, axis=0)
    npindices = np.where(~booleans)
    indices = []
    indices.extend(npindices[0])
    if add_indices is not None:
        indices.extend(add_indices)
    # Optionally linearly interpolate the missing signals - starts from channel 1
    # If one of the neighboring channels is also missing, copies the other neighbor
    # IMPORTANT: Will not work for the first channel(s) if Channel 1 AND 2 (AND 3, ...) are all missing
    for i in indices:
        if i-1 < 0:
            data[:, i] = data[:, i+1]
        elif (i+1) > (data.shape[1]-1) or np.all(data[:, i+1] < 1e4):
            data[:, i] = data[:, i-1]
        else:
            data[:, i] = (data[:, i-1] + data[:, i+1]) / 2
    return data

def ridge_filter(shotno, diagname, start=2.31, end=2.32, signalrange=[0, 47], sigmas=[1,2], 
                 normalize=True, plot=False, repair=True, vmin=1e4):
    """Downsamples, optionally repairs signals, then executes the Sato ridge following algorithm with :sigmas:
    Optionally normalizes the result of the algorithm in each timestep
    Optionally plots the new and old data
    Returns 3-element tuple of the new data, the downsampled timesignal, and the raw data
    """
    downsampled_data, downsampled_time = downsample(shotno, diagname, start=start, end=end, signalrange=signalrange)
    numofsignals = signalrange[1] - signalrange[0] + 1
    # Check which diodes gave very low signal
    booleans = ~np.all(downsampled_data < 1e4, axis=0)
    indices = np.where(~booleans)
    add_indices = None
    if diagname == DHT:
        add_indices = [16]

    if repair:
        tobefiltered = repair_2d_data(downsampled_data, add_indices=add_indices)
    else:
        tobefiltered = downsampled_data[:, booleans]

    # Sato ridge following algorithm - including higher sigmas smoothens and spreads the result more
    # If data is not repaired, this is done only on an array from which the missing channels were removed
    array = skimage.filters.sato(tobefiltered, sigmas=sigmas, black_ridges=False)

    # Whether or not normalize the result in every timestep so that the sum of the channels give 1 
    if normalize:
        row_sums = np.linalg.norm(array, axis=1)
        data = array / row_sums[:, np.newaxis]
    else:
        data = array / np.max(array)

    # If the data was not repaired, plug back the faulty channels into the array
    if not repair:
        for index in indices[0]:
            data = np.insert(data, index, 0, axis=1)

    if plot:
        # basic pcolormesh plotting 
        # - 1st the ridge followed result - linear
        # - 2nd the original data - log - default vlim=(1e4, 1e8)
        fig, axes = plt.subplots(1, 2, figsize=[10, 3])
        axes[0].pcolormesh(downsampled_time, range(downsampled_data.shape[1]), data.T, shading='nearest', 
                           norm='linear')
        axes[1].pcolormesh(downsampled_time, range(downsampled_data.shape[1]), downsampled_data.T, shading='nearest', 
                           norm='log', vmin=vmin, vmax=1e8)
        plt.show()

    return data, downsampled_time, tobefiltered

def show_poloidal(sector=16, vidx=12, hidx=35, u=ELLIPSE_U, v=ELLIPSE_V, a=ELLIPSE_A, b=ELLIPSE_B):
    """Shows the intersections in a sector, a specified point from the intersection matrix
    and the ellipse used to exclude the intersections outside of the vacuum-vessel
    """
    imatfile = 'LOSisects.h5'
    f = h5py.File(imatfile,'r')
    key = 'S' + str(sector) 
    imat = f[key][()]
    fig, ax = plt.subplots(dpi=150, figsize=(4,6))
    ax.set_aspect(1)

    gc_d = sf.getgc()
    for gc in gc_d.values():
        ax.plot(gc.r, gc.z, lw=.5, color='black')

    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(*imat.reshape(-1,2).T, '.', ms=3, zorder=1)
    plt.plot(u+a*np.cos(t), v+b*np.sin(t))
    plt.scatter(imat[vidx, hidx][0], imat[vidx, hidx][1], s=5,
                     color='orange', zorder=2)
    plt.xlim(1, 2.4)
    plt.xlabel('R [m]')
    plt.ylim(-1.3, 1.3)
    plt.ylabel('z [m]')
    plt.title('Sector ' + str(sector))
    plt.show()

def plot_qsurfaces(shot_or_equ, t_in, axes=None, diag="EQH"):
    """Plots the integer q-surfaces from equlibrium :diag: on a given :axes: at :t_in: time 
    AND returns the matplotlib.Line2D objects of the q-surfaces and the aug_sfutils.EQU equlibrium object

    OR if :axes: is None - returns the q=2 surface as a shapely.geom.Polygon and the aug_sfutils.EQU object 
    """
    if isinstance(shot_or_equ, str) or isinstance(shot_or_equ, float) or isinstance(shot_or_equ, int):
        equ = sf.EQU(int(shot_or_equ), diag=diag)
    elif isinstance(shot_or_equ, sf.EQU):
        equ = shot_or_equ

    if axes is not None:
        i = 1
        qcoords = []
        qs = []
        lines = []
        temp = 1
        while not np.isnan(temp):
            i += 1
            temp = sf.mapeq.get_q_surf(equ, qvalue=i, t_in=t_in, coord_out='rho_pol')
            qcoords.append(temp)
            qs.append(i)
    
        qsurfaces = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=qcoords, coord_in='rho_pol', all_lines=False)
        sep_Rz = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
        sep = axes.plot(sep_Rz[0][0][0], sep_Rz[1][0][0], alpha=0.7, ls='--', lw=1, label='LCFS')
        lines.append(sep)

        for i in range(len(qsurfaces[0][0]) - 1):
            surf = axes.plot(qsurfaces[0][0][i], qsurfaces[1][0][i], alpha=0.7, ls='-.', lw=0.5, label='q={}'.format(qs[i]))
            lines.append(surf)

        return lines, equ
    else:
        q2surf = sf.mapeq.get_q_surf(equ, qvalue=2, t_in=t_in, coord_out='rho_pol')
        sepcoords = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
        q2coords = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=q2surf, coord_in='rho_pol', all_lines=False)
        sep_polycoords = np.vstack((sepcoords[0][0][0], sepcoords[1][0][0])).T
        q2polycoords = np.vstack((q2coords[0][0][0], q2coords[1][0][0])).T
        try:
            q2poly = geom.Polygon(q2polycoords)
        except:
            q2poly = None

        try:
            sep_poly = geom.Polygon(sep_polycoords)
        except:
            sep_poly = None

        return q2poly, sep_poly, equ

def generate_isect_data3d(vertrange, horizrange, datavert, datahoriz, normalize=None, dontmultiply=False):
    """Generates new data at every intersection point based on :datavert: and :datahoriz:
    Creates a 48x48 matrix for every t timestep 
        and sets the value in each matrix[i,j] point to :datavert:[t, i] * :datahoriz:[t, j]
    The intended case is to use data from the ridge_filter() function
    It is important to provide the correct ranges

    At this point the normalize argument takes only the 'horiz' string

    Returns :multiplied: the new 3D array
    """
    vlen = datavert.shape[0]  # number of datapoints in time
    multiplied = np.zeros((vlen, 48, 48))
    normrange = None
    if normalize is not None:
        if normalize == 'horiz':
            normrange = horizrange
        else:
            print("Currently only 'horiz' is valid argument for normalize")
            return multiplied

    if vlen == datahoriz.shape[0]:
        matrixv = np.zeros((vlen, 48))
        matrixh = np.zeros((vlen, 48))
        for i, vertidx in enumerate(range(vertrange[0], vertrange[1])):
            matrixv[:, vertidx] = datavert[:, i]

        for j, horidx in enumerate(range(horizrange[0], horizrange[1])):
            matrixh[:, horidx] = datahoriz[:, j]

        # Suppose all radiation comes from inside the LCFS
        # Normalize the horizontal array to 1 inside the LCFS
        # Then multiply the vertical array with this normalized array
        if normrange is not None:
            row_sums = np.sum(matrixh, axis=1)
            matrixh = matrixh / row_sums[:, np.newaxis]

        if not dontmultiply:
            for k in range(vlen):
                multiplied[k] = np.outer(matrixv[k, :], matrixh[k, :])   
        elif dontmultiply:  # just sum the values from the 2 LOSs
            for t in range(vlen):
                for i in range(48):
                    for j in range(48):
                        multiplied[t, i, j] = matrixv[t, i] + matrixh[t, j]

        return multiplied
    else:
        print('ERROR: data arrays are not the same length in timesteps')
        return multiplied

def radiation_poloidal(data3d, timesignal, time, shotno, sector, multiplier=20, polygon=ELLIPSE,
                       interp=None, ax=None, plot=True, save=False, given_name=None):
    """Interpolate over the data from the 3D array generated by ridge_filter() - processed data
                                                             or by generate_isect_data3d() - raw data
    Optionally plot the sliced 3D data as scatter at a given time with the q-surfaces and the SPI vectors
        - either on a new figure, or on a given :ax: axis
        - :save: - bool, optional - save plot as PNG
        - :given_name: - str, optional - for saving
    If multiplier == "raw", then it is somewhat capable of outputting nice looking figures from raw data
    >WIP<

    Returns five values in a tuple
        1. the interpolated data array :grid_interp:
        2. :points: (R, z) coordinates of the valid intersection points
        3. :values: the multiplied values from the slice of the 3D input array at :points: locations
        4. and 5. :Rloc: and :zloc: R and z coordinates of the maximum based on the interpolation
    """
    imatfile = 'LOSisects.h5'
    f = h5py.File(imatfile,'r')
    key = 'S' + str(sector) 
    imat = f[key][()]
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
    else:
        x, y = polygon.exterior.xy
    res0 = []
    for i in range(48):
        res_temp = [p for p in isecs[i] if polygon.contains(geom.Point(p[:2]))]
        if len(res_temp) != 0:
            res0.append(res_temp)
    
    res = np.vstack(res0)
    points = np.array([res[:, 0], res[:, 1]]).T
    values = res[:, 2]

    if interp is not None:
        grid_R, grid_z = np.meshgrid(np.linspace(1, 2.2, int(600)),
                                    np.linspace(-1, 1.1, int(1100)), indexing='ij')
        grid_interp = griddata(points, values, (grid_R, grid_z), method=interp)
        valueindex = np.unravel_index(np.nanargmax(grid_interp, axis=None), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        zloc = grid_z[0, valueindex[1]]
    else:
        grid_interp = None
        maxrowindex = np.argmax(res[:, 2], axis=0)
        Rloc = res[maxrowindex, 0]
        zloc = res[maxrowindex, 1]

    if ax or plot:
        ax.set_aspect(1)
        # ax.set_facecolor('black')
        # ax.spines['bottom'].set_color('white')
        # ax.spines['top'].set_color('white') 
        # ax.spines['right'].set_color('white')
        # ax.spines['left'].set_color('white')
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
        # ax.yaxis.label.set_color('white')
        # ax.xaxis.label.set_color('white')
        # ax.title.set_color('white')

        gc_d = sf.getgc()

        for gc in gc_d.values():
            ax.plot(gc.r, gc.z, lw=.5, color='k')

        plot_qsurfaces(shotno, time, axes=ax)
        ax.plot([1.5, 2.275708], [-0.153213, 0.308526], lw=0.5, ls='--', c='grey', label='SPI1')
        ax.plot([1.5, 2.295673], [-0.071579, 0.324483], lw=0.5, ls='-', c='grey', label='SPI2')
        ax.plot([1.5, 2.271826], [0.0975929, 0.361589], lw=0.5, ls='-.', c='grey', label='SPI3')
        if multiplier == "raw":
            scale_array = np.zeros(len(res[:,2]))
            for i in range(len(scale_array)):
                if res[i,2] is None or res[i,2] < 1e11:
                    scale_array[i] = 0
                else:
                    scale_array[i] = ((np.log10(res[i,2]) - 11) ** 3) * 4
            scale_array[scale_array<1.5] = 0
            res[:,2] = res[:,2] / 1e11
            scatter = ax.scatter(res[:,0], res[:,1], s=scale_array, c=scale_array, cmap='inferno_r', zorder=10)
        else:
            scatter = ax.scatter(res[:,0], res[:,1], s=res[:,2]*multiplier, c=res[:,2], cmap='inferno_r', zorder=10)
        ax.plot(Rloc, zloc, '1', mew=2, ms=20, c="lime")
        ax.set_xlim(1, 2.6)
        ax.set_xlabel('R [m]')
        ax.set_ylim(-1.3, 1.3)
        ax.set_ylabel('z [m]')
        ax.set_title('#{} S-{} @ {:.6f} s'.format(shotno, sector, time))
        ax.legend(facecolor='white', markerscale=10, fontsize='small', labelcolor='k', framealpha=1, loc='lower right')
        if plot:
            cb = plt.colorbar(scatter, anchor=(0.0, 0.75), shrink=0.5)
            # cb.ax.yaxis.set_tick_params(color="white")
            # cb.outline.set_edgecolor("white")
            # plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")
            if save:
                savename = '_'.join([given_name, key, str(shotno)])
                plt.savefig(savename + ".png", dpi=150)
                plt.close(fig)
            else:
                plt.show()

    return grid_interp, points, values, Rloc, zloc

def radiation_inside_q2surface(shotno, start, end, sector=16):
    """Calculates the integral of the total multiplied intersection data AND inside q=2
    Returns a 3-element tuple: sum inside q=2, total sum, timesignal
    """
    if sector == 5:
        vertdiagname = DVC
        horizdiagname = DHC
        vertrange = VERT_S5_RANGE
        horizrange = HORIZ_S5_RANGE
    elif sector == 16:
        vertdiagname = D16
        horizdiagname = DHT
        vertrange = VERT_S16_RANGE
        horizrange = HORIZ_S16_RANGE

    _, _, datavert = ridge_filter(shotno, vertdiagname, start=start, end=end, signalrange=vertrange)
    _, timesignalh, datahoriz = ridge_filter(shotno, horizdiagname, start=start, end=end,
                                             signalrange=horizrange)
    
    multiplied = generate_isect_data3d(vertrange, horizrange, datavert, datahoriz)
    q2poly, sep_poly, equ = plot_qsurfaces(shotno, start, axes=None, diag="EQH")

    q2sum = np.zeros(len(timesignalh))
    totalsum = np.zeros(len(timesignalh))
    for i, thistime in enumerate(timesignalh):
        q2poly, sep_poly, _ = plot_qsurfaces(equ, thistime, axes=None)
        _, _, valuesq2, _, _ = radiation_poloidal(multiplied, timesignalh, time=thistime, 
                                                  shotno=shotno, sector=sector, save=False, 
                                                  plot=False, interp=None, polygon=q2poly)
        _, _, valuestotal, _, _ = radiation_poloidal(multiplied, timesignalh, time=thistime, 
                                                     shotno=shotno, sector=sector, save=False, 
                                                     plot=False, interp=None, polygon=sep_poly)
        q2sum[i] = np.sum(valuesq2)
        totalsum[i] = np.sum(valuestotal)

    return q2sum, totalsum, timesignalh

def radiation_inside_q2surface_2(shotno, start, end, sector=16):
    """Calculates the integral of the total multiplied intersection data AND inside q=2
    Returns a 3-element tuple: sum inside q=2, total sum, timesignal
    NOTE check why is this here as a second function
    """
    if sector == 5:
        vertdiagname = DVC
        horizdiagname = DHC
        vertrange = VERT_S5_RANGE
        horizrange = HORIZ_S5_RANGE
    elif sector == 16:
        vertdiagname = D16
        horizdiagname = DHT
        vertrange = VERT_S16_RANGE
        horizrange = HORIZ_S16_RANGE

    _, _, datavert = ridge_filter(shotno, vertdiagname, start=start, end=end, signalrange=vertrange)
    _, timesignalh, datahoriz = ridge_filter(shotno, horizdiagname, start=start, end=end,
                                             signalrange=horizrange)
    
    multiplied = np.sqrt(generate_isect_data3d(vertrange, horizrange, datavert, datahoriz))
    q2poly, sep_poly, equ = plot_qsurfaces(shotno, start, axes=None, diag="EQH")

    q2sum = np.zeros(len(timesignalh))
    totalsum = np.zeros(len(timesignalh))
    for i, thistime in enumerate(timesignalh):
        q2poly, sep_poly, _ = plot_qsurfaces(equ, thistime, axes=None)
        _, _, valuesq2, _, _ = radiation_poloidal(multiplied, timesignalh, time=thistime, 
                                                  shotno=shotno, sector=sector, save=False, 
                                                  plot=False, interp=None, polygon=q2poly)
        _, _, valuestotal, _, _ = radiation_poloidal(multiplied, timesignalh, time=thistime, 
                                                     shotno=shotno, sector=sector, save=False, 
                                                     plot=False, interp=None, polygon=sep_poly)
        q2sum[i] = np.sum(valuesq2)
        totalsum[i] = np.sum(valuestotal)

    return q2sum, totalsum, timesignalh

def animate_poloidal(data3d, timesignal, shotno, sector, 
                     maximum=True, step=1, multiplier=20, save=False, given_name=None):
    """Displays or saves an animation of scatter plots based on :data3d: with q-surfaces and SPI vectors
    :maximum: if True plots the maximum location based on interpolation
    :step: determines how many timestep difference should be between the frames of the animation
    :multiplier: scales the scatter dot sizes
    """
    imatfile = 'LOSisects.h5'
    f = h5py.File(imatfile,'r')
    key = 'S' + str(sector) 
    imat = f[key][()]
    f.close()
    fig, ax = plt.subplots(dpi=150, figsize=(4,6))
    fig.set_facecolor('black')
    ax.set_aspect(1)
    ax.set_facecolor('black')

    timeindex = 0
    time = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    x, y = ELLIPSE.exterior.xy
    res0 = []
    for i in range(48):
        res_temp = [p for p in isecs[i] if ELLIPSE.contains(geom.Point(p[:2]))]
        if len(res_temp) != 0:
            res0.append(res_temp)
    
    res = np.vstack(res0)

    # Make every axis and labels, etc white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    grid_R, grid_z = np.meshgrid(np.linspace(1, 2.2, 1200),
                                 np.linspace(-1, 1.1, 2200), indexing='ij')
    if maximum:
        points = np.array([res[:, 0], res[:, 1]]).T
        values = res[:, 2]
        grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')
        valueindex = np.unravel_index(np.nanargmax(grid_interp, axis=None), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        zloc = grid_z[0, valueindex[1]]

    lines = []
    gc_d = sf.getgc()
    for gc in gc_d.values():
        structure = ax.plot(gc.r, gc.z, lw=.5, color='white')
        lines.append(structure)

    _, equ = plot_qsurfaces(shotno, timesignal[-1], axes=ax)

    spi1 = ax.plot([1.5, 2.275708], [-0.153213, 0.308526], lw=0.5, ls='--', c='grey', label='SPI1')
    spi2 = ax.plot([1.5, 2.295673], [-0.071579, 0.324483], lw=0.5, ls='-', c='grey', label='SPI2')
    spi3 = ax.plot([1.5, 2.271826], [0.0975929, 0.361589], lw=0.5, ls='-.', c='grey', label='SPI3')
    lines.append(spi1)
    lines.append(spi2)
    lines.append(spi3)

    #norm = matplotlib.colors.Normalize(0, multiplier/2)
    scatter = plt.scatter(res[:,0], res[:,1], s=res[:,2]*multiplier, c=res[:,2], cmap='inferno')#, norm=norm)
    if maximum:
        ax.plot(Rloc, zloc, '1', mew=2, ms=20, c="lime")
    plt.xlim(1, 2.4)
    plt.xlabel('R [m]')
    plt.ylim(-1.3, 1.3)
    plt.ylabel('z [m]')
    title = plt.title('#{} S-{} @ {:.5f} s'.format(shotno, sector, time))
    axis = plt.gca()

    numframes = len(timesignal) // step
    anim = animation.FuncAnimation(fig, update_animate_poloidal, frames=range(numframes),
                                   fargs=(imat, data3d, timesignal, shotno, sector, lines, equ, maximum, grid_R, grid_z,
                                   axis, ELLIPSE, step, gc_d, multiplier), blit=False)

    if save:
        if step > 2:
            fps = 2
        else:
            fps = 5
        writervideo = animation.FFMpegWriter(fps=fps) 
        if given_name is None:
            given_name = 'plots/{}/{}'.format(shotno, key)
        anim.save(given_name + '_' + str(shotno) + '.mp4', writer=writervideo) 
        plt.close() 

    return anim

def update_animate_poloidal(i, imat, data3d, timesignal, shotno, sector, lines, equ, maximum, grid_R, grid_z,
                              axis, ellipse, step, gc_d, multiplier=20):
    """Updates the animation in animate_poloidal()"""
    timeindex = i * step
    time = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res0 = []
    for i in range(48):
        res_temp = [p for p in isecs[i] if ellipse.contains(geom.Point(p[:2]))]
        if len(res_temp) != 0:
            res0.append(res_temp)

    
    res = np.vstack(res0)
    axis.clear()
    
    for line in lines:
        axis.add_line(line[0])
    
    _, _ = plot_qsurfaces(equ, time, axes=axis)

    if maximum:
        points = np.array([res[:, 0], res[:, 1]]).T
        values = res[:, 2]
        grid_interp = griddata(points, values, (grid_R, grid_z), method='linear')
        valueindex = np.unravel_index(np.nanargmax(grid_interp, axis=None), grid_interp.shape)
        Rloc = grid_R[valueindex[0], 0]
        zloc = grid_z[0, valueindex[1]]

    #norm = matplotlib.colors.Normalize(0, multiplier/2)
    scatter = axis.scatter(res[:,0], res[:,1], s=res[:,2]*multiplier, c=res[:,2], cmap='inferno', zorder=10)#, norm=norm)
    if maximum:
        axis.plot(Rloc, zloc, '1', mew=2, ms=20, c="lime")
    axis.set_title('#{} S-{} @ {:.5f} s'.format(shotno, sector, time), color='white')
    axis.set_xlabel('R [m]', color='white')
    axis.set_ylabel('z [m]', color='white')
    axis.set_xlim(1, 2.4)
    axis.set_ylim(-1.3, 1.3)
    return scatter,

def interp_anim(data3d, timesignal, shotno, sector, multiplier=20, step=1):
    """Basic animation of the interpolated data based on :data3d:"""
    imatfile = 'LOSisects.h5'
    f = h5py.File(imatfile,'r')
    key = 'S' + str(sector) 
    imat = f[key][()]
    fig, ax = plt.subplots(dpi=150, figsize=(4,6))
    fig.set_facecolor('black')
    ax.set_aspect(1)
    ax.set_facecolor('black')

    timeindex = 0
    time = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :]
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    x, y = ELLIPSE.exterior.xy
    res0 = []
    for i in range(48):
        res_temp = [p for p in isecs[i] if ELLIPSE.contains(geom.Point(p[:2]))]
        if len(res_temp) != 0:
            res0.append(res_temp)
    
    res = np.vstack(res0)

    ############
    R_axis = np.linspace(1, 2.2, 600)
    z_axis = np.linspace(-1, 1.1, 1100)
    grid_R, grid_z = np.meshgrid(R_axis, z_axis, indexing='ij')
    points = np.array([res[:, 0], res[:, 1]]).T
    values = res[:, 2]
    grid_interp = griddata(points, values, (grid_R, grid_z), method="linear")
    valueindex = np.unravel_index(np.nanargmax(grid_interp, axis=None), grid_interp.shape)
    Rloc = grid_R[valueindex[0], 0]
    zloc = grid_z[0, valueindex[1]]

    # Make every axis and labels, etc white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    pcm = plt.pcolormesh(R_axis, z_axis, grid_interp.T)

    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    title = plt.title('# {} S-{} @ {:.4f} s'.format(shotno, sector, time))
    axis = plt.gca()

    numframes = len(timesignal) // step
    anim = animation.FuncAnimation(fig, update_interp_anim, frames=range(numframes),
                                   fargs=(data3d, timesignal, shotno, sector, grid_R, grid_z,
                                   axis, ELLIPSE, step, multiplier, R_axis, z_axis), blit=False)

    return anim

def update_interp_anim(i, data3d, timesignal, shotno, sector, grid_R, grid_z,
                       axis, ellipse, step, multiplier, R_axis, z_axis):
    """Updates interp_anim()"""
    imatfile = 'LOSisects.h5'
    f = h5py.File(imatfile,'r')
    key = 'S' + str(sector) 
    imat = f[key][()]
    f.close()
    timeindex = i * step
    time = timesignal[timeindex]

    isecs = np.zeros((48, 48, 3))
    isecs[:, :, 2] = data3d[timeindex, :, :] * multiplier
    for i in range(48):
        for j in range(48):
            isecs[i, j, :2] = imat[i, j]

    res0 = []
    for i in range(48):
        res_temp = [p for p in isecs[i] if ellipse.contains(geom.Point(p[:2]))]
        if len(res_temp) != 0:
            res0.append(res_temp)

    res = np.vstack(res0)

    ############
    points = np.array([res[:, 0], res[:, 1]]).T
    values = res[:, 2]
    grid_interp = griddata(points, values, (grid_R, grid_z), method="linear")
    valueindex = np.unravel_index(np.nanargmax(grid_interp, axis=None), grid_interp.shape)
    Rloc = grid_R[valueindex[0], 0]
    zloc = grid_z[0, valueindex[1]]

    axis.clear()

    pcm = axis.pcolormesh(R_axis, z_axis, grid_interp.T)

    axis.set_title('# {} S-{} @ {:.4f} s'.format(shotno, sector, time), color='white')
    axis.set_xlabel('R [m]', color='white')
    axis.set_ylabel('z [m]', color='white')
    axis.set_xlim(1, 2.4)
    axis.set_ylim(-1.3, 1.3)
    return pcm,

def plot_overview_1(shotno, start=2.3, end=2.4, vmin=1e4, vmax=1e8, save=False, vtimes=None, title=None,
                    sector=16, plot=False):
    """Plots or saves D16 or DVC signal, plasma current, integrated certical signals and radiation inside q=2

    :param shotno: int or str, mandatory
        AUG discharge number
    :param start: float, optional
        the experiment time from when to start data loading
    :param end: float, optional
        the experiment time until when to load data
    :param vmin: float, optional
        for pcolormesh
    :param vmax: float, optional
        for pcolormesh
    :param save: bool, optional
        to save or not to save
    :param vtimes: 2-element tuple or np.arr, optional
        plot vertical lines at these time coordinates
        for the determined current dip and peak times
    """
    shotno = str(shotno)    
    if sector == 16:
        diag = D16
    elif sector == 5:
        diag = DVC
    signalnames, timesignal, indices, lenofsignals, f = read_data_base(diag, start=start, end=end, shotno=shotno)
    numofsignals = len(signalnames)
    tickrange = np.arange(2, 49, 2)
    plotrange = range(1, 49)
    data2d = np.zeros([numofsignals, lenofsignals])
    
    for i in range(numofsignals):
        data2d[i, :] = f['/'.join([shotno, diag, signalnames[i]])][()][indices[0]:indices[1]]
        print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')
    
    if plot:
        fig = plt.figure(figsize=[16,9])
        if title is not None:
            fig.suptitle(title)

        subfigs = fig.subfigures(1, 2, wspace=0.05)

        subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 0.4])
        axsnest0 = subfigsnest[0].subplots(1, 1)

        im = axsnest0.pcolormesh(timesignal, plotrange, data2d, norm="log", vmin=vmin, vmax=vmax, zorder=1)
        cbar = plt.colorbar(im, ax=axsnest0)
        cbar.set_label('Diode signals [a. u.]')
        axsnest0.set_facecolor('black')
        axsnest0.set_yticks(tickrange)
        axsnest0.xaxis.grid(True, linestyle='--', zorder=5)
        axsnest0.set_ylabel('Channel number')

        subfigsnest[1].suptitle('Plasma current [kA]')
        axsnest1 = subfigsnest[1].subplots(1, 1)

        axsRight = subfigs[1].subplots(2, 1)

        fpc = sf.SFREAD(int(shotno), 'FPC', experiment='AUGD')
        Ip = fpc.getobject("IpiFP", cal=True)
        time = fpc.gettimebase("IpiFP")
        indices = calculate_indices(time, start, end)
        timesignal = time[indices[0]:indices[1]]
        datatoplot = Ip[indices[0]:indices[1]]/1000
        axsnest1.plot(timesignal, datatoplot)
        if vtimes is not None:
            (diptime, peaktime) = vtimes
            for axis in [axsnest0, axsnest1, axsRight[0], axsRight[1]]:
                axis.axvline(x=diptime, color='green', linestyle='--', label='Dip: {:.6f} s'.format(diptime),
                            linewidth=1, zorder=10)
                axis.axvline(x=peaktime, color='blue', linestyle='--', label='Peak: {:.6f} s'.format(peaktime),
                            linewidth=1, zorder=11)
            axsnest1.legend(loc='lower left')
        
        axsnest1.grid()
        axsnest1.set_xlabel('Time [s]')
        
        plot_integral_combined(shotno, array=VERT, yscale='linear', start=start, end=end, axes=axsRight[0])

    q2sum, totalsum, timesignal = radiation_inside_q2surface(shotno=shotno, start=start, end=end, sector=sector)
    fw = h5py.File('data_export/' + shotno + '_' + str(sector) + '_q2sum.h5','w' )
    fw.create_dataset('q2sum', data=q2sum)
    fw.create_dataset('total', data=totalsum)
    fw.create_dataset('time', data=timesignal)
    fw.close()

    if plot:
        axsRight[1].plot(timesignal, q2sum, label="q2")
        axsRight[1].plot(timesignal, totalsum, label="tot")
        axsRight[1].legend(loc="upper left")
        axsRight[1].set_ylabel("Multiplied raw signals")
        ax2 = axsRight[1].twinx()
        ax2.plot(timesignal, q2sum/totalsum, c='green')
        ax2.set_ylabel("inside q=2 / total", c='green')
        ax2.set_ylim(0, 1)
        axsRight[1].set_xlim(timesignal[0], timesignal[-1])
        
        if save:
            plt.savefig('spec_plots/overviews/' + shotno + ".png", dpi=300)
            plt.close(fig)
            print('Saved ' + shotno)
        else:
            plt.show()
    
    f.close()

def save_sqrt(shotno, start=2.3, end=2.4, sector=16):

    shotno = str(shotno)    
    if sector == 16:
        diag = D16
    elif sector == 5:
        diag = DVC
    signalnames, timesignal, indices, lenofsignals, f = read_data_base(diag, start=start, end=end, shotno=shotno)
    numofsignals = len(signalnames)
    tickrange = np.arange(2, 49, 2)
    plotrange = range(1, 49)
    data2d = np.zeros([numofsignals, lenofsignals])
    
    for i in range(numofsignals):
        data2d[i, :] = f['/'.join([shotno, diag, signalnames[i]])][()][indices[0]:indices[1]]
        print('{}/{} signals loaded'.format(i + 1, numofsignals), end='\r')

    q2sum, totalsum, timesignal = radiation_inside_q2surface_2(shotno=shotno, start=start, end=end, sector=sector)
    fw = h5py.File('data_export/' + shotno + '_' + str(sector) + '_q2sum_sqrt.h5','w' )
    fw.create_dataset('q2sum', data=q2sum)
    fw.create_dataset('total', data=totalsum)
    fw.create_dataset('time', data=timesignal)
    fw.close()
    f.close()

def get_intersecting_LOSs(shotno, t_in, diag1=D16, diag2=None, plot=False, savename=None, equ=None, get_data=False):
    shotno = int(shotno)
    if equ is None:
        equ = sf.EQU(shotno, diag="EQH")

    q2surf = sf.mapeq.get_q_surf(equ, qvalue=2, t_in=t_in, coord_out='rho_pol')
    sepcoords = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
    q2coords = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=q2surf, coord_in='rho_pol', all_lines=False)
    sep_polycoords = np.vstack((sepcoords[0][0][0], sepcoords[1][0][0])).T
    q2polycoords = np.vstack((q2coords[0][0][0], q2coords[1][0][0])).T
    try:
        q2poly = geom.Polygon(q2polycoords)
    except:
        q2poly = None
        print("NO q=2 SURFACE FOUND!")
        return equ, -3, -3, -3, -3

    try:
        sep_poly = geom.Polygon(sep_polycoords)
    except:
        sep_poly = None
        print("NO SEPARATRIX FOUND!")

    signals1 = SIGNAL_NAMES[diag1][1]
    startindex = np.where(LOS_DB["RAW"] == bytes(signals1[0], 'utf-8'))[0][0]
    crosses_separatrix = []
    crosses_q2 = []
    for signalname in signals1:
        los = LOS(signalname)
        if not los.intersects(sep_poly).is_empty:
            if not los.intersects(q2poly).is_empty:
                crosses_q2.append(los)
            else:
                crosses_separatrix.append(los)

    qs = crosses_q2[0].dbi - startindex
    qe = crosses_q2[-1].dbi - startindex
    ql = len(crosses_q2)
    ss = crosses_separatrix[0].dbi - startindex
    se = crosses_separatrix[-1].dbi - startindex
    sl = len(crosses_separatrix)
    legendq = [str(qs), '-', str(qe)]
    legends = [str(ss), '-', str(se)]

    if diag2 is not None:
        signals2 = SIGNAL_NAMES[diag2][1]
        startindex2 = np.where(LOS_DB["RAW"] == bytes(signals2[0], 'utf-8'))[0][0]
        for signalname in signals2:
            los = LOS(signalname)
            if not los.intersects(sep_poly).is_empty:
                if not los.intersects(q2poly).is_empty:
                    crosses_q2.append(los)
                else:
                    crosses_separatrix.append(los)

        qs2 = crosses_q2[ql].dbi - startindex2
        qe2 = crosses_q2[-1].dbi - startindex2
        ss2 = crosses_separatrix[sl].dbi - startindex2
        se2 = crosses_separatrix[-1].dbi - startindex2
        legendq.extend(['; ', str(qs2), '-', str(qe2)])
        legends.extend(['; ', str(ss2), '-', str(se2)])

    if plot:
        fig, ax = plt.subplots(dpi=150, figsize=(4,6))
        ax.set_aspect(1)

        gc_d = sf.getgc()
        for gc in gc_d.values():
            ax.plot(gc.r, gc.z, lw=.5, color='black')

        sep = ax.plot(sepcoords[0][0][0], sepcoords[1][0][0], alpha=1, ls='--', lw=1, label='LCFS idx: ' + ''.join(legends))
        q2 = ax.plot(q2coords[0][0][0], q2coords[1][0][0], alpha=1, ls='--', lw=1, label='q=2 idx: ' + ''.join(legendq))
        for los in crosses_separatrix:
            x, y = los.line.xy
            ax.plot(x, y, lw=0.5, c="green", ls=":")
        for los in crosses_q2:
            x, y = los.line.xy
            ax.plot(x, y, lw=0.5, c="red", ls=":")

        plt.xlim(1, 2.4)
        plt.xlabel('R [m]')
        plt.ylim(-1.3, 1.3)
        plt.ylabel('z [m]')
        plt.legend(loc="upper right")
        plt.title("#" + str(shotno) + " Sector 16")
        if savename is not None:
            plt.savefig(savename, dpi=150)
        else:
            plt.show()

        return legends
    else:
        if diag2 is None and get_data is False:
            return equ, crosses_q2[0].signalname, crosses_q2[-1].signalname, crosses_separatrix[0].signalname, crosses_separatrix[-1].signalname
        elif get_data is True:
            return equ, crosses_q2, crosses_separatrix

def animate_with_q2(shotno, diagname, start, end, steps=5, scale="linear", diag2=None, repair=False):
    when = start
    filename = 'smoothed_data/' + shotno + '_sm.h5'    
    f = h5py.File(filename, 'r')
    signalnames = list(f['/'.join([shotno, diagname])].keys())

    equ, firstsignal, lastsignal, sep1, sep2 = get_intersecting_LOSs(shotno, when, diag1=diagname, diag2=None,
                                                                     plot=False, equ=None)

    timesignalname = None
    for signal in signalnames:
        if '_time' in signal: timesignalname = signal

    timesignal = f['/'.join([shotno, diagname, timesignalname])][()]
    tsindex = signalnames.index(timesignalname)
    signalnames.pop(tsindex)

    firstch = signalnames.index(firstsignal + "_data") + 0.5
    lastch = signalnames.index(lastsignal + "_data") + 1.5
    sepch1 = signalnames.index(sep1 + "_data") + 0.5
    sepch2 = signalnames.index(sep2 + "_data") + 1.5

    numofsignals = len(signalnames)
    ti_start = bisect.bisect_left(timesignal, start)
    ti_end = bisect.bisect_left(timesignal, end) + 1
    timesignal = timesignal[ti_start:ti_end]
    lenofsignals = ti_end - ti_start
    data2d = np.zeros([numofsignals, lenofsignals])

    for i in range(numofsignals):
        data2d[i] = f['/'.join([shotno, diagname, signalnames[i]])][()][ti_start:ti_end]

    if repair:
        data2d = repair_2d_data(data2d)

    if diag2 is not None:
        signalnames2 = list(f['/'.join([shotno, diag2])].keys())
        equ2, firstsignal2, lastsignal2, sep21, sep22 = get_intersecting_LOSs(shotno, when, diag1=diag2, diag2=None,
                                                                              plot=False, equ=None)
        timesignalname2 = None
        for signal in signalnames2:
            if '_time' in signal: timesignalname2 = signal
        tsindex2 = signalnames2.index(timesignalname2)
        signalnames2.pop(tsindex2)
        numofsignals2 = len(signalnames2)
        data2d2 = np.zeros([numofsignals2, lenofsignals])
        for i in range(numofsignals2):
            data2d2[i] = f['/'.join([shotno, diag2, signalnames2[i]])][()][ti_start:ti_end]
        
        if repair:
            data2d2 = repair_2d_data(data2d2)

        firstch2 = signalnames2.index(firstsignal2 + "_data") + 0.5
        lastch2 = signalnames2.index(lastsignal2 + "_data") + 1.5
        sepch21 = signalnames2.index(sep21 + "_data") + 0.5
        sepch22 = signalnames2.index(sep22 + "_data") + 1.5

    f.close()

    if diag2 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=[6,4])
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12,4])
        l21 = ax2.step(range(1, numofsignals + 1), data2d.T[0], where='mid')
        l22 = ax2.axvline(firstch2, c="orange", lw=0.5, ls="--")
        l23 = ax2.axvline(lastch2, c="orange", lw=0.5, ls="--")
        l24 = ax2.axvline(sepch21, c="blue", lw=0.5, ls="--")
        l25 = ax2.axvline(sepch22, c="blue", lw=0.5, ls="--")
        title2 = ax2.set_title('{}'.format(diag2.split("_")[0]))
        ax2.set_xlim(1, numofsignals2 + 1)
        ax2.set_ylim(0, 1.05 * data2d2.max())
    l1 = ax1.step(range(1, numofsignals + 1), data2d.T[0], where='mid')
    l2 = ax1.axvline(firstch, c="orange", lw=0.5, ls="--", label='"q=2"')
    l3 = ax1.axvline(lastch, c="orange", lw=0.5, ls="--")
    l4 = ax1.axvline(sepch1, c="blue", lw=0.5, ls="--", label='"LCFS"')
    l5 = ax1.axvline(sepch2, c="blue", lw=0.5, ls="--")
    other = [firstch, lastch, sepch1, sepch2]
    if diag2 is None:
        lines = [l1]
    else:
        lines = [l1, l2, l3, l4, l5, l21, l22, l23, l24, l25]
    title = ax1.set_title('#{} {} @ {:.6f} s'.format(shotno, diagname.split("_")[0], when))
    legend = ax1.legend(loc="upper right")
    ax1.set_xlim(1, numofsignals + 1)
    ax1.set_ylim(0, 1.05 * data2d.max())
    
    if diag2 is not None:
        axes = (ax1, ax2)
    else: 
        axes = ax1
        equ2 = None
        signalnames2 = None
        diag2 = None
        data2d2 = None

    anim = animation.FuncAnimation(fig, update_with_q2, frames=range(int(np.floor(lenofsignals/steps))),
                                   fargs=(fig, shotno, data2d, diagname, equ, timesignal, signalnames, axes, steps, scale,
                                          equ2, signalnames2, diag2, data2d2, title, lines, other), 
                                   blit=False)

    plt.close(fig)
    return anim

def update_with_q2(index, fig, shotno, data2d, diagname, equ, timesignal, signalnames, axes, steps, scale, equ2=None, 
                   signalnames2=None, diag2=None, data2d2=None, title=None, lines=None, other=None):
    if not isinstance(axes, tuple):
        axes.clear()
        index = index * steps

        # firstch = signalnames.index(firstsignal + "_data") + 0.5
        # lastch = signalnames.index(lastsignal + "_data") + 1.5

        numofsignals = len(signalnames)

        axes.set_xlim(1, numofsignals + 1)
        if scale == "linear":
            axes.set_ylim(0, 1.05 * data2d.max())
        elif scale == "log":
            axes.set_ylim(1e6, 1e8)
            axes.set_yscale("log")
        
        l1 = axes.step(range(1, numofsignals + 1), data2d.T[index], where='mid')
        l2 = axes.axvline(other[0], c="orange", lw=0.5, ls="--", label='"q=2"')
        l3 = axes.axvline(other[1], c="orange", lw=0.5, ls="--")
        l4 = axes.axvline(other[2], c="blue", lw=0.5, ls="--", label='"LCFS"')
        l5 = axes.axvline(other[3], c="blue", lw=0.5, ls="--")
        lines = [l1, l2, l3]
        title = axes.set_title('#{} {} @ {:.6f} s'.format(shotno, diagname.split("_")[0], timesignal[index]))
        legend = axes.legend(loc="upper right")

    else:
        ax1, ax2 = axes
        index = index * steps

        numofsignals = len(signalnames)
        numofsignals2 = len(signalnames2)

        ax1.set_xlim(1, numofsignals + 1)
        ax2.set_xlim(1, numofsignals2 + 1)
        if scale == "linear":
            ax1.set_ylim(0, 1.05 * data2d.max())
            ax2.set_ylim(0, 1.05 * data2d2.max())
        elif scale == "log":
            ax1.set_ylim(1e6, 1e8)
            ax1.set_yscale("log")
            ax2.set_ylim(1e6, 1e8)
            ax2.set_yscale("log")

        lines[0][0].set_data(range(1, numofsignals + 1), data2d.T[index])
        # lines[1].set_data([firstch, firstch], [0, 1])
        # lines[2].set_data([lastch, lastch], [0, 1])
        # lines[3].set_data([sepch1, sepch1], [0, 1])
        # lines[4].set_data([sepch2, sepch2], [0, 1])
        lines[5][0].set_data(range(1, numofsignals + 1), data2d2.T[index])
        # lines[6].set_data([firstch2, firstch2], [0, 1])
        # lines[7].set_data([lastch2, lastch2], [0, 1])
        # lines[8].set_data([sepch21, sepch21], [0, 1])
        # lines[9].set_data([sepch22, sepch22], [0, 1])
        title.set_text('#{} {} @ {:.6f} s'.format(shotno, diagname.split("_")[0], timesignal[index]))

def sum_LOSs(shotno, start, end, diagname):
    shotno = str(shotno)
    filename = 'smoothed_data/' + shotno + '_sm.h5' 
    f = h5py.File(filename, 'r')
    signalnames = list(f['/'.join([shotno, diagname])].keys())

    equ, crq2, crsep = get_intersecting_LOSs(shotno, start, diag1=diagname, get_data=True)

    timesignalname = None
    for signal in signalnames:
        if '_time' in signal: timesignalname = signal

    timesignal = f['/'.join([shotno, diagname, timesignalname])][()]
    tsindex = signalnames.index(timesignalname)
    signalnames.pop(tsindex)

    numofsignals = len(signalnames)
    ti_start = bisect.bisect_left(timesignal, start)
    ti_end = bisect.bisect_left(timesignal, end) + 1
    timesignal = timesignal[ti_start:ti_end]
    lenofsignals = ti_end - ti_start
    core_rad = np.zeros(lenofsignals)
    edge_rad = np.zeros(lenofsignals)

    print(lenofsignals)

    for signal in crsep:
        edge_rad += f['/'.join([shotno, diagname, signal.signalname + "_data"])][()][ti_start:ti_end]
    for signal in crq2:
        core_rad += f['/'.join([shotno, diagname, signal.signalname + "_data"])][()][ti_start:ti_end]

    # for i in range(lenofsignals):
        # _, crq2, crsep = get_intersecting_LOSs(shotno, timesignal[i], diag1=diagname, get_data=True, equ=equ)
        # for signal in crsep:
        #     edge_rad[i] += f['/'.join([shotno, diagname, signal.signalname + "_data"])][()][i + ti_start]
        # for signal in crq2:
        #     core_rad[i] += f['/'.join([shotno, diagname, signal.signalname + "_data"])][()][i + ti_start]
        # print(str(i + 1) + " step  ", end='\r')

    return edge_rad, core_rad, timesignal

def save_LOS_sums(shotno, start, end, diptime, peaktime, title, save=True):
    f = h5py.File('data_export/LOS_sums/' + str(shotno) + '.h5', 'w')
    for diag in [D16, DHT]:
        diag_short = diag.split("_")[0]
        edge_rad, core_rad, timesignal = sum_LOSs(shotno, start, end, diag)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        ax.plot(timesignal, edge_rad, label="Sum of edge LOSs")
        ax.plot(timesignal, core_rad, label="Sum of core LOSs")

        ax.axvline(diptime, ls="--", lw=1, c="green", label="Current dip")
        ax.axvline(peaktime, ls="--", lw=1, c="blue", label="Current peak")

        ax.set_ylabel("Sum of LOS values in each timestep [a. u.]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper left")
        ax.set_ylim(0, None)
        ax.set_xlim(timesignal[0], None)
        ax.grid()

        ax2 = ax.twinx()
        ax2.set_title(title + " " + diag_short)

        ax2.set_xlim(timesignal[0], None)
        ax2.set_ylabel("Cumulative sum of LOS values [a. u.]")
        ax2.plot(timesignal, np.cumsum(edge_rad), label="Cumulative sum - edge", c="green")
        ax2.plot(timesignal, np.cumsum(core_rad), label="Cumulative sum - core", c="red")
        ax2.legend(loc="center left")
        ax2.set_ylim(0, None)
        if save:
            plt.savefig("spec_plots/LOS sums/" + str(shotno) + "_" + diag_short + ".png", dpi=150)
            plt.close()
        else:
            plt.show()

        f.create_dataset(diag_short +"_core", data=core_rad)
        f.create_dataset(diag_short +"_edge", data=edge_rad)
        f.create_dataset(diag_short +"_time", data=timesignal)

    f.close()

