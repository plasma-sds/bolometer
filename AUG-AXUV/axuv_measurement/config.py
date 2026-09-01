"""
axuv/config.py  –  constants, diagnostic metadata, and pre-loaded databases.

All other axuv sub-modules import from here; nothing in this file imports
from the rest of the axuv package.
"""
import numpy as np
import shapely.geometry as geom

ROOTFOLDER = "/shares/departments/AUG/users/lefer/AXUV/"


# ---------------------------------------------------------------------------
# Signal-name generator
# ---------------------------------------------------------------------------

def _make_signals(sector: int, spec: list) -> list:
    """Return a list of AXUV signal names for *sector*.

    :param sector: integer used as the sector prefix (e.g. 6  ->  'S6…')
    :param spec:   list of ``(lane, first_channel, last_channel)`` tuples,
                   all bounds inclusive.
    """
    result = []
    for lane, start, end in spec:
        result.extend(f"S{sector}L{lane}A{i:02d}" for i in range(start, end + 1))
    return result


# ---------------------------------------------------------------------------
# Signal-name dictionary   { key: [shotfile_diagnostic, [sig1, sig2, …]] }
# ---------------------------------------------------------------------------

SIGNAL_NAMES: dict = {
    # 3 lanes × 16 channels  =  48 signals each
    "DVC_S5_vert":   ["XVR", _make_signals(0, [(0, 0, 15), (1, 0, 15), (2, 0, 15)])],
    "DHC_S5_horiz":  ["XVR", _make_signals(1, [(0, 0, 15), (1, 0, 15), (2, 0, 15)])],
    "D13_S13_vert":  ["XVS", _make_signals(3, [(0, 0, 15), (1, 0, 15), (2, 0, 15)])],
    # Lane 0 = 32 ch, lane 1 = 16 ch  (or split across two diagnostics)
    "D01_S1_vert":   ["XVU", _make_signals(6, [(0, 0, 31), (1, 0, 15)])],
    "D16_S16_vert":  ["XVU", _make_signals(6, [(1, 16, 31), (2, 0, 31)])],
    "DHT_S16_horiz": ["XVU", _make_signals(7, [(1, 16, 31), (2, 0, 31)])],
    "D15_S15_vert":  ["XVU", _make_signals(7, [(0, 0, 31), (1, 0, 15)])],
}

SIG_KEY_LIST: list = list(SIGNAL_NAMES.keys())


# ---------------------------------------------------------------------------
# Diagnostic name aliases
# ---------------------------------------------------------------------------

D15 = "D15_S15_vert"    # Sector 15 vertical   – some channels problematic, mostly not used
D16 = "D16_S16_vert"    # Sector 16 vertical   – at SPI location; poloidal cross-section mappable
D01 = "D01_S1_vert"     # Sector 01 vertical   – CCW neighbour of S16
DVC = "DVC_S5_vert"     # Sector 05 vertical   – poloidal cross-section mappable
D13 = "D13_S13_vert"    # Sector 13 vertical
DHC = "DHC_S5_horiz"    # Sector 05 horizontal – poloidal cross-section mappable
DHT = "DHT_S16_horiz"   # Sector 16 horizontal – poloidal cross-section mappable

# Categorisation by orientation
VERT:                list = [D16, D01, DVC, D13]
HORIZ:               list = [DHT, DHC]
DIAGNAMES_IN_S5_AND_S16: list = [D16, DHT, DVC, DHC]
NEWAXUV:             list = [D16, D01, DHT]   # diagnostics with higher time resolution


# ---------------------------------------------------------------------------
# Default channel index ranges for the poloidal cross-section analysis
# ---------------------------------------------------------------------------

VERT_S5_RANGE:   list = [0, 47]
HORIZ_S5_RANGE:  list = [3, 47]
VERT_S16_RANGE:  list = [0, 29]
HORIZ_S16_RANGE: list = [3, 44]


# ---------------------------------------------------------------------------
# Ellipse used to mask data outside the vacuum vessel in the poloidal plane
# ---------------------------------------------------------------------------

ELLIPSE_U: float = 1.59   # centre R-coordinate  [m]
ELLIPSE_V: float = 0.11   # centre z-coordinate  [m]
ELLIPSE_A: float = 0.58   # R semi-axis          [m]
ELLIPSE_B: float = 1.03   # z semi-axis          [m]

ELLIPSE_T  = np.linspace(0, 2 * np.pi, 100)
ELLIPSE_R  = (ELLIPSE_A * ELLIPSE_B) / np.sqrt(
    (ELLIPSE_B * np.cos(ELLIPSE_T)) ** 2 + (ELLIPSE_A * np.sin(ELLIPSE_T)) ** 2
)
ELLIPSE_XY = np.stack(
    [ELLIPSE_U + ELLIPSE_R * np.cos(ELLIPSE_T),
     ELLIPSE_V + ELLIPSE_R * np.sin(ELLIPSE_T)],
    axis=1,
)
ELLIPSE = geom.Polygon(ELLIPSE_XY)


# ---------------------------------------------------------------------------
# Pre-loaded CSV databases (load failures are non-fatal; attribute set to None)
# ---------------------------------------------------------------------------

try:
    SPI_DB = np.genfromtxt(ROOTFOLDER + "CSVs/pellets.csv", delimiter=",")
except Exception as e:
    print(f'\033[31mWARNING! Could not read pellets.csv: {e}\033[0m')
    SPI_DB = None

SPI_COLUMN_NAMES = [
    "#", " Ne% ", " GT-", " SpeedA ", " first light ", " Delay ",
    " SpeedM ", " SpeedML ", " SpeedMU ", " dSpeedM ", " Diameter ", " Mode ",
]

try:
    CURRENT_DB = np.genfromtxt(ROOTFOLDER + "CSVs/startofincline.csv", delimiter=",")
except Exception as e:
    print(f'\033[31mWARNING! Could not read "current" database: {e}\033[0m')
    CURRENT_DB = None

try:
    DESUBLIM_DB = np.genfromtxt(ROOTFOLDER + "CSVs/desublimation.csv", delimiter=",")
except Exception as e:
    print(f'\033[31mWARNING! Could not read desublimation.csv: {e}\033[0m')
    DESUBLIM_DB = None

try:
    FULL_DB = np.genfromtxt(
        ROOTFOLDER + "CSVs/db_with_sqrt.csv",
        delimiter=",", names=True,
        dtype="i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8",
    )
except Exception as e:
    print(f'\033[31mWARNING! Could not read comprehensive database: {e}\033[0m')
    FULL_DB = None

try:
    LOS_DB = np.genfromtxt(
        ROOTFOLDER + "CSVs/AXUV_LOSs.csv",
        delimiter=",", names=True,
        dtype="S3,i8,i8,i8,S7,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8",
    )
except Exception as e:
    print(f'\033[31mWARNING! Could not read AXUV LOS file: {e}\033[0m')
    LOS_DB = None