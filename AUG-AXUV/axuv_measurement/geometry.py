"""
axuv/geometry.py  –  Line-of-sight geometry, q-surface utilities, and the
                     intersection-data generator.
"""
import h5py
import shapely
import numpy as np
import aug_sfutils as sf
import shapely.affinity as affinity
import shapely.geometry as geom
import matplotlib.pyplot as plt

from .config import (
    LOS_DB, DIAGNAMES_IN_S5_AND_S16, NEWAXUV,
    ELLIPSE_U, ELLIPSE_V, ELLIPSE_A, ELLIPSE_B,
    SIGNAL_NAMES, D16,
)


# ---------------------------------------------------------------------------
# Private helpers shared by several functions in this module and plotting.py
# ---------------------------------------------------------------------------

def _load_isect_matrix(sector: int) -> np.ndarray:
    """Return the (48, 48, 2) LOS-intersection coordinate matrix for *sector*.

    Reads from the static file ``LOSisects.h5`` which never changes at
    runtime, so callers are encouraged to load this once and pass it around
    rather than calling this function on every animation frame.
    """
    with h5py.File('LOSisects.h5', 'r') as f:
        return f['S' + str(sector)][()]


def _filter_by_polygon(isecs: np.ndarray, polygon) -> np.ndarray:
    """Return intersection points from *isecs* that fall inside *polygon*.

    :param isecs:   (48, 48, 3) array whose last axis holds (R, z, value).
    :param polygon: shapely Polygon used as the spatial mask.
    :returns:       (N, 3) array of the accepted points.
    """
    res0 = []
    for row in isecs:
        res_temp = [p for p in row if polygon.contains(geom.Point(p[:2]))]
        if res_temp:
            res0.append(res_temp)
    return np.vstack(res0)


# ---------------------------------------------------------------------------
# Line-of-sight class
# ---------------------------------------------------------------------------

class LOS:
    def __init__(self, signalname: str):
        self.dbi = np.where(LOS_DB["RAW"] == bytes(signalname, 'utf-8'))[0][0]
        diagname = LOS_DB["Cam"][self.dbi].decode('utf-8')
        for diag in DIAGNAMES_IN_S5_AND_S16:
            if diagname in diag:
                self.diagname = diag
        self.signalname = signalname
        self.Rstart = LOS_DB["R_start"][self.dbi]
        self.zstart = LOS_DB["z_start"][self.dbi]
        self.Rend   = LOS_DB["R_end"][self.dbi]
        self.zend   = LOS_DB["z_end"][self.dbi]
        self.midpoint_R = (self.Rstart + self.Rend) / 2
        self.length = LOS_DB["length"][self.dbi]
        self.vector = np.array(
            [(self.Rend - self.Rstart) ** 2, (self.zend - self.zstart) ** 2]
        ) / (self.length ** 2)
        # FIX: was geom.Point(self.Rstart, self.Rend) – wrong second coordinate
        self.startpoint = geom.Point(self.Rstart, self.zstart)
        line = geom.LineString([(self.Rstart, self.zstart), (self.Rend, self.zend)])
        self.line_0 = line
        self.line   = affinity.scale(line, xfact=2, yfact=2)

    def intersects(self, polygon):
        return shapely.intersection(self.line, polygon)

    def plot(self, color=None, lw: float = 0.5):
        plt.plot([self.Rstart, self.Rend], [self.zstart, self.zend], lw=lw, c=color)

    def load_data(self, file, shotno: str, ids=None, downsample: bool = True):
        """Load signal data from an open HDF5 file into ``self.data``.

        :param file:       open h5py.File instance
        :param shotno:     discharge number string (HDF5 group key)
        :param ids:        optional (start_index, end_index) slice
        :param downsample: if True, average-downsample to uniform time resolution
        """
        key = '/'.join([shotno, self.diagname, self.signalname + "_data"])
        if ids is not None:
            data = file[key][()][ids[0]:ids[1]]
        else:
            data = file[key][()]

        if downsample:
            factor = 10 if any(x == self.diagname for x in NEWAXUV) else 4
            newend = len(data) - (len(data) % factor)
            self.data = data[:newend].reshape(-1, factor).mean(axis=1)
        else:
            self.data = data

        # Initialise a uniform mask (all samples valid)
        self.mask = np.ones(len(self.data))

    def set_data(self, data: np.ndarray):
        self.data = data

    # FIX: was named 'mask', which clashed with the instance attribute of the
    # same name set in load_data() and permanently shadowed the method.
    def set_mask(self, values: np.ndarray):
        self.mask = values


# ---------------------------------------------------------------------------
# Poloidal-cross-section visualisation
# ---------------------------------------------------------------------------

def show_poloidal(sector: int = 16, vidx: int = 12, hidx: int = 35,
                  u: float = ELLIPSE_U, v: float = ELLIPSE_V,
                  a: float = ELLIPSE_A, b: float = ELLIPSE_B):
    """Show the LOS intersections for *sector*, a selected point, and the
    ellipse mask used to exclude points outside the vacuum vessel.
    """
    imat = _load_isect_matrix(sector)
    fig, ax = plt.subplots(dpi=150, figsize=(4, 6))
    ax.set_aspect(1)

    gc_d = sf.getgc()
    for gc in gc_d.values():
        ax.plot(gc.r, gc.z, lw=0.5, color='black')

    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(*imat.reshape(-1, 2).T, '.', ms=3, zorder=1)
    ax.plot(u + a * np.cos(t), v + b * np.sin(t))
    ax.scatter(imat[vidx, hidx][0], imat[vidx, hidx][1],
               s=5, color='orange', zorder=2)
    ax.set_xlim(1, 2.4)
    ax.set_xlabel('R [m]')
    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel('z [m]')
    ax.set_title('Sector ' + str(sector))
    plt.show()


# ---------------------------------------------------------------------------
# Equilibrium / q-surface utilities
# ---------------------------------------------------------------------------

def plot_qsurfaces(shot_or_equ, t_in, axes=None, diag: str = "EQH"):
    """Plot integer q-surfaces from equilibrium *diag* at time *t_in*.

    If *axes* is provided, plots the surfaces on that axes and returns
    ``(lines, equ)``.

    If *axes* is None, returns ``(q2poly, sep_poly, equ)`` — the q=2 and
    separatrix contours as shapely Polygons (or None if unavailable).

    :param shot_or_equ: shot number (int/str/float) or a pre-loaded sf.EQU object
    :param t_in:        time at which to evaluate the surfaces [s]
    :param axes:        matplotlib Axes or None
    :param diag:        equilibrium shotfile diagnostic name
    """
    if isinstance(shot_or_equ, (str, float, int)):
        equ = sf.EQU(int(shot_or_equ), diag=diag)
    elif isinstance(shot_or_equ, sf.EQU):
        equ = shot_or_equ
    else:
        raise TypeError(
            f"shot_or_equ must be a shot number (int/str/float) or sf.EQU, "
            f"got {type(shot_or_equ)}"
        )

    if axes is not None:
        i = 1
        qcoords = []
        qs = []
        lines = []
        temp = 1.0
        while not np.isnan(temp):
            i += 1
            temp = sf.mapeq.get_q_surf(equ, qvalue=i, t_in=t_in, coord_out='rho_pol')
            qcoords.append(temp)
            qs.append(i)

        qsurfaces = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=qcoords,
                                     coord_in='rho_pol', all_lines=False)
        sep_Rz = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
        sep = axes.plot(sep_Rz[0][0][0], sep_Rz[1][0][0],
                        alpha=0.7, ls='--', lw=1, label='LCFS')
        lines.append(sep)

        for i in range(len(qsurfaces[0][0]) - 1):
            surf = axes.plot(qsurfaces[0][0][i], qsurfaces[1][0][i],
                             alpha=0.7, ls='-.', lw=0.5, label='q={}'.format(qs[i]))
            lines.append(surf)

        return lines, equ

    else:
        q2surf    = sf.mapeq.get_q_surf(equ, qvalue=2, t_in=t_in, coord_out='rho_pol')
        sepcoords = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
        q2coords  = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=q2surf,
                                     coord_in='rho_pol', all_lines=False)
        sep_polycoords = np.vstack((sepcoords[0][0][0], sepcoords[1][0][0])).T
        q2polycoords   = np.vstack((q2coords[0][0][0],  q2coords[1][0][0])).T

        try:
            q2poly = geom.Polygon(q2polycoords)
        except Exception:
            q2poly = None

        try:
            sep_poly = geom.Polygon(sep_polycoords)
        except Exception:
            sep_poly = None

        return q2poly, sep_poly, equ


# ---------------------------------------------------------------------------
# Intersection-data generator
# ---------------------------------------------------------------------------

def generate_isect_data3d(vertrange, horizrange, datavert, datahoriz,
                           normalize=None, dontmultiply: bool = False):
    """Generate a 3-D (T × 48 × 48) intersection-value array.

    For every timestep *t* and every pair of LOS indices *(i, j)* the entry
    is set to ``datavert[t, i] * datahoriz[t, j]`` (default) or
    ``datavert[t, i] + datahoriz[t, j]`` when *dontmultiply* is True.

    The intended input is the output of ``ridge_filter()``.  It is important
    to pass the correct *vertrange* / *horizrange* so the data columns map to
    the right 48-channel positions.

    :param vertrange:    [first_ch, last_ch] for the vertical diagnostic
    :param horizrange:   [first_ch, last_ch] for the horizontal diagnostic
    :param datavert:     (T, N_vert) array
    :param datahoriz:    (T, N_horiz) array
    :param normalize:    pass ``'horiz'`` to normalise the horizontal row sums to 1
    :param dontmultiply: if True, sum the two arrays instead of multiplying
    :returns:            (T, 48, 48) array, or zeros on dimension mismatch
    """
    vlen = datavert.shape[0]
    multiplied = np.zeros((vlen, 48, 48))

    if normalize is not None and normalize != 'horiz':
        print("Currently only 'horiz' is a valid argument for normalize")
        return multiplied

    if vlen != datahoriz.shape[0]:
        print('ERROR: data arrays are not the same length in timesteps')
        return multiplied

    # Place data into the full 48-channel matrices
    matrixv = np.zeros((vlen, 48))
    matrixh = np.zeros((vlen, 48))

    for i, vertidx in enumerate(range(vertrange[0], vertrange[1])):
        matrixv[:, vertidx] = datavert[:, i]

    for j, horidx in enumerate(range(horizrange[0], horizrange[1])):
        matrixh[:, horidx] = datahoriz[:, j]

    # Optionally normalise the horizontal direction so that the sum equals 1
    if normalize == 'horiz':
        row_sums = np.sum(matrixh, axis=1)
        matrixh = matrixh / row_sums[:, np.newaxis]

    # Vectorised outer product / sum over all timesteps using NumPy broadcasting:
    # shape (T, 48, 1) op (T, 1, 48)  →  (T, 48, 48)
    if dontmultiply:
        multiplied = matrixv[:, :, np.newaxis] + matrixh[:, np.newaxis, :]
    else:
        multiplied = matrixv[:, :, np.newaxis] * matrixh[:, np.newaxis, :]

    return multiplied


# ---------------------------------------------------------------------------
# LOS-to-surface intersection finder
# ---------------------------------------------------------------------------

def get_intersecting_LOSs(shotno, t_in, diag1=D16, diag2=None,
                           plot: bool = False, savename=None,
                           equ=None, get_data: bool = False):
    """Find which LOSs of *diag1* (and optionally *diag2*) cross the q=2
    surface and the separatrix at time *t_in*.

    When *plot* is True, draws the geometry on a new figure and returns the
    legend strings.

    When *plot* is False:
      - if *diag2* is None and *get_data* is False:
          returns ``(equ, first_q2_sig, last_q2_sig, first_sep_sig, last_sep_sig)``
      - if *get_data* is True:
          returns ``(equ, crosses_q2_list, crosses_sep_list)``
    """
    shotno = int(shotno)
    if equ is None:
        equ = sf.EQU(shotno, diag="EQH")

    q2surf    = sf.mapeq.get_q_surf(equ, qvalue=2, t_in=t_in, coord_out='rho_pol')
    sepcoords = sf.rho2rz(equ, 1, t_in=t_in, coord_in='rho_pol')
    q2coords  = sf.mapeq.rho2rz(equ, t_in=t_in, rho_in=q2surf,
                                  coord_in='rho_pol', all_lines=False)

    sep_polycoords = np.vstack((sepcoords[0][0][0], sepcoords[1][0][0])).T
    q2polycoords   = np.vstack((q2coords[0][0][0],  q2coords[1][0][0])).T

    try:
        q2poly = geom.Polygon(q2polycoords)
    except Exception:
        q2poly = None
        print("NO q=2 SURFACE FOUND!")
        return equ, -3, -3, -3, -3

    try:
        sep_poly = geom.Polygon(sep_polycoords)
    except Exception:
        sep_poly = None
        print("NO SEPARATRIX FOUND!")

    signals1   = SIGNAL_NAMES[diag1][1]
    startindex = np.where(LOS_DB["RAW"] == bytes(signals1[0], 'utf-8'))[0][0]

    crosses_separatrix: list = []
    crosses_q2: list         = []

    for signalname in signals1:
        los = LOS(signalname)
        if not los.intersects(sep_poly).is_empty:
            if not los.intersects(q2poly).is_empty:
                crosses_q2.append(los)
            else:
                crosses_separatrix.append(los)

    qs  = crosses_q2[0].dbi - startindex
    qe  = crosses_q2[-1].dbi - startindex
    ql  = len(crosses_q2)
    ss  = crosses_separatrix[0].dbi - startindex
    se  = crosses_separatrix[-1].dbi - startindex
    sl  = len(crosses_separatrix)

    legendq = [str(qs), '-', str(qe)]
    legends = [str(ss), '-', str(se)]

    if diag2 is not None:
        signals2   = SIGNAL_NAMES[diag2][1]
        startindex2 = np.where(LOS_DB["RAW"] == bytes(signals2[0], 'utf-8'))[0][0]

        for signalname in signals2:
            los = LOS(signalname)
            if not los.intersects(sep_poly).is_empty:
                if not los.intersects(q2poly).is_empty:
                    crosses_q2.append(los)
                else:
                    crosses_separatrix.append(los)

        qs2 = crosses_q2[ql].dbi  - startindex2
        qe2 = crosses_q2[-1].dbi  - startindex2
        ss2 = crosses_separatrix[sl].dbi  - startindex2
        se2 = crosses_separatrix[-1].dbi  - startindex2

        legendq.extend(['; ', str(qs2), '-', str(qe2)])
        legends.extend(['; ', str(ss2), '-', str(se2)])

    if plot:
        fig, ax = plt.subplots(dpi=150, figsize=(4, 6))
        ax.set_aspect(1)

        gc_d = sf.getgc()
        for gc in gc_d.values():
            ax.plot(gc.r, gc.z, lw=0.5, color='black')

        ax.plot(sepcoords[0][0][0], sepcoords[1][0][0],
                alpha=1, ls='--', lw=1, label='LCFS idx: ' + ''.join(legends))
        ax.plot(q2coords[0][0][0],  q2coords[1][0][0],
                alpha=1, ls='--', lw=1, label='q=2 idx: '  + ''.join(legendq))

        for los in crosses_separatrix:
            x, y = los.line.xy
            ax.plot(x, y, lw=0.5, c='green', ls=':')
        for los in crosses_q2:
            x, y = los.line.xy
            ax.plot(x, y, lw=0.5, c='red', ls=':')

        ax.set_xlim(1, 2.4)
        ax.set_xlabel('R [m]')
        ax.set_ylim(-1.3, 1.3)
        ax.set_ylabel('z [m]')
        ax.legend(loc='upper right')
        ax.set_title('#' + str(shotno) + ' Sector 16')

        if savename is not None:
            plt.savefig(savename, dpi=150)
        else:
            plt.show()

        return legends

    else:
        if diag2 is None and not get_data:
            return (equ,
                    crosses_q2[0].signalname,
                    crosses_q2[-1].signalname,
                    crosses_separatrix[0].signalname,
                    crosses_separatrix[-1].signalname)
        elif get_data:
            return equ, crosses_q2, crosses_separatrix