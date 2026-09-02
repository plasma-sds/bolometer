import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Poloidal cross-section extent used for the rectangular interpolation grid.
# These values define the cell *centres* of the outermost voxels, consistent
# with raytransfer_sensitivity._build_voxel_grid which computes cell vertices
# by extending +-0.5 delta beyond these endpoints.
POLOIDAL_RMIN = 1.0
POLOIDAL_RMAX = 2.2
POLOIDAL_ZMIN = -1.2
POLOIDAL_ZMAX = 1.0

# Default supersampling factor.  The irregular source data are first projected
# onto a grid `_SUPERSAMPLE` times finer than the target grid in each
# dimension, a Gaussian low-pass filter is applied, and the result is
# downsampled to the target resolution.
_SUPERSAMPLE = 4

# A coarse grid cell is assigned a non-zero plasma value only when this
# fraction of its Gaussian-kernel area overlaps the convex hull of the source
# data.  Below the threshold the cell is considered outside the plasma and is
# set to *exactly* 0.0.  A value of 0.5 permits at most one Gaussian sigma
# (~0.5 coarse cells) of boundary tolerance while keeping all clearly
# out-of-plasma cells strictly zero.
_HULL_COVERAGE_THRESHOLD = 0.5


def interpolate_param(
    param, points, grid_R, grid_z, method="linear", supersample=_SUPERSAMPLE
):
    """Interpolate a single parameter from the JOREK irregular grid onto a
    rectangular grid using supersampling and Gaussian anti-aliasing.

    Physical motivation
    -------------------
    The JOREK grid is an irregular set of (R, z) points in the poloidal
    cross-section.  When that data is denser than the target rectangular grid,
    direct ``griddata`` interpolation can alias high-frequency spatial content.
    Supersampling followed by Gaussian low-pass filtering avoids this.

    Boundary treatment
    ------------------
    Cells that lie outside the convex hull of the source data contain *no
    plasma*.  They must remain exactly zero — introducing artificial values
    through nearest-neighbour extrapolation or letting the Gaussian kernel
    smear real plasma values into the void would be unphysical.

    This is achieved via *normalized Gaussian convolution*:

    1. Project the irregular data onto a fine grid (``griddata``).
       Points outside the convex hull receive NaN.
    2. Build a binary coverage mask (1 inside the hull, 0 outside) and
       zero-fill the NaN data cells — no extrapolation, no invented values.
    3. Apply the same Gaussian filter to both the data array and the mask.
    4. Divide filtered-data by filtered-mask.  This renormalizes boundary cells
       so that only real plasma contributions are counted.
    5. Any cell whose filtered mask falls below ``_HULL_COVERAGE_THRESHOLD``
       (i.e., less than half its kernel area came from inside the plasma) is
       set to exactly 0.0.
    6. Stride-downsample the result to the coarse target resolution.

    Fine-grid sizing
    ----------------
    The fine grid uses ``(n - 1) * supersample + 1`` points along each axis.
    This guarantees that striding by ``supersample`` recovers *exactly* the
    coarse cell-centre positions (``np.linspace(R_min, R_max, nR)``).  Using
    ``n * supersample`` would silently shift every sample point, introducing
    a systematic positional error proportional to the coarse cell width.

    Gaussian sigma
    --------------
    ``sigma = supersample / 2`` fine-grid pixels corresponds to smoothing over
    half the width of one coarse cell.  This attenuates spatial frequencies
    above the Nyquist limit of the coarse grid (the standard anti-aliasing
    criterion for downsampling by factor ``supersample``).

    Parameters
    ----------
    param : array_like, shape (N,) or (N, 1)
        Scalar field values at each of the N irregular source points.
    points : array_like, shape (N, 2)
        (R, z) coordinates of the N source points in metres.
    grid_R, grid_z : ndarray, shape (nz, nR)
        Output meshgrid arrays, as returned by
        ``np.meshgrid(linspace_R, linspace_z)``.  These define the cell
        *centres* of the coarse rectangular target grid.
    method : str
        Interpolation method forwarded to ``scipy.griddata``
        (``'linear'``, ``'nearest'``, or ``'cubic'``).
    supersample : int
        Oversampling factor for the intermediate fine grid.  Must be >= 1.
        ``supersample=1`` disables supersampling (Gaussian sigma = 0.5, which
        is effectively an identity transform).

    Returns
    -------
    ndarray, shape (nR, nz)
        Interpolated field on the coarse rectangular grid.  The first axis
        indexes R (radial), the second indexes z (vertical), matching the
        storage convention of ``interpolate_parameters``.
    """
    param_1d = np.asarray(param).ravel()

    # meshgrid(linspace_R, linspace_z) → shape (nz, nR)
    nz, nR = grid_R.shape
    R_min, R_max = float(grid_R[0, 0]), float(grid_R[0, -1])
    z_min, z_max = float(grid_z[0, 0]), float(grid_z[-1, 0])

    # ── Step 1: project onto fine rectangular grid ────────────────────────────
    # (n-1)*k + 1 points: striding the result by k gives exactly the nR (resp.
    # nz) coarse cell-centre positions, no positional offset.
    fine_R, fine_z = np.meshgrid(
        np.linspace(R_min, R_max, (nR - 1) * supersample + 1),
        np.linspace(z_min, z_max, (nz - 1) * supersample + 1),
    )
    result_fine = griddata(points, param_1d, (fine_R, fine_z), method=method)

    # ── Step 2: build coverage mask; zero-fill outside hull ───────────────────
    # griddata returns NaN for query points outside the convex hull of `points`.
    # We record where data is valid, then set NaN → 0 without extrapolating.
    valid = ~np.isnan(result_fine)
    data = np.where(valid, result_fine, 0.0)
    mask = valid.astype(np.float64)  # 1.0 inside hull, 0.0 outside

    # ── Step 3 & 4: normalized Gaussian convolution ───────────────────────────
    # Filter data and mask independently, then renormalize.
    #
    # For a cell well inside the hull:   mask_filtered ≈ 1  → result ≈ data_filtered
    # For a cell at the hull boundary:   mask_filtered < 1  → renormalized value
    # For a cell far outside the hull:   mask_filtered → 0  → set to 0.0
    #
    # This correctly handles the boundary without smearing plasma values into
    # the vacuum.
    sigma = supersample / 2.0
    data_filtered = gaussian_filter(data, sigma=sigma)
    mask_filtered = gaussian_filter(mask, sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            mask_filtered >= _HULL_COVERAGE_THRESHOLD,
            data_filtered / mask_filtered,
            0.0,
        )

    # ── Step 5: downsample to coarse target resolution ────────────────────────
    result_coarse = result[::supersample, ::supersample]

    return result_coarse.T  # → (nR, nz)


def interpolate_parameters(
    points,
    resolution_R,
    resolution_z,
    neonlist,
    eTemp,
    eDens,
    method="linear",
    supersample=_SUPERSAMPLE,
):
    """Interpolate all plasma parameters from the JOREK irregular grid onto a
    rectangular ``(resolution_R × resolution_z)`` grid.

    Each parameter is processed independently by :func:`interpolate_param`,
    which uses supersampling and normalized Gaussian anti-aliasing.  Grid cells
    that lie outside the convex hull of the source data (i.e., where there is
    no plasma) are set to exactly 0.0.

    Grid convention
    ---------------
    The ``resolution_R`` output points along R are the cell *centres* of the
    raytransfer voxels, placed at ``np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX,
    resolution_R)`` — consistent with ``raytransfer_sensitivity._build_voxel_grid``,
    which derives cell vertices by extending ±0.5·Δ beyond these endpoints.
    The same convention applies along z.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        (R, z) coordinates of the N JOREK grid points in metres.
    resolution_R : int
        Number of voxel columns in the radial (R) direction.
    resolution_z : int
        Number of voxel rows in the vertical (z) direction.
    neonlist : list of array_like
        Exactly 11 arrays of shape (N,), one per neon charge state (Ne0–Ne10).
    eTemp : array_like, shape (N,)
        Electron temperature at each source point.
    eDens : array_like, shape (N,)
        Electron density at each source point.
    method : str
        Interpolation method forwarded to ``scipy.griddata``.
    supersample : int
        Supersampling factor forwarded to :func:`interpolate_param`.

    Returns
    -------
    interpolated_neon : ndarray, shape (11, resolution_R, resolution_z)
    interpolated_eTemp : ndarray, shape (resolution_R, resolution_z)
    interpolated_eDens : ndarray, shape (resolution_R, resolution_z)
    """
    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)
    grid_R, grid_z = np.meshgrid(linspace_R, linspace_z)

    print("Interpolating densities and temperatures...")
    interpolated_neon = np.zeros((11, resolution_R, resolution_z))
    for i, neon in enumerate(neonlist):
        interpolated_neon[i] = interpolate_param(
            neon, points, grid_R, grid_z, method, supersample
        )

    interpolated_eTemp = interpolate_param(
        eTemp, points, grid_R, grid_z, method, supersample
    )
    interpolated_eDens = interpolate_param(
        eDens, points, grid_R, grid_z, method, supersample
    )
    return interpolated_neon, interpolated_eTemp, interpolated_eDens
