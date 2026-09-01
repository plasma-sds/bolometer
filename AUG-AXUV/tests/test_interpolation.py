"""Unit tests for axuv.interpolation."""

import numpy as np
import pytest

from axuv.interpolation import (
    POLOIDAL_RMAX,
    POLOIDAL_RMIN,
    POLOIDAL_ZMAX,
    POLOIDAL_ZMIN,
    interpolate_param,
    interpolate_parameters,
)


# ── Constants sanity ──────────────────────────────────────────────────────────
def test_poloidal_constants_ordering():
    """Grid extent constants must form a valid (min < max) rectangle."""
    assert POLOIDAL_RMIN < POLOIDAL_RMAX
    assert POLOIDAL_ZMIN < POLOIDAL_ZMAX


def test_poloidal_constants_positive_r():
    """R coordinates must be strictly positive (tokamak major radius)."""
    assert POLOIDAL_RMIN > 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_uniform_points(n=200, seed=42):
    """Return n uniform random (R, z) points within the poloidal domain.

    The four domain corners are always appended so the convex hull exactly
    spans the full rectangular grid.  Without them, ``np.random.uniform``
    (open upper bound) means the extreme corners can fall just outside the
    hull, producing spurious zeros in constant-field tests.
    """
    rng = np.random.default_rng(seed)
    R = rng.uniform(POLOIDAL_RMIN, POLOIDAL_RMAX, n)
    z = rng.uniform(POLOIDAL_ZMIN, POLOIDAL_ZMAX, n)
    # Pin the four corners so the hull is exactly [RMIN, RMAX] × [ZMIN, ZMAX].
    R = np.concatenate(
        [R, [POLOIDAL_RMIN, POLOIDAL_RMAX, POLOIDAL_RMIN, POLOIDAL_RMAX]]
    )
    z = np.concatenate(
        [z, [POLOIDAL_ZMIN, POLOIDAL_ZMIN, POLOIDAL_ZMAX, POLOIDAL_ZMAX]]
    )
    return np.column_stack([R, z])


# ── interpolate_param ─────────────────────────────────────────────────────────
def test_interpolate_param_uniform_data():
    """Interpolating a constant field must reproduce that constant everywhere.

    Because the domain corners are pinned in _make_uniform_points the convex
    hull covers the full grid, so every coarse cell has a Gaussian-kernel
    coverage fraction of 1.0 and must return the input constant (7.0).
    """
    points = _make_uniform_points()
    param = np.ones(len(points))[:, np.newaxis] * 7.0

    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, 5),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, 5),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    np.testing.assert_allclose(result, 7.0, atol=1e-5)


def test_interpolate_param_output_shape():
    """Output shape must be (nR, nz) — R axis first, z axis second.

    ``np.meshgrid(linspace_R, linspace_z)`` returns arrays of shape (nz, nR).
    ``interpolate_param`` transposes the downsampled fine grid so the caller
    always gets the R-major layout used by ``interpolate_parameters``.
    """
    n = 100
    points = _make_uniform_points(n)
    param = np.ones(len(points))[:, np.newaxis]

    nr, nz = 8, 12
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nr),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, nz),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    assert result.shape == (nr, nz)


def test_interpolate_param_no_nan_in_output():
    """The result must contain no NaN values.

    The normalized Gaussian convolution replaces NaN (griddata's marker for
    outside-hull cells) with 0.0 via a coverage-mask division guarded by a
    threshold, so NaN can never appear in the output.
    """
    points = _make_uniform_points()
    param = np.ones(len(points))[:, np.newaxis]
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, 6),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, 6),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    assert not np.any(np.isnan(result))


def test_interpolate_param_linear_field():
    """A linearly varying field must be reproduced accurately at well-interior
    grid cells after supersampling + Gaussian anti-aliasing.

    A Gaussian filter preserves linear (affine) functions exactly in the
    interior of a domain where the coverage mask is uniformly 1.  The domain
    corners are pinned so the hull spans the full grid and the mask is indeed
    1 everywhere.

    ``scipy.ndimage.gaussian_filter`` uses 'reflect' boundary conditions, which
    introduces artefacts within roughly 2–3 sigma (≈ 1–2 coarse cells) of the
    domain edge.  We therefore test only the strictly interior region,
    skipping the 2 outermost coarse cells on each side.
    """
    rng = np.random.default_rng(7)
    n = 500
    R = rng.uniform(POLOIDAL_RMIN, POLOIDAL_RMAX, n)
    z = rng.uniform(POLOIDAL_ZMIN, POLOIDAL_ZMAX, n)
    # Pin corners so the hull covers the full domain.
    R = np.concatenate(
        [R, [POLOIDAL_RMIN, POLOIDAL_RMAX, POLOIDAL_RMIN, POLOIDAL_RMAX]]
    )
    z = np.concatenate(
        [z, [POLOIDAL_ZMIN, POLOIDAL_ZMIN, POLOIDAL_ZMAX, POLOIDAL_ZMAX]]
    )
    points = np.column_stack([R, z])
    # Linear field f(R, z) = 3·R + 2·z  (defined at all source points)
    param = 3.0 * R + 2.0 * z

    nr, nz = 10, 14
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nr),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, nz),
    )
    result = interpolate_param(param, points, grid_R, grid_z)

    # Expected values on the coarse grid, shape (nr, nz) after transpose.
    expected = 3.0 * grid_R.T + 2.0 * grid_z.T

    # Skip 2-cell border: Gaussian boundary artefacts (sigma = supersample/2 = 2
    # fine-grid pixels) can reach ~2 sigma ≈ 1–2 coarse-cell widths from edge.
    np.testing.assert_allclose(result[2:-2, 2:-2], expected[2:-2, 2:-2], atol=0.05)


def test_interpolate_param_ellipse_zeros_outside():
    """Cells outside the plasma boundary must be exactly zero.

    Physical requirement: if there is no plasma data at a location, the
    interpolated value there must be zero — no artificial values must be
    introduced via nearest-neighbour extrapolation or Gaussian bleeding.

    This test confines source data to an elliptical cross-section in (R, z).
    All four corners of the rectangular domain lie well outside the ellipse
    (normalised distance ≈ 2.6 semi-axis radii from the centre), so their
    Gaussian kernel coverage fraction is effectively zero and the normalized
    convolution must return exactly 0.0 there.

    Cells deep inside the ellipse (within 0.5 × the semi-axis radii from the
    centre) must reproduce the constant input value accurately.
    """
    rng = np.random.default_rng(0)
    # Ellipse: centre (R=1.6, z=-0.1), semi-axes a_R=0.35 m, a_z=0.55 m.
    R_c, z_c, a_R, a_z = 1.6, -0.1, 0.35, 0.55

    # Sample ~400 points uniformly distributed inside the ellipse.
    n_target = 400
    R_cands = rng.uniform(R_c - a_R, R_c + a_R, n_target * 4)
    z_cands = rng.uniform(z_c - a_z, z_c + a_z, n_target * 4)
    inside = ((R_cands - R_c) / a_R) ** 2 + ((z_cands - z_c) / a_z) ** 2 <= 1.0
    R = R_cands[inside][:n_target]
    z = z_cands[inside][:n_target]
    points = np.column_stack([R, z])
    param = np.ones(len(points))  # constant plasma field = 1.0 inside ellipse

    nr, nz = 15, 20
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nr),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, nz),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    # result has shape (nr, nz); result[iR, iz] corresponds to
    # (linspace_R[iR], linspace_z[iz]).

    # ── Corners: well outside ellipse → must be exactly 0.0 ──────────────────
    assert result[0, 0] == 0.0, "Corner (RMIN, ZMIN) must be exactly zero"
    assert result[-1, 0] == 0.0, "Corner (RMAX, ZMIN) must be exactly zero"
    assert result[0, -1] == 0.0, "Corner (RMIN, ZMAX) must be exactly zero"
    assert result[-1, -1] == 0.0, "Corner (RMAX, ZMAX) must be exactly zero"

    # ── Deep interior: must reproduce the constant value ──────────────────────
    # grid_R.T[iR, iz] = R-coordinate of result[iR, iz]
    R_centres = grid_R.T  # shape (nr, nz)
    z_centres = grid_z.T  # shape (nr, nz)
    deep_inside = ((R_centres - R_c) / a_R) ** 2 + (
        (z_centres - z_c) / a_z
    ) ** 2 <= 0.25
    assert deep_inside.any(), "No coarse cells fall in the inner half of the ellipse"
    np.testing.assert_allclose(result[deep_inside], 1.0, atol=0.05)


# ── interpolate_parameters ───────────────────────────────────────────────────
@pytest.fixture
def uniform_plasma_data():
    """Plasma data on a uniform point cloud spanning the full domain.

    _make_uniform_points appends the four corners, so len(points) = n + 4.
    All parameter arrays are sized to match.
    """
    points = _make_uniform_points(150)
    n = len(points)
    neonlist = [np.ones(n) for _ in range(11)]
    eTemp = np.ones(n)
    eDens = np.ones(n)
    return points, neonlist, eTemp, eDens


def test_interpolate_parameters_output_shapes(uniform_plasma_data):
    """Return shapes must be (11, res_R, res_z), (res_R, res_z), (res_R, res_z)."""
    points, neonlist, eTemp, eDens = uniform_plasma_data
    res_R, res_z = 12, 20

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, res_R, res_z, neonlist, eTemp, eDens
    )

    assert i_neon.shape == (11, res_R, res_z)
    assert i_eTemp.shape == (res_R, res_z)
    assert i_eDens.shape == (res_R, res_z)


def test_interpolate_parameters_no_nan(uniform_plasma_data):
    """No NaN values must appear in any of the three returned arrays."""
    points, neonlist, eTemp, eDens = uniform_plasma_data

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, 10, 15, neonlist, eTemp, eDens
    )

    assert not np.any(np.isnan(i_neon))
    assert not np.any(np.isnan(i_eTemp))
    assert not np.any(np.isnan(i_eDens))


def test_interpolate_parameters_outside_hull_is_nonnegative():
    """Grid cells outside the plasma hull must be >= 0.0.

    Cells well outside the convex hull of the source data have a Gaussian
    coverage fraction below the threshold and are set to exactly 0.0 by the
    normalized convolution.  Cells at or inside the hull receive renormalized
    Gaussian-averaged values, which are >= 0 whenever the input data are >= 0.
    No nearest-neighbour extrapolation is performed; the only "fill" is the
    zero introduced for below-threshold cells.
    """
    rng = np.random.default_rng(1)
    n = 80
    # Cluster points in a small sub-region so many grid cells are outside the hull.
    R = rng.uniform(1.5, 1.7, n)
    z = rng.uniform(-0.2, 0.2, n)
    points = np.column_stack([R, z])

    neonlist = [np.ones(n) for _ in range(11)]
    eTemp = np.ones(n)
    eDens = np.ones(n)

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, 10, 15, neonlist, eTemp, eDens
    )

    assert np.all(i_neon >= 0)
    assert np.all(i_eTemp >= 0)
    assert np.all(i_eDens >= 0)


def test_interpolate_parameters_neon_count():
    """There are always exactly 11 neon charge-state arrays returned."""
    points = _make_uniform_points()
    neonlist = [np.ones(len(points)) for _ in range(11)]
    eTemp = np.ones(len(points))
    eDens = np.ones(len(points))

    i_neon, _, _ = interpolate_parameters(points, 8, 8, neonlist, eTemp, eDens)
    assert i_neon.shape[0] == 11
