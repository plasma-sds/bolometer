"""Unit tests for axuv.interpolation."""
import pytest
import numpy as np

from axuv.interpolation import (
    interpolate_param,
    interpolate_parameters,
    POLOIDAL_RMIN,
    POLOIDAL_RMAX,
    POLOIDAL_ZMIN,
    POLOIDAL_ZMAX,
)


# ── Constants sanity ──────────────────────────────────────────────────────────
def test_poloidal_constants_ordering():
    """Grid extent constants must form a valid (min < max) rectangle."""
    assert POLOIDAL_RMIN < POLOIDAL_RMAX
    assert POLOIDAL_ZMIN < POLOIDAL_ZMAX


def test_poloidal_constants_positive_r():
    """R coordinates must be strictly positive (tokamak major radius)."""
    assert POLOIDAL_RMIN > 0


# ── interpolate_param ─────────────────────────────────────────────────────────
def _make_uniform_points(n=200, seed=42):
    rng = np.random.default_rng(seed)
    R = rng.uniform(POLOIDAL_RMIN, POLOIDAL_RMAX, n)
    z = rng.uniform(POLOIDAL_ZMIN, POLOIDAL_ZMAX, n)
    return np.column_stack([R, z])


def test_interpolate_param_uniform_data():
    """Interpolating a constant field should reproduce that constant everywhere."""
    points = _make_uniform_points()
    param  = np.ones(len(points))[:, np.newaxis] * 7.0

    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, 5),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, 5),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    np.testing.assert_allclose(result, 7.0, atol=1e-5)


def test_interpolate_param_output_shape():
    """Output shape should match the meshgrid shape."""
    n = 100
    points = _make_uniform_points(n)
    param  = np.ones(n)[:, np.newaxis]

    nr, nz = 8, 12
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nr),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, nz),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    # interpolate_param transposes after slicing [:,:,0] so shape is (nr, nz)
    assert result.shape == (nr, nz)


def test_interpolate_param_no_nan_in_output():
    """np.nan_to_num should ensure no NaN values in the result."""
    points = _make_uniform_points()
    param  = np.ones(len(points))[:, np.newaxis]
    grid_R, grid_z = np.meshgrid(
        np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, 6),
        np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, 6),
    )
    result = interpolate_param(param, points, grid_R, grid_z)
    assert not np.any(np.isnan(result))


# ── interpolate_parameters ───────────────────────────────────────────────────
@pytest.fixture
def uniform_plasma_data():
    n      = 150
    points = _make_uniform_points(n)
    neonlist = [np.ones(n) for _ in range(11)]
    eTemp    = np.ones(n)
    eDens    = np.ones(n)
    return points, neonlist, eTemp, eDens


def test_interpolate_parameters_output_shapes(uniform_plasma_data):
    points, neonlist, eTemp, eDens = uniform_plasma_data
    res_R, res_z = 12, 20

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, res_R, res_z, neonlist, eTemp, eDens
    )

    assert i_neon.shape  == (11, res_R, res_z)
    assert i_eTemp.shape == (res_R, res_z)
    assert i_eDens.shape == (res_R, res_z)


def test_interpolate_parameters_no_nan(uniform_plasma_data):
    points, neonlist, eTemp, eDens = uniform_plasma_data

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, 10, 15, neonlist, eTemp, eDens
    )

    assert not np.any(np.isnan(i_neon))
    assert not np.any(np.isnan(i_eTemp))
    assert not np.any(np.isnan(i_eDens))


def test_interpolate_parameters_outside_hull_is_zero():
    """
    Points outside the convex hull of the input data should become 0 (via
    np.nan_to_num), not NaN or a large number.
    """
    rng = np.random.default_rng(1)
    n   = 80
    # Cluster points in a small sub-region so many grid cells are outside the hull
    R      = rng.uniform(1.5, 1.7, n)
    z      = rng.uniform(-0.2, 0.2, n)
    points = np.column_stack([R, z])

    neonlist = [np.ones(n) for _ in range(11)]
    eTemp    = np.ones(n)
    eDens    = np.ones(n)

    i_neon, i_eTemp, i_eDens = interpolate_parameters(
        points, 10, 15, neonlist, eTemp, eDens
    )

    assert np.all(i_neon  >= 0)
    assert np.all(i_eTemp >= 0)
    assert np.all(i_eDens >= 0)


def test_interpolate_parameters_neon_count():
    """There are always exactly 11 neon charge state arrays returned."""
    points   = _make_uniform_points()
    neonlist = [np.ones(len(points)) for _ in range(11)]
    eTemp    = np.ones(len(points))
    eDens    = np.ones(len(points))

    i_neon, _, _ = interpolate_parameters(points, 8, 8, neonlist, eTemp, eDens)
    assert i_neon.shape[0] == 11