"""Unit tests for axuv.geometry — pure math, no raysect required except for point3d_to_rz."""
import math
import pytest
import numpy as np


# ── toroidal_to_cartesian ─────────────────────────────────────────────────────
from axuv.geometry import toroidal_to_cartesian


def test_toroidal_to_cartesian_returns_ndarray():
    result = toroidal_to_cartesian(1.0, 45.0, 0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_toroidal_to_cartesian_phi0():
    """At phi=0: X=R, Y=0, Z=z."""
    result = toroidal_to_cartesian(2.0, 0.0, 0.5)
    np.testing.assert_allclose(result, [2.0, 0.0, 0.5])


def test_toroidal_to_cartesian_phi90():
    """At phi=90 deg: X=0, Y=R, Z=z."""
    result = toroidal_to_cartesian(1.5, 90.0, -0.3)
    np.testing.assert_allclose(result, [0.0, 1.5, -0.3], atol=1e-10)


def test_toroidal_to_cartesian_phi180():
    """At phi=180 deg: X=-R, Y≈0, Z=z."""
    result = toroidal_to_cartesian(1.0, 180.0, 0.0)
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-10)


def test_toroidal_to_cartesian_phi270():
    """At phi=270 deg: X=0, Y=-R, Z=z."""
    result = toroidal_to_cartesian(1.0, 270.0, 0.0)
    np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-10)


@pytest.mark.parametrize("z", [-1.5, 0.0, 1.2])
def test_toroidal_to_cartesian_z_passthrough(z):
    """Z coordinate passes through unchanged regardless of R and phi."""
    result = toroidal_to_cartesian(1.0, 30.0, z)
    assert result[2] == pytest.approx(z)


@pytest.mark.parametrize("R", [0.5, 1.0, 2.2])
def test_toroidal_to_cartesian_r_magnitude(R):
    """sqrt(X^2 + Y^2) should always equal R."""
    result = toroidal_to_cartesian(R, 37.5, 0.1)
    assert math.isclose(math.hypot(result[0], result[1]), R, rel_tol=1e-10)


def test_toroidal_to_cartesian_full_circle_closes():
    """Going around phi=0..360 should return to the same X,Y."""
    start = toroidal_to_cartesian(1.8, 0.0, 0.0)
    end   = toroidal_to_cartesian(1.8, 360.0, 0.0)
    np.testing.assert_allclose(start[:2], end[:2], atol=1e-10)


# ── point3d_to_rz ─────────────────────────────────────────────────────────────
def test_point3d_to_rz_basic():
    """R = hypot(x, y), z passes through unchanged."""
    from raysect.core import Point3D
    from axuv.geometry import point3d_to_rz

    p  = Point3D(3.0, 4.0, 0.7)
    rz = point3d_to_rz(p)
    assert rz[0] == pytest.approx(5.0)   # sqrt(9 + 16)
    assert rz[1] == pytest.approx(0.7)


def test_point3d_to_rz_on_axis():
    """A point on the symmetry axis (x=y=0) should have R=0."""
    from raysect.core import Point3D
    from axuv.geometry import point3d_to_rz

    p  = Point3D(0.0, 0.0, -1.0)
    rz = point3d_to_rz(p)
    assert rz[0] == pytest.approx(0.0)
    assert rz[1] == pytest.approx(-1.0)


def test_point3d_to_rz_negative_xy():
    """R is always non-negative even for negative x, y."""
    from raysect.core import Point3D
    from axuv.geometry import point3d_to_rz

    p  = Point3D(-3.0, -4.0, 0.0)
    rz = point3d_to_rz(p)
    assert rz[0] == pytest.approx(5.0)