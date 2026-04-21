import math
import numpy as np
from raysect.core import Point2D, Point3D, Vector3D

# Convenient axis constants reused across the whole package
XAXIS  = Vector3D(1, 0, 0)
YAXIS  = Vector3D(0, 1, 0)
ZAXIS  = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)

def point3d_to_rz(point: Point3D) -> Point2D:
    """Projects a 3D Cartesian point onto the poloidal (R, z) plane."""
    return Point2D(math.hypot(point.x, point.y), point.z)

def toroidal_to_cartesian(R: float, phi: float, z: float) -> np.ndarray:
    """
    Converts toroidal coordinates (R, phi [deg], z) to Cartesian (X, Y, Z).
    Returns a length-3 numpy array.
    """
    phi_rad = np.deg2rad(phi)
    return np.array([R * np.cos(phi_rad), R * np.sin(phi_rad), z])
