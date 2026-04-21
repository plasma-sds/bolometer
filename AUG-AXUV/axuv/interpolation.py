import numpy as np
from scipy.interpolate import griddata

# Poloidal cross-section extent used for the rectangular interpolation grid
POLOIDAL_RMIN = 1.0
POLOIDAL_RMAX = 2.2
POLOIDAL_ZMIN = -1.2
POLOIDAL_ZMAX = 1.0

def interpolate_param(param, points, grid_R, grid_z, method="linear"):
    """Interpolates a single parameter from the JOREK grid to a rectangular grid."""
    return np.nan_to_num(
        griddata(points, param, (grid_R, grid_z), method=method)[:, :, 0]
    ).T

def interpolate_parameters(points, resolution_R, resolution_z, neonlist, eTemp, eDens, method="linear"):
    """
    Interpolates all plasma parameters from the JOREK grid onto a
    (resolution_R × resolution_z) rectangular grid.

    Returns (interpolated_neon [11, R, z], interpolated_eTemp, interpolated_eDens).
    """
    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)
    grid_R, grid_z = np.meshgrid(linspace_R, linspace_z)

    print("Interpolating densities and temperatures...")
    interpolated_neon = np.zeros([11, resolution_R, resolution_z])
    for i, neon in enumerate(neonlist):
        interpolated_neon[i] = interpolate_param(neon, points, grid_R, grid_z, method)

    interpolated_eTemp = interpolate_param(eTemp, points, grid_R, grid_z, method)
    interpolated_eDens = interpolate_param(eDens, points, grid_R, grid_z, method)
    return interpolated_neon, interpolated_eTemp, interpolated_eDens
