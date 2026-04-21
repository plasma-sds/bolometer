import numpy as np
from raysect.core import Point3D, Vector3D
from raysect.optical import Spectrum

def get_spectrum_part(part: int, min_wavelengths, max_wavelengths, spectral_bins: int):
    """Returns a linspace of wavelengths for spectral window `part`."""
    return np.linspace(min_wavelengths[part], max_wavelengths[part], spectral_bins)

def emission_function_3d(x, y, z, part, plasma, min_wavelengths, max_wavelengths, spectral_bins):
    """
    Returns emission [W m^-3 sr^-1 nm^-1] at (x, y, z) integrated over
    spectral window `part`.
    """
    spectrum  = Spectrum(min_wavelengths[part], max_wavelengths[part], spectral_bins - 1)
    direction = Vector3D(0, 0, 1)
    point     = Point3D(x, y, z)
    emission  = np.zeros(spectral_bins - 1)
    for model in plasma.models:
        emission += model.emission(point, direction, spectrum.new_spectrum()).samples
    return emission
