import pytest
from raysect.optical.observer import Pixel

from axuv.cameras import WAVELENGTH_BIN_EDGES
from axuv.raytransfer_sensitivity import _set_wavelength_range


@pytest.mark.parametrize(
    "bins",
    [
        # resumed run whose first pending bin lies above Raysect's default range
        [len(WAVELENGTH_BIN_EDGES) - 2],
        # completed bins skipped between pending ones
        [0, 99, 129, 239, 298],
        # every bin in order, as in a fresh run
        list(range(len(WAVELENGTH_BIN_EDGES) - 1)),
        # bins in reverse order
        [298, 129, 0],
    ],
)
def test_set_wavelength_range_from_any_previous_range(bins):
    pixel = Pixel()
    for j in bins:
        low, high = WAVELENGTH_BIN_EDGES[j], WAVELENGTH_BIN_EDGES[j + 1]
        _set_wavelength_range(pixel, low, high)
        assert (pixel.min_wavelength, pixel.max_wavelength) == (low, high)
