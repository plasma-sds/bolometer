"""Unit tests for axuv.plasma — no raysect/cherab required for get_spectrum_part."""
import pytest
import numpy as np

from axuv.plasma import get_spectrum_part

MIN_WL = [0.25, 12.4,  124.0]
MAX_WL = [12.4, 124.0, 1240.0]
BINS   = 100


@pytest.mark.parametrize("part", [0, 1, 2])
def test_get_spectrum_part_length(part):
    """Output array should have exactly SPECTRAL_BINS elements."""
    result = get_spectrum_part(part, MIN_WL, MAX_WL, BINS)
    assert len(result) == BINS


@pytest.mark.parametrize("part", [0, 1, 2])
def test_get_spectrum_part_bounds(part):
    """First and last values should match the configured wavelength bounds."""
    result = get_spectrum_part(part, MIN_WL, MAX_WL, BINS)
    assert result[0]  == pytest.approx(MIN_WL[part])
    assert result[-1] == pytest.approx(MAX_WL[part])


@pytest.mark.parametrize("part", [0, 1, 2])
def test_get_spectrum_part_monotonic(part):
    """Wavelengths should be strictly increasing."""
    result = get_spectrum_part(part, MIN_WL, MAX_WL, BINS)
    assert np.all(np.diff(result) > 0), f"Part {part} is not monotonically increasing"


def test_get_spectrum_part_returns_ndarray():
    result = get_spectrum_part(0, MIN_WL, MAX_WL, BINS)
    assert isinstance(result, np.ndarray)


def test_get_spectrum_part_positive_wavelengths():
    """All wavelengths must be strictly positive (physical requirement)."""
    for part in range(3):
        result = get_spectrum_part(part, MIN_WL, MAX_WL, BINS)
        assert np.all(result > 0), f"Non-positive wavelength in part {part}"


def test_get_spectrum_part_spectral_windows_ordered():
    """The three spectral windows should be ordered: UV → soft X-ray → hard X-ray."""
    parts = [get_spectrum_part(i, MIN_WL, MAX_WL, BINS) for i in range(3)]
    # Each window's maximum should be less than or equal to the next window's minimum
    for i in range(len(parts) - 1):
        assert parts[i][-1] <= parts[i + 1][0] + 1e-10, \
            f"Spectral window {i} overlaps with window {i+1}"


@pytest.mark.parametrize("bins", [10, 50, 200])
def test_get_spectrum_part_custom_bins(bins):
    """Function should work for any number of bins."""
    result = get_spectrum_part(0, MIN_WL, MAX_WL, bins)
    assert len(result) == bins