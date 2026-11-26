"""Tests for the dark-offset subtraction in axuv_measurement.io.get_AXUV_signals."""

import numpy as np
import pytest

pytest.importorskip("aug_sfutils")

import axuv_measurement.io as mio  # noqa: E402
from axuv_measurement.config import D16, SIGNAL_NAMES  # noqa: E402

TIME = np.linspace(-0.1, 7.9, 8001)
OFFSET = 3e5  # dark offset of every channel [W/m²]
PULSE = 2e7  # plasma signal between 2.3 s and 2.4 s [W/m²]


class FakeSFREAD:
    """Stands in for aug_sfutils.SFREAD: every channel is OFFSET plus a pulse during the plasma."""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _window(tbeg, tend):
        mask = np.ones_like(TIME, dtype=bool)
        if tbeg is not None:
            mask &= TIME >= tbeg
        if tend is not None:
            mask &= TIME <= tend
        return mask

    def gettimebase(self, sig, tbeg=None, tend=None, cal=True):
        return TIME[self._window(tbeg, tend)]

    def getobject(self, sig, tbeg=None, tend=None, cal=True):
        values = OFFSET + PULSE * ((TIME > 2.3) & (TIME < 2.4))
        return values[self._window(tbeg, tend)]


@pytest.fixture
def fake_shotfile(monkeypatch):
    monkeypatch.setattr(mio.sf, "SFREAD", FakeSFREAD)


def test_offset_from_end_of_timebase_is_subtracted(fake_shotfile):
    data, time = mio.get_AXUV_signals(40673, D16, tbeg=2.0, tend=2.6, gaussian_sigma=1)
    assert data.shape == (len(SIGNAL_NAMES[D16][1]), len(time))
    assert np.allclose(data[:, time < 2.29], 0.0)
    assert np.allclose(data[:, (time > 2.31) & (time < 2.39)], PULSE)


def test_offset_subtraction_can_be_switched_off(fake_shotfile):
    data, time = mio.get_AXUV_signals(40673, D16, tbeg=2.0, tend=2.6, gaussian_sigma=1, offset_window=None)
    assert np.allclose(data[:, time < 2.29], OFFSET)


def test_offset_window_overlapping_the_data_is_rejected(fake_shotfile):
    with pytest.raises(ValueError, match="overlaps"):
        mio.get_AXUV_signals(40673, D16, tbeg=2.0, tend=7.5)
