import numpy as np
import pytest

# axuv_measurement.processing imports aug_sfutils, from the measurement extra
pytest.importorskip("aug_sfutils")

import axuv_measurement.processing as processing  # noqa: E402
from axuv_measurement.config import D16, DVC


def _fake_signals(n_channels=48, n_time=103):
    """(N_channels, T) data in the layout get_AXUV_signals returns.

    Channel c at time index t has the value 1e5 * (c + 1) + t, so every
    entry identifies its channel and time.
    """
    channels = np.arange(n_channels)[:, None]
    times = np.arange(n_time)[None, :]
    data = 1e5 * (channels + 1) + times
    timesignal = 2.31 + 1e-6 * np.arange(n_time)
    return data.astype(float), timesignal


@pytest.fixture
def fake_reader(monkeypatch):
    data, timesignal = _fake_signals()
    monkeypatch.setattr(processing, "get_AXUV_signals",
                        lambda *a, **kw: (data.copy(), timesignal.copy()))
    return data, timesignal


@pytest.mark.parametrize("diagname, factor", [(D16, 10), (DVC, 4)])
def test_downsample_shape_and_values(fake_reader, diagname, factor):
    data, timesignal = fake_reader
    out, time = processing.downsample(40673, diagname)

    n_blocks = data.shape[1] // factor
    assert out.shape == (n_blocks, 48)          # (T_down, N_channels)
    assert time.shape == (n_blocks,)
    # block b of channel c averages time indices b*factor ... b*factor + factor - 1
    expected = 1e5 * (np.arange(48)[None, :] + 1) + (
        np.arange(n_blocks)[:, None] * factor + (factor - 1) / 2
    )
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(time, timesignal[: n_blocks * factor].reshape(-1, factor).mean(1))


def test_downsample_signalrange_is_inclusive(fake_reader):
    out, _ = processing.downsample(40673, D16, signalrange=(3, 44))
    assert out.shape[1] == 42
    # first column is channel 3, last is channel 44
    assert out[0, 0] // 1e5 == 4
    assert out[0, -1] // 1e5 == 45


def test_ridge_filter_runs_on_reader_layout(fake_reader):
    data, _ = fake_reader
    filtered, time, raw = processing.ridge_filter(40673, D16, signalrange=(0, 29))
    assert filtered.shape == raw.shape == (time.shape[0], 30)
    assert np.all(np.isfinite(filtered))
