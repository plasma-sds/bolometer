"""Unit tests for the log/linear diode time-evolution plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize

from axuv.emission_to_measurement import plot_time_evolution
from axuv.plotting import (
    DEFAULT_LIMITS,
    SCALES,
    plot_diode_evolution,
    resolve_color_limits,
    scale_suffix,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _make_data(n_diodes=48, n_times=20, peak=3e7):
    """Return (times, data) with values between 0 and *peak*."""
    times = np.linspace(2.3276, 2.3295, n_times)
    data = np.linspace(0.0, peak, n_diodes * n_times).reshape(n_diodes, n_times)
    return times, data


# ── resolve_color_limits ──────────────────────────────────────────────────────
def test_linear_defaults_to_zero_and_data_max():
    _, data = _make_data(peak=3e7)
    assert resolve_color_limits(data, "linear") == (0.0, 3e7)


def test_log_defaults_to_four_decades_below_data_max():
    _, data = _make_data(peak=3e7)
    assert resolve_color_limits(data, "log") == pytest.approx((3e3, 3e7))


@pytest.mark.parametrize("scale", SCALES)
def test_given_limits_override_defaults(scale):
    _, data = _make_data()
    assert resolve_color_limits(data, scale, vmin=1e3, vmax=5e7) == (1e3, 5e7)


@pytest.mark.parametrize("scale", SCALES)
def test_only_vmin_given_keeps_data_max(scale):
    _, data = _make_data(peak=3e7)
    assert resolve_color_limits(data, scale, vmin=1e3) == (1e3, 3e7)


def test_log_vmin_follows_given_vmax():
    _, data = _make_data()
    assert resolve_color_limits(data, "log", vmax=1e8) == pytest.approx((1e4, 1e8))


def test_nan_values_are_ignored():
    _, data = _make_data(peak=3e7)
    data[0, 0] = np.nan
    assert resolve_color_limits(data, "linear") == (0.0, 3e7)


def test_invalid_scale_raises():
    _, data = _make_data()
    with pytest.raises(ValueError, match="scale"):
        resolve_color_limits(data, "symlog")


def test_default_limits_cover_all_scales():
    assert set(DEFAULT_LIMITS) == set(SCALES)
    assert DEFAULT_LIMITS["linear"] == (0.0, None)


def test_scale_suffix_keeps_log_names():
    assert scale_suffix("log") == ""
    assert scale_suffix("linear") == "_lin"


# ── plot_diode_evolution ──────────────────────────────────────────────────────
def test_linear_plot_uses_linear_norm_and_default_limits():
    times, data = _make_data(peak=3e7)
    _, _, pcm = plot_diode_evolution(times, data, scale="linear")
    assert isinstance(pcm.norm, Normalize) and not isinstance(pcm.norm, LogNorm)
    assert pcm.norm(1.5e7) == pytest.approx(0.5)
    assert (pcm.norm.vmin, pcm.norm.vmax) == (0.0, 3e7)


def test_log_plot_uses_log_norm():
    times, data = _make_data()
    _, _, pcm = plot_diode_evolution(times, data, scale="log", vmin=1e4, vmax=1e8)
    assert isinstance(pcm.norm, LogNorm)
    assert (pcm.norm.vmin, pcm.norm.vmax) == (1e4, 1e8)


def test_plot_has_one_row_per_diode():
    times, data = _make_data(n_diodes=48)
    _, ax, _ = plot_diode_evolution(times, data)
    assert ax.get_ylim() == pytest.approx((0.5, 48.5))


def test_plot_draws_on_given_axis():
    times, data = _make_data()
    fig, ax = plt.subplots()
    fig_out, ax_out, _ = plot_diode_evolution(times, data, ax=ax)
    assert fig_out is fig and ax_out is ax


# ── plot_time_evolution with precomputed data ─────────────────────────────────
@pytest.mark.parametrize("scale", SCALES)
def test_time_evolution_honours_given_limits(scale):
    times, data = _make_data()
    _, _, pcm, _, _ = plot_time_evolution(
        0, 48, [], None, vmin=1e3, vmax=5e7, scale=scale, data=data, times=times
    )
    assert (pcm.norm.vmin, pcm.norm.vmax) == (1e3, 5e7)


def test_time_evolution_linear_defaults():
    times, data = _make_data(peak=3e7)
    _, _, pcm, _, _ = plot_time_evolution(
        0, 48, [], None, scale="linear", data=data, times=times
    )
    assert (pcm.norm.vmin, pcm.norm.vmax) == (0.0, 3e7)


def test_time_evolution_leaves_given_times_unchanged():
    times, data = _make_data()
    original = times.copy()
    _, _, _, _, plot_times = plot_time_evolution(
        0, 48, [], None, remove_offset=True, data=data, times=times
    )
    np.testing.assert_array_equal(times, original)
    assert plot_times[0] == 0.0
