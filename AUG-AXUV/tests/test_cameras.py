"""
Unit tests for axuv.cameras — tests only the pure-Python parts
(SECTOR_CAMERAS config and get_sensor_data geometry logic).
Raysect/cherab camera construction tests belong in integration tests.
"""
import math
import pytest
import pandas as pd

from axuv.cameras import SECTOR_CAMERAS, HORIZONTAL_CAMERA_IDS, get_sensor_data


# ── SECTOR_CAMERAS dict ───────────────────────────────────────────────────────
def test_sector_cameras_expected_sectors_present():
    """The two main sectors used in publications must always be present."""
    assert "S5"  in SECTOR_CAMERAS
    assert "S16" in SECTOR_CAMERAS


def test_sector_cameras_all_have_box_and_frustum_keys():
    """Every sector entry must contain both 'box' and 'frustum' lists."""
    for sector, config in SECTOR_CAMERAS.items():
        assert "box"     in config, f"{sector} missing 'box'"
        assert "frustum" in config, f"{sector} missing 'frustum'"
        assert isinstance(config["box"],     list), f"{sector}['box'] must be a list"
        assert isinstance(config["frustum"], list), f"{sector}['frustum'] must be a list"


def test_sector_cameras_no_overlap_within_sector():
    """No camera name should appear in both 'box' and 'frustum' for the same sector."""
    for sector, config in SECTOR_CAMERAS.items():
        overlap = set(config["box"]) & set(config["frustum"])
        assert not overlap, f"{sector}: {overlap} appear in both box and frustum"


def test_horizontal_cameras_only_in_box():
    """DHT and DHC are box cameras and must never appear in a frustum list."""
    for sector, config in SECTOR_CAMERAS.items():
        for cam in config["frustum"]:
            assert cam not in HORIZONTAL_CAMERA_IDS, (
                f"Horizontal camera {cam!r} found in frustum list of sector {sector}"
            )


def test_sector_cameras_s5_known_cameras():
    assert "DHC" in SECTOR_CAMERAS["S5"]["box"]
    assert "DVC" in SECTOR_CAMERAS["S5"]["frustum"]


def test_sector_cameras_s16_known_cameras():
    assert "DHT" in SECTOR_CAMERAS["S16"]["box"]
    assert "D16" in SECTOR_CAMERAS["S16"]["frustum"]


def test_sector_cameras_all_entries_non_empty():
    """Each sector must have at least one camera (box or frustum)."""
    for sector, config in SECTOR_CAMERAS.items():
        total = len(config["box"]) + len(config["frustum"])
        assert total > 0, f"{sector} has no cameras defined"


# ── get_sensor_data with a mock DataFrame ────────────────────────────────────
def _make_mock_df(n_channels=48, R=1.8, phi=22.5, z=0.0):
    """
    Creates a minimal mock of the AXUV geometry DataFrame with n_channels rows
    all pointing in the same direction, which simplifies expected-value checks.
    """
    rows = []
    for ch in range(1, n_channels + 1):
        rows.append({
            "Cam":            "DHT",
            "chan":           ch,
            "alpha":          float(ch - n_channels // 2),  # spread around 0
            "delta":          0.0,
            "d(Folie-Blende)": 0.05,
            "f_Blende":       2.41449e-06,  # slit area (m^2), typical value from AXUV_LOS_geom.txt
            "RAW":            f"S1L0A{ch:02d}",
            # start and end points define the LOS direction
            "R_start":  R,
            "Phi_start": phi,
            "z_start":  z,
            "R_end":    R + 0.5,
            "Phi_end":  phi,
            "z_end":    z,
        })
    return pd.DataFrame(rows)


def test_get_sensor_data_return_lengths():
    """angles, distances, signalnames must each have n_channels entries."""
    n = 16
    df = _make_mock_df(n_channels=n)
    angles, distances, signalnames, fwd, origin, up, slit_area, tilts = get_sensor_data("DHT", df)

    assert len(angles)      == n
    assert len(distances)   == n
    assert len(tilts)       == n
    assert len(signalnames) == n


def test_get_sensor_data_forward_vector_is_unit():
    """The returned forward vector must be a unit vector."""
    df = _make_mock_df(n_channels=48)
    _, _, _, fwd, _, _, _, _ = get_sensor_data("DHT", df)
    magnitude = math.sqrt(fwd.x**2 + fwd.y**2 + fwd.z**2)  # type: ignore[attr-defined]
    assert magnitude == pytest.approx(1.0, abs=1e-9)


def test_get_sensor_data_up_vector_is_unit():
    """The returned up vector must be a unit vector."""
    df = _make_mock_df(n_channels=48)
    _, _, _, _, _, up, _, _ = get_sensor_data("DHT", df)
    magnitude = math.sqrt(up.x**2 + up.y**2 + up.z**2)  # type: ignore[attr-defined]
    assert magnitude == pytest.approx(1.0, abs=1e-9)


def test_get_sensor_data_channel_idx_slices_correctly():
    """When channelIDX=16, only channels 17..32 should be selected."""
    n = 48
    df = _make_mock_df(n_channels=n)
    # Add a second camera block so filtering is non-trivial
    df2 = df.copy()
    df2["Cam"] = "DVC"
    full_df = pd.concat([df, df2], ignore_index=True)

    angles, distances, _, _, _, _, _, _ = get_sensor_data("DHT", full_df, channelIDX=16)
    assert len(angles) == 16
    assert len(distances) == 16


def test_get_sensor_data_signalnames_are_strings():
    df = _make_mock_df()
    _, _, signalnames, _, _, _, _, _ = get_sensor_data("DHT", df)
    assert all(isinstance(s, str) for s in signalnames)

# ── Foil orientation ─────────────────────────────────────────────────────────
def test_foils_face_along_the_chip_normal():
    """Each foil lies flat on its chip: its normal is tilted by delta from the camera
    Z-axis, independent of the diode's viewing angle alpha."""
    import numpy as np
    from raysect.core import World

    from axuv.cameras import make_axuv_camera_box

    alphas = np.array([60.0, 42.0, 25.0, 10.0, 0.0, -10.0])
    deltas = np.array([42.0, 42.0, 42.0, 0.0, 0.0, 0.0])
    names = [f"S7L1A{k:02d}" for k in range(len(alphas))]
    camera = make_axuv_camera_box(alphas, np.full(len(alphas), 0.045), deltas, names, "DHT", 2.2e-6)
    camera.parent = World()
    for foil, delta in zip(camera.foil_detectors, deltas):
        normal = foil.normal_vector
        expected = np.array([-np.sin(np.deg2rad(delta)), 0.0, np.cos(np.deg2rad(delta))])
        assert np.allclose([normal.x, normal.y, normal.z], expected, atol=1e-9)
