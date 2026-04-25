"""
emission_to_measurement.py
--------------------------
Stage 3 of the AXUV synthetic signal workflow.

After the geometry matrix (raytransfer_sensitivity.py) and the per-voxel
spectral emissions (calculate_emissions_for_raytransfer.py) have been
computed, this script folds the diode spectral power through the AXUV
spectral responsivity curve to produce a single scalar signal per diode.

All visualisation helpers live in axuv.plotting.
All responsivity curve helpers live in axuv.responsivity.
All data-loading helpers live in axuv.io.
"""

from axuv.io import load_raytransfer_data, open_emission_data
from axuv.responsivity import get_weighted_power

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply AXUV responsivity to compute synthetic diode signals."
    )
    parser.add_argument(
        "sensitivity_file",
        help="Path to the raytransfer HDF5 file (from raytransfer_sensitivity.py)",
    )
    parser.add_argument(
        "emissions_file",
        help="Path to the emissions HDF5 file (from calculate_emissions_for_raytransfer.py)",
    )
    parser.add_argument(
        "--degraded",
        action="store_true",
        default=False,
        help="Use the degraded-diode responsivity curve.",
    )
    args = parser.parse_args()

    # ── Load geometry matrix ─────────────────────────────────────────────────
    rt_data = load_raytransfer_data(args.sensitivity_file)
    sensitivity_matrix = rt_data["sensitivity_matrix"]
    grid_centres = rt_data["grid_centres"]
    inverse_voxel_map = rt_data["inverse_voxel_map"]
    diode_names = rt_data["diode_names"]
    wavelength_edges = rt_data["wavelength_bin_edges"]
    energy_edges = rt_data["energy_bin_edges_eV"]

    # ── Load emissions ────────────────────────────────────────────────────────
    em_data = open_emission_data(args.emissions_file)
    emissions = em_data["emissions"]
    energies = em_data["energies"]
    diode_measurements = em_data["diode_measurements"]

    print(f"Sensitivity matrix : {sensitivity_matrix.shape}")
    print(f"Emissions          : {emissions.shape}")
    print(f"Diode measurements : {diode_measurements.shape}")
    print(f"Diode names        : {diode_names[:5]} ...")

    # ── Weighted power per diode ──────────────────────────────────────────────
    weighted = get_weighted_power(diode_measurements, energies, degraded=args.degraded)
    print(f"\nWeighted power range: {weighted.min():.3e} – {weighted.max():.3e} W")
