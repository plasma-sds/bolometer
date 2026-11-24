"""Radiated energy and peak radiated power of the DREAM SPI cases, from DREAM itself.

Integrates other/fluid/Tcold_radiation over the plasma volume (VpVol * dr)
and over time, along the joined series init_0, init_1, ... up to the TQ
onset (settings/init/filetimeindex of main.h5), then main. Rows of
other/fluid belong to the interval ending at t[k + 1], so each row is
weighted by its time step. Gives the with/without impurities difference in
DREAM's own radiation, to compare with the Cherab sightline runs.

    python publications/dream_spi/dream_radiated_energy.py [--dream-dir DIR]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

DREAM_DIR = Path.home() / "Data" / "AUG-AXUV" / "DREAM_SPI"
SHOTS = ("40737", "40949")
CASES = ("noImpurities", "withImpurities")


def radiated(case_dir):
    """(energy J, peak power W) of one case."""
    inits = sorted(case_dir.glob("init_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
    with h5py.File(case_dir / "main.h5", "r") as f:
        onset = int(f["settings/init/filetimeindex"][0])
    energy, peak = 0.0, 0.0
    for path in [*inits, case_dir / "main.h5"]:
        with h5py.File(path, "r") as f:
            t = f["grid/t"][()]
            volume = f["grid/VpVol"][()] * f["grid/dr"][()]
            power = (f["other/fluid/Tcold_radiation"][()] * volume).sum(axis=1)
        if path == inits[-1]:
            power, t = power[:onset], t[:onset + 1]
        energy += np.sum(power * np.diff(t))
        peak = max(peak, power.max())
    return energy, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dream-dir", type=Path, default=DREAM_DIR,
                        help="Directory with the <shot>_nr11_<case> case directories.")
    args = parser.parse_args()
    for shot in SHOTS:
        energies = {}
        for case in CASES:
            energy, peak = radiated(args.dream_dir / f"{shot}_nr11_{case}")
            energies[case] = energy
            print(f"{shot} {case:15s} E_rad {energy / 1e6:.3f} MJ, peak {peak / 1e6:.0f} MW")
        change = energies["withImpurities"] / energies["noImpurities"] - 1
        print(f"{shot} with/without impurities: {change:+.1%}")


if __name__ == "__main__":
    main()
