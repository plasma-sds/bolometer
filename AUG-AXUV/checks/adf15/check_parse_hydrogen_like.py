"""Does Cherab's parse_adf15 read the B pec93 files with header_format='hydrogen-like'?

    python checks/adf15/check_parse_hydrogen_like.py ADAS_ROOT [--ionelec FILE]
"""

import argparse
import sys
from pathlib import Path

from cherab.core.atomic import boron
from cherab.openadas.parse.adf15 import parse_adf15

sys.path.insert(0, str(Path(__file__).parent))
from paths import B_FILES, b_pec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("adas_root", type=Path)
    parser.add_argument("--ionelec", type=Path, help="Optional B0 ionelec_pec_b0.dat file.")
    args = parser.parse_args()
    files = [(z, b_pec(args.adas_root, name), f"pec93_b_{name.replace('#', '_')}") for z, name in B_FILES]
    if args.ionelec:
        files.append((0, args.ionelec, "ionelec_pec_b0"))
    for z, path, label in files:
        try:
            rates, _ = parse_adf15(boron, z, str(path), header_format="hydrogen-like")
            count = {k: sum(len(t) for t in rates[k][boron].values()) for k in ("excitation", "recombination")}
            print(label, "OK", count, "example key", next(iter(rates["excitation"][boron][z])))
        except Exception as e:
            print(label, "ERR", type(e).__name__, e)


if __name__ == "__main__":
    main()
