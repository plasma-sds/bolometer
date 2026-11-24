"""Does Cherab's own parse_adf15 read the W pec40 and B pec93 files, and how many
transitions does it keep?

For each file, prints the block counts and grid ranges from an independent
reader (rawadf15.py) next to Cherab's result or the error it raises. Cherab
keys transitions by level pair only, so it drops blocks that share a pair;
axuv.adf15 is the reader the runs use.

    python checks/adf15/check_parse.py ADAS_ROOT [--ionelec FILE]
"""

import argparse
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from cherab.core.atomic import boron, tungsten
from cherab.openadas.parse.adf15 import parse_adf15

sys.path.insert(0, str(Path(__file__).parent))
from paths import B_FILES, W_CHARGES, W_COUPLINGS, b_pec, w_pec  # noqa: E402
from rawadf15 import read_blocks  # noqa: E402


def ntrans(rates, key):
    try:
        return sum(len(v2) for v in rates[key].values() for v2 in v.values())
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("adas_root", type=Path, help="Open-ADAS tree (adf15/pec40#w/..., adf15/pec93#b/...).")
    parser.add_argument("--ionelec", type=Path, help="Optional B0 ionelec_pec_b0.dat file.")
    args = parser.parse_args()

    print("cherab", version("cherab"))
    jobs = []
    for z in W_CHARGES:
        for c in W_COUPLINGS:
            jobs.append((tungsten, z, w_pec(args.adas_root, c, z), f"W{z} {c}"))
    for z, name in B_FILES:
        jobs.append((boron, z, b_pec(args.adas_root, name), f"B{z} {name[:3]}"))
    if args.ionelec:
        jobs.append((boron, 0, args.ionelec, "B0 ionelec"))

    for element, z, path, label in jobs:
        if not path.exists():
            print(f"{label:10s} missing {path}")
            continue
        raw = read_blocks(path)
        ex = sum(b["type"] == "excit" for b in raw)
        rc = sum(b["type"] == "recom" for b in raw)
        te = (min(b["te"].min() for b in raw), max(b["te"].max() for b in raw))
        ne = (min(b["ne"].min() for b in raw), max(b["ne"].max() for b in raw))
        try:
            rates, _ = parse_adf15(element, z, str(path))
            status = (f"OK  cherab exc={ntrans(rates, 'excitation')} rec={ntrans(rates, 'recombination')} "
                      f"cx={ntrans(rates, 'thermalcx')}")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            status = f"ERR {type(e).__name__}: {str(e)[:60]} @ {tb.name}:{tb.lineno} `{tb.line}`"
        print(f"{label:10s} raw exc={ex:2d} rec={rc:2d} Te={te[0]:.3g}-{te[1]:.4g} eV "
              f"ne={ne[0]:.0e}-{ne[1]:.0e} cm-3 | {status}")


if __name__ == "__main__":
    main()
