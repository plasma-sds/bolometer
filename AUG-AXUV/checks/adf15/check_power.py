"""ADF15 excitation line power against ADF11 PLT, per W charge state and coupling.

P_ADF15(Te, ne) = sum over excitation blocks of PEC [cm^3 s^-1] * h c / lambda,
in W cm^3, divided by PLT from the ADF11 files plt50, plt41, plt42 and plt89
(Cherab parse_adf11; charge = Z1 - 1 as in Cherab's install). Also gives the
fraction of the ADF15 power that Cherab's parse_adf15 keeps.

Interpolation is linear in log10(PEC or PLT) against log10(Te), at grid
densities 1e13 and 1e14 cm^-3. ADF15 is not extrapolated outside each file's
Te grid (cell None or "-"); PLT outside its Te grid is extrapolated linearly in
log-log and flagged. Writes check3_results.json and check3_tables.txt to OUT_DIR.

    python checks/adf15/check_power.py ADAS_ROOT OUT_DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from cherab.core.atomic import tungsten
from cherab.openadas.parse.adf11 import parse_adf11
from cherab.openadas.parse.adf15 import parse_adf15

sys.path.insert(0, str(Path(__file__).parent))
from paths import W_CHARGES, W_COUPLINGS, W_PLT, w_pec, w_plt  # noqa: E402
from rawadf15 import read_blocks  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("adas_root", type=Path, help="Open-ADAS tree (adf15/pec40#w/..., adf11/plt*/...).")
parser.add_argument("out_dir", type=Path)
args = parser.parse_args()
d, out = args.adas_root, args.out_dir
out.mkdir(parents=True, exist_ok=True)
HC = 6.62607015e-34 * 2.99792458e8          # J m
ZS = W_CHARGES
CPL = W_COUPLINGS
TES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
NES = [1e13, 1e14]
PLTS = W_PLT

def loginterp_1d(x, xp, yp):
    lx, lxp, lyp = np.log10(x), np.log10(xp), np.log10(np.maximum(yp, 1e-300))
    if lx < lxp[0] or lx > lxp[-1]:
        i = 0 if lx < lxp[0] else -2
        s = (lyp[i+1]-lyp[i])/(lxp[i+1]-lxp[i]); return 10**(lyp[i] + s*(lx-lxp[i])), True
    return 10**np.interp(lx, lxp, lyp), False

plt = {}
for p in PLTS:
    r = parse_adf11(tungsten, str(w_plt(d, p)))[tungsten]
    plt[p] = {z1 - 1: dict(ne=10**r[z1]['ne'], te=10**r[z1]['te'], rate=10**r[z1]['rates']) for z1 in r}   # W cm^3, Z1-1 convention as cherab install.py

def plt_at(p, z, te, ne):
    t = plt[p][z]; lne = np.log10(t['ne'])
    # log-linear interpolation in ne (exact grid point for plt41/42/50; plt89 grid has no 1e13/1e14)
    lrow = np.array([np.interp(np.log10(ne), lne, np.log10(t['rate'][:, k])) for k in range(len(t['te']))])
    return loginterp_1d(te, t['te'], 10**lrow)

res = []
for z in ZS:
    for c in CPL:
        path = w_pec(d, c, z)
        blocks = [b for b in read_blocks(path) if b['type'] == 'excit']
        te0, te1 = blocks[0]['te'][0], blocks[0]['te'][-1]
        assert all(np.allclose(b['te'], blocks[0]['te']) and np.allclose(b['ne'], blocks[0]['ne']) for b in blocks)
        # cherab 1.6 retained set (rates in m^3/s, wavelength nm)
        try:
            rates, wls = parse_adf15(tungsten, z, str(path))
            ex = rates['excitation'][tungsten][z]; wl = wls[tungsten][z]
            ncher = len(ex)
        except Exception as e:
            ex = None; ncher = None
        row = dict(z=z, c=c, nblk=len(blocks), ncher=ncher, te_min=float(te0), te_max=float(te1), vals={})
        for ne in NES:
            j = int(np.argmin(abs(np.log10(blocks[0]['ne']) - np.log10(ne))))
            assert abs(np.log10(blocks[0]['ne'][j]) - np.log10(ne)) < 1e-3
            P = sum(b['pec'][j] * HC / (b['wl'] * 1e-10) for b in blocks)          # W cm^3 on file Te grid
            Pc = None
            if ex is not None:
                Pc = sum(v['rate'][j] * 1e6 * HC / (wl[k] * 1e-9) for k, v in ex.items())   # m^3 -> cm^3
            for te in TES:
                key = f'{ne:.0e}_{te}'
                if te < te0 * (1 - 1e-6) or te > te1 * (1 + 1e-6):
                    row['vals'][key] = None; continue
                p15, _ = loginterp_1d(te, blocks[0]['te'], P)
                v = dict(p15=float(p15))
                if Pc is not None:
                    v['cher_frac'] = float(loginterp_1d(te, blocks[0]['te'], Pc)[0] / p15)
                for p in PLTS:
                    pl, ext = plt_at(p, z, te, ne)
                    v[p] = float(p15 / pl); v[p + '_ext'] = bool(ext)
                row['vals'][key] = v
        res.append(row)
json.dump(res, open(out / 'check3_results.json', 'w'), indent=1)

# text tables
lines = []
lines.append('Te range of pec40#w files (eV); identical for all four couplings of a charge state? ' +
             str(all(len({(r['te_min'], r['te_max']) for r in res if r['z'] == z}) == 1 for z in ZS)))
for z in ZS:
    r = [x for x in res if x['z'] == z][0]
    lines.append(f'  W{z:<2d} Te {r["te_min"]:.4g} - {r["te_max"]:.4g} eV')
lines.append('')
lines.append('Blocks in file / transitions kept by cherab 1.6 parse_adf15')
for z in ZS:
    lines.append(f'  W{z:<2d} ' + '  '.join(f'{x["c"]}:{x["nblk"]}/{x["ncher"]}' for x in res if x['z'] == z))
for p in PLTS:
    for ne in NES:
        lines.append('')
        lines.append(f'Ratio P_ADF15(all excit blocks) / {p}   ne = {ne:.0e} cm^-3   ("-" outside ADF15 Te grid, "^" PLT extrapolated)')
        lines.append('z  cpl  ' + ''.join(f'{te:>9d}' for te in TES))
        for x in res:
            cells = []
            for te in TES:
                v = x['vals'][f'{ne:.0e}_{te}']
                cells.append(f'{"-":>9s}' if v is None else f'{v[p]:8.2e}' + ('^' if v[p + "_ext"] else ' '))
            lines.append(f'W{x["z"]:<2d} {x["c"]:3s} ' + ''.join(cells))
lines.append('')
for ne in NES:
    lines.append(f'Fraction of ADF15 power kept by cherab 1.6 parse_adf15 (key collisions), ne = {ne:.0e}')
    lines.append('z  cpl  ' + ''.join(f'{te:>7d}' for te in TES))
    for x in res:
        cells = []
        for te in TES:
            v = x['vals'][f'{ne:.0e}_{te}']
            cells.append(f'{"-":>7s}' if v is None or 'cher_frac' not in v else f'{v["cher_frac"]:7.3f}')
        lines.append(f'W{x["z"]:<2d} {x["c"]:3s} ' + ''.join(cells))
open(out / 'check3_tables.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[:45]))
