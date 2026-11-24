"""Compact markdown tables from check3_results.json. Usage: python checks/adf15/compact_tables.py OUT_DIR"""
import json, sys
import numpy as np
r = json.load(open(f'{sys.argv[1]}/check3_results.json'))
TES = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
out = []
def tab(p, ne):
    out.append(f'\n### P_ADF15 / {p}, ne={ne}  (- = outside pec40 Te grid, ^ = PLT extrapolated in Te)')
    out.append('| W | cpl | ' + ' | '.join(f'{t} eV' for t in TES) + ' |')
    out.append('|---|---|' + '---|' * len(TES))
    for x in r:
        c = []
        for t in TES:
            v = x['vals'][f'{ne}_{t}']
            c.append('-' if v is None else f'{v[p]:.2g}' + ('^' if v[p + '_ext'] else ''))
        out.append(f'| {x["z"]} | {x["c"]} | ' + ' | '.join(c) + ' |')
for p in ('plt41', 'plt50'):
    tab(p, '1e+13')
out.append('\n### ne dependence: max over Te of |log10(ratio@1e14 / ratio@1e13)|, plt41')
for x in r:
    d = [abs(np.log10(x['vals'][f'1e+14_{t}']['plt41'] / x['vals'][f'1e+13_{t}']['plt41'])) for t in TES if x['vals'][f'1e+13_{t}'] is not None]
    out.append(f'W{x["z"]} {x["c"]}: {10**max(d):.2f}x' if d else f'W{x["z"]} {x["c"]}: n/a')
out.append('\n### fraction of ADF15 power kept by cherab 1.6 (ne=1e13), ls and ic only (ca/cl keep all)')
out.append('| W | cpl | ' + ' | '.join(f'{t} eV' for t in TES) + ' |')
for x in r:
    if x['c'] not in ('ls', 'ic'): continue
    c = ['-' if x['vals'][f'1e+13_{t}'] is None else f'{x["vals"][f"1e+13_{t}"]["cher_frac"]:.2f}' for t in TES]
    out.append(f'| {x["z"]} | {x["c"]} | ' + ' | '.join(c) + ' |')
open(f'{sys.argv[1]}/check3_compact.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
