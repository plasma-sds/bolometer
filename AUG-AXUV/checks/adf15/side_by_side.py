"""Side-by-side ratio table (plt41 | plt50 at ne=1e13, Te 5/20/100/500 eV) and ne dependence for Te>=20 eV.
Usage: python checks/adf15/side_by_side.py OUT_DIR"""
import json, math, sys
r = json.load(open(f'{sys.argv[1]}/check3_results.json'))
T = (5, 20, 100, 500)
lines = ['W cpl | plt41: ' + ' '.join(f'{t:>6}' for t in T) + ' | plt50: ' + ' '.join(f'{t:>6}' for t in T)]
for x in r:
    c = []
    for p in ('plt41', 'plt50'):
        for t in T:
            v = x['vals'][f'1e+13_{t}']
            c.append(f"{'-':>6}" if v is None else f"{v[p]:>6.2g}")
    lines.append(f"W{x['z']:<2} {x['c']} | " + ' '.join(c[:4]) + ' | ' + ' '.join(c[4:]))
lines.append('\nmax factor ratio(1e14)/ratio(1e13) over Te>=20 eV, plt41:')
for x in r:
    d = [abs(math.log10(x['vals'][f'1e+14_{t}']['plt41'] / x['vals'][f'1e+13_{t}']['plt41'])) for t in (20, 50, 100, 200, 500, 1000) if x['vals'][f'1e+13_{t}'] is not None]
    lines.append(f"W{x['z']} {x['c']}: {10**max(d):.2f}" if d else f"W{x['z']} {x['c']}: n/a")
open(f'{sys.argv[1]}/check3_side_by_side.txt', 'w').write('\n'.join(lines) + '\n')
