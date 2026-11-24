"""Minimal independent ADF15 block reader (wavelength, type, ne, te, pec). Units as in file."""
import re
import numpy as np

HDR = re.compile(r'^\s*([0-9]+\.[0-9]*)\s*[aA]?\s+(\d+)\s+(\d+)\s*/(.*)$')

def read_blocks(path):
    with open(path) as f:
        lines = f.readlines()
    nblk = int(lines[0].split()[0])
    blocks = []
    i = 1
    while i < len(lines) and len(blocks) < nblk:
        m = HDR.match(lines[i])
        if not m:
            i += 1
            continue
        wl = float(m.group(1)); nn = int(m.group(2)); nt = int(m.group(3)); rest = m.group(4)
        t = re.search(r'type\s*=\s*([a-zA-Z]+)', rest, re.I).group(1).lower()
        vals = []
        i += 1
        need = nn + nt + nn * nt
        while len(vals) < need:
            vals += [float(x.replace('D', 'E')) for x in lines[i].split()]
            i += 1
        ne = np.array(vals[:nn]); te = np.array(vals[nn:nn + nt])
        pec = np.array(vals[nn + nt:]).reshape(nn, nt)
        blocks.append(dict(wl=wl, type=t, ne=ne, te=te, pec=pec))
    return blocks
