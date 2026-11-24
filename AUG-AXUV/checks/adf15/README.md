# ADF15 checks for the W and B PEC files

Scripts that check the W (`pec40#w`, couplings `ls`, `ic`, `ca`, `cl`) and B (`pec93#b`, `llu`, `llr`) ADF15 files behind `axuv.adf15` and the `--pec-variant` choice.
Rerun them when the Open-ADAS files or the Cherab version change.
They print tables; they do not assert, since the ratios span orders of magnitude.

- `check_parse.py`: block counts and grid ranges of each file from an independent reader (`rawadf15.py`), next to what Cherab's own `parse_adf15` keeps or the error it raises.
- `check_parse_hydrogen_like.py`: the same for the B files with `header_format="hydrogen-like"`.
- `check_power.py`: summed excitation line power of each W file divided by ADF11 PLT (`plt50`, `plt41`, `plt42`, `plt89`), per charge state, coupling, Te and ne, plus the fraction of the power Cherab's parser keeps. Writes `check3_results.json` and `check3_tables.txt`.
- `compact_tables.py`, `side_by_side.py`: shorter tables from `check3_results.json`.
- `paths.py`: file locations in an Open-ADAS tree and the ions checked.

## Input files

The scripts take the root of an Open-ADAS tree, laid out as on open.adas.ac.uk/download: `adf15/pec40#w/pec40#w_cl#w3.dat`, `adf15/pec93#b/pec93#b_llu#b0.dat`, `adf11/plt50/plt50_w.dat`.
Cherab's download cache (`~/.cherab/openadas/repository/_download_cache`) has this layout but only the files installed so far; the rest can be fetched into a separate directory (bash; Open-ADAS writes `#` as `][` in URLs, and `curl -g` keeps the brackets literal):

```bash
A=~/Data/AUG-AXUV/adas-check   # any directory
mkdir -p "$A/adf15/pec40#w" "$A/adf15/pec93#b" "$A"/adf11/plt{50,41,42,89}
get() { curl -gsf -o "$A/$1" "https://open.adas.ac.uk/download/${1//#/][}" || echo "missing $1"; }
for z in 0 1 2 3 4 5 6 7 8 9 10 15 20 30 40 50 60 73; do
  for c in ls ic ca cl; do get "adf15/pec40#w/pec40#w_${c}#w${z}.dat"; done
done
for n in llu#b0 llu#b1 llu#b2 llu#b3 llr#b3 llr#b4; do get "adf15/pec93#b/pec93#b_${n}.dat"; done
for p in plt50 plt41 plt42 plt89; do get "adf11/$p/${p}_w.dat"; done
grep -rl "OPEN-ADAS Error" "$A"   # files the server answered with an error page
```

## Running

```bash
python checks/adf15/check_parse.py "$A"
python checks/adf15/check_parse_hydrogen_like.py "$A"
python checks/adf15/check_power.py "$A" OUT_DIR
python checks/adf15/compact_tables.py OUT_DIR
python checks/adf15/side_by_side.py OUT_DIR
```

`check_power.py` takes a few minutes. Both parse checks take `--ionelec FILE` for an extra B0 file (`ionelec_pec_b0.dat`), which is not on Open-ADAS under the paths above.

## Reference results

Results with Cherab 1.5 and 1.6 (2026-11-23) are in `~/Data/AUG-AXUV/output/cherab16-wb-check/`.
On 2026-11-24 the scripts reproduced `check2_cherab16.txt` and all `check3_*` files exactly with Cherab 1.6.0.
