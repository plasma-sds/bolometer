import aug_sfutils as sf
import pandas as pd

amplifiers = ["6", "7"]
levels = ["0", "1", "2"]

objects = []

for s in amplifiers:
    for l in levels:
        for a in range(32):
            objects.append(f"CS{s}L{l}A{a:02d}")

shot = 40986

blc = sf.SFREAD(shot, "BLC", experiment="AUGD")

rows = []

for objname in objects:

    params = blc.getlist(objname)
    #help(params)
    #break

    row = {
        "ParameterSet": objname,
        "NCALSTEP": params["NCALSTEP"].data
    }

    for i in range(5):
        row[f"MULTIA{i:02d}"] = params[f"MULTIA{i:02d}"].data
        row[f"SHIFTB{i:02d}"] = params[f"SHIFTB{i:02d}"].data

    rows.append(row)

df = pd.DataFrame(rows)

df.to_excel(f"blc_calibration_{shot}.xlsx", index=False)