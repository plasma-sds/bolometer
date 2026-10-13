# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import raysect

from axuv.datafiles import data_path
from axuv.io import OUTPUT_DIR
from axuv.plotting import set_plt_rcparams

set_plt_rcparams()

output_dir = OUTPUT_DIR
output_dir.mkdir(parents=True, exist_ok=True)

# %% Load the tungsten n/k data shipped with Raysect (also used by axuv.cad_files)
with open(Path(raysect.__file__).parent / "optical/library/metal/data/tungsten.json") as f:
    data = json.load(f)
    wavelength = np.array(data["wavelength"])
    energy = 1240.0 / wavelength
    extinction = np.array(data["extinction"])
    refractive_index = np.array(data["index"])

nominal = np.loadtxt(data_path("responsivity/axuv_sensitivity.csv"), delimiter=",")
degraded = np.loadtxt(data_path("responsivity/degraded_sensitivity.csv"), delimiter=",")


# %% plot the data with logarithmic energy (eV) x-axis
x_full = np.logspace(
    np.log10(nominal[:, 0].min()),
    np.log10(max(nominal[:, 0].max(), degraded[:, 0].max())),
    500,
)

extinction_interpolated = np.interp(x_full, np.flip(energy), np.flip(extinction))
refractive_index_interpolated = np.interp(
    x_full, np.flip(energy), np.flip(refractive_index)
)

fig, ax = plt.subplots(dpi=(100))
ax.plot(x_full, extinction_interpolated, label="Extinction Coefficient", color="k")
ax.plot(x_full, refractive_index_interpolated, label="Refractive Index", linestyle="--", color="r")
ax.set_xscale("log")
ax.set_xlabel("Energy (eV)")
ax.set_ylim(0, )
ax.legend()
plt.savefig(output_dir / "tungsten_refractivity.pdf", format="PDF")
plt.show()
