# %%
import json

import matplotlib.pyplot as plt
import numpy as np

from axuv.io import PROJECT_ROOT
from axuv.plotting import set_plt_rcparams
from publications.plot_sightlines import line

set_plt_rcparams()

project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# %% Load data from data_dir / tungsten_refractivity_data.json
with open(data_dir / "tungsten_refractivity_data.json") as f:
    data = json.load(f)
    wavelength = np.array(data["wavelength"])
    energy = 1240.0 / wavelength
    extinction = np.array(data["extinction"])
    refractive_index = np.array(data["index"])

nominal = np.loadtxt(data_dir / "axuv_sensitivity.csv", delimiter=",")
degraded = np.loadtxt(data_dir / "degraded_sensitivity.csv", delimiter=",")


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
plt.savefig(output_dir / "tungsten_refractivity.eps", format="eps")
plt.show()
