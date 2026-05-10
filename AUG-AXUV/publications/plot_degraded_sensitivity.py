# %%
import numpy as np
import matplotlib.pyplot as plt

from axuv.io import PROJECT_ROOT
from axuv.plotting import set_plt_rcparams

set_plt_rcparams()

project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# %% Load data
nominal = np.loadtxt(data_dir / "axuv_sensitivity.csv", delimiter=",")
degraded = np.loadtxt(data_dir / "degraded_sensitivity.csv", delimiter=",")
errorbars = np.loadtxt(data_dir / "degraded_errorbars.csv", delimiter=",")

# errorbars has pairs of rows per degraded point: upper bound then lower bound
upper = errorbars[0::2, 1]
lower = errorbars[1::2, 1]
yerr = np.array([degraded[:, 1] - lower, upper - degraded[:, 1]])

# Build dashed degraded line over the full energy range:
# - within the measured range: interpolate between degraded points
# - outside the measured range: follow the nominal curve
degraded_min_e = degraded[:, 0].min()
degraded_max_e = degraded[:, 0].max()
x_full = np.logspace(
    np.log10(nominal[:, 0].min()),
    np.log10(max(nominal[:, 0].max(), degraded_max_e)),
    500,
)
nominal_interp = np.interp(x_full, nominal[:, 0], nominal[:, 1], right=nominal[-1, 1])
degraded_interp = np.interp(x_full, degraded[:, 0], degraded[:, 1])
in_measured_range = (x_full >= degraded_min_e) & (x_full <= degraded_max_e)
combined_y = nominal_interp.copy()
combined_y[in_measured_range] = degraded_interp[in_measured_range]

# %% Plot
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(nominal[:, 0], nominal[:, 1], color="k", label="Nominal")
ax.plot(x_full, combined_y, linestyle=":", color="r", label="Degraded (estimated)")
ax.errorbar(
    degraded[:, 0], degraded[:, 1],
    yerr=yerr,
    fmt="o",
    capsize=4,
    color="b",
    label="Measured (degraded)",
)

ax.set_xscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Sensitivity [A/W]")

ax.set_ylim(0, )
ax.set_xlim(1, )

ax.legend()

plt.savefig(output_dir / "degraded_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "degraded_sensitivity.eps", format="eps", bbox_inches="tight")
plt.show()
