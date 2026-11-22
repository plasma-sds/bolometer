# %%
import bisect
import numpy as np
import matplotlib.pyplot as plt

from axuv.datafiles import data_path
from axuv.io import OUTPUT_DIR
from axuv.plotting import set_plt_rcparams

set_plt_rcparams()

output_dir = OUTPUT_DIR
output_dir.mkdir(parents=True, exist_ok=True)

# %% Load data
nominal = np.loadtxt(data_path("responsivity/axuv_sensitivity.csv"), delimiter=",")
degraded = np.loadtxt(data_path("responsivity/degraded_sensitivity.csv"), delimiter=",")
errorbars = np.loadtxt(data_path("responsivity/degraded_errorbars.csv"), delimiter=",")

# errorbars has pairs of rows per degraded point: upper bound then lower bound
upper_error = errorbars[0::2, 1]
lower_error = errorbars[1::2, 1]
yerr = np.array([degraded[:, 1] - lower_error, upper_error - degraded[:, 1]])

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
lower_error_interp = np.interp(x_full, degraded[:, 0], lower_error)

# change the degraded response from nominal to measured in the measurement range
in_measured_range = (x_full >= degraded_min_e) & (x_full <= degraded_max_e)
non_measured_range = (x_full <= degraded_min_e)
combined_degraded = nominal_interp.copy()
combined_degraded[in_measured_range] = degraded_interp[in_measured_range]


# Create a worst estimated responsivity function
# Take -10% of the degraded response in the non-measured range as a first extimation for the worst response
worst_estimation = nominal_interp.copy()
worst_estimation[non_measured_range] = nominal_interp[non_measured_range] * 0.9
# in the 4-10 eV range we suppose that the diodes can be "fully" blind
in_fully_blind_range = (x_full >= 4.0) & (x_full <= 10.0)
# in the 10-60 eV range we suppose that the diodes degraded significantly
in_significantly_degraded_range = (x_full >= 10.0) & (x_full <= 60.0)

worst_estimation[in_fully_blind_range] = 0.01  # A/W
index_at_60eV = bisect.bisect_left(x_full, 60.0)
worst_value_at_60eV = worst_estimation[index_at_60eV]
# worst value at 10 eV is 0.01 A/W
# incline (value at 60 eV - 0.01) / (60 - 10)
# x_temp to start at 10 eV
x_temp = x_full - 10.0
incline = (worst_value_at_60eV - 0.01) / (60.0 - 10.0)
worst_estimation[in_significantly_degraded_range] = 0.01 + incline * x_temp[in_significantly_degraded_range]
# In measured range, take the lower errors and interpolate in between
worst_estimation[in_measured_range] = lower_error_interp[in_measured_range]

# Save x_full and worst_estimation as two columns in csv. Copy it to the data key
# responsivity/worst_estimation.csv (and upload it) to make it an input of the workflow.
np.savetxt(output_dir / "worst_estimation.csv", np.column_stack((x_full, worst_estimation)), delimiter=",")

# %% Plot
fig, ax = plt.subplots(figsize=(7, 4.5))

line1, = ax.plot(nominal[:, 0], nominal[:, 1], color="k", label="Nominal")
line2, = ax.plot(x_full, combined_degraded, linestyle="--", color="r", label="Degraded")
err2 = ax.errorbar(
    degraded[:, 0], degraded[:, 1],
    yerr=yerr,
    fmt="o",
    capsize=4,
    color="r",
    label="Measured (degraded)",
)
# Measured points below the lowest point used for the degraded responsivity.
# Digitised from Bernert et al. 2014, Rev. Sci. Instrum. 85 033503, figures 4 and 5
# (the two figures agree where both show a point). The 50 eV point is left out: its
# average lies above the published axis (> 0.5 A/W).
# Bernert et al. attribute the rise below 70 eV to the large error bars.
# Bars that leave the plot range run off the axis.
Y_TOP = 0.55
low_e = np.array([60.0, 70.0, 80.0])
low_val = np.array([0.275, 0.218, 0.214])
low_lower = np.array([-0.1, -0.1, 0.131])            # below 0: bar runs off the axis
low_upper = np.array([Y_TOP + 0.1, 0.450, 0.297])  # above Y_TOP: bar runs off the axis
err_low = ax.errorbar(
    low_e, low_val,
    yerr=np.array([low_val - low_lower, low_upper - low_val]),
    fmt="s",
    markersize=5,
    capsize=4,
    color="gray",
    label="Measured, not used",
)
line3, = ax.plot(x_full, worst_estimation, linestyle="-.", color="b", label="Worst estimation", zorder=20)

ax.set_xscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel("Sensitivity [A/W]")

ax.set_ylim(0, Y_TOP)
ax.set_xlim(1, )

ax.legend(handles=[line1, line2, err2, err_low, line3], fontsize="small", loc="upper right")

plt.savefig(output_dir / "degraded_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "degraded_sensitivity.pdf", format="PDF", bbox_inches="tight")
plt.show()
