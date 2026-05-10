"""
This IPython notebook plots the sightlines for the AXUV cameras in sector S16.
First, the cameras are set up in Cherab, and the lines of sight are plotted against the AUG data.
Second, the publication figure is created that contains the lines of sight and SPI vectors, LCFS and PFC contours.
"""
# %%
import pickle
import matplotlib.pyplot as plt

import aug_sfutils as sf

from axuv.cameras import create_observable_world
from axuv.io import load_axuv_df, PROJECT_ROOT, _GC_LINES_PATH
from axuv.plotting import set_plt_rcparams, show_camera_lines_of_sight
from axuv_measurement.geometry import plot_qsurfaces


# Set default plotting parameters
set_plt_rcparams()

savepath = PROJECT_ROOT / "output" / "sightline_comparison.eps"
savepath_png = PROJECT_ROOT / "output" / "sightline_comparison.png"

# Load AXUV geometry data and PFC component outlines
axuv_df = load_axuv_df()
with open(_GC_LINES_PATH, "rb") as fp:
    gc_d_lines = pickle.load(fp)

# Create Cherab world for setting up cameras
world, cameras = create_observable_world(sectors=["S16"], axuv_df=axuv_df, cad_mesh=True)

# Plot camera lines of sight from the Cherab world
fig, ax = show_camera_lines_of_sight(cameras, show=False)

for index, row in axuv_df.iterrows():
    if row["Cam"] in ["D16", "DHT"]:
        ax.plot([row["R_start"], row["R_end"]], [row["z_start"], row["z_end"]], lw=0.5, c="k", ls='--')

# Plot PFCs
for line in gc_d_lines:
    ax.plot(line[0], line[1], lw=.5, c="k")

plt.savefig(savepath_png, format='png', dpi=300, bbox_inches='tight')
plt.savefig(savepath, format='eps', bbox_inches='tight')

plt.show()
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(6, 8))

# Plot PFCs
for line in gc_d_lines:
    ax.plot(line[0], line[1], lw=.5, c="k")

# Plot an example LCFS
_, _, equ = plot_qsurfaces(40673, 2.3)
sep_Rz = sf.rho2rz(equ, 1, t_in=2.3, coord_in='rho_pol')
sep = ax.plot(sep_Rz[0][0][0], sep_Rz[1][0][0], color="b", ls='--', lw=2, label='LCFS')

# from the axuv_df panndas dataframe, take the R_start, z_start columns and R_end, z_end columns 
# for each row that has either D16 or DHT in "Cam" column and plot them
for index, row in axuv_df.iterrows():
    if row["Cam"] == "D16":
        ax.plot([row["R_start"], row["R_end"]], [row["z_start"], row["z_end"]], lw=0.5, c="r")
    elif row["Cam"] == "DHT":
        ax.plot([row["R_start"], row["R_end"]], [row["z_start"], row["z_end"]], lw=0.5, c="k")

# Annotate the plot with the camera names, but only once for each camera
annotated_D16 = False
annotated_DHT = False
for index, row in axuv_df.iterrows():
    if row["Cam"] == "D16" and not annotated_D16:
        # Annotate the first sightline of D16 in red
        ax.annotate("D16", (row["R_start"], row["z_start"]), color="r", ha='left', va='bottom', xytext=(15, 2), textcoords='offset points')
        annotated_D16 = True
    elif row["Cam"] == "DHT" and not annotated_DHT:
        # Annotate the first sightline of DHT in black 
        ax.annotate("DHT", (row["R_start"], row["z_start"]), color="k", ha='right', va='top', xytext=(0, -50), textcoords='offset points')
        annotated_DHT = True

# Plot SPI vectors
ax.plot([1.5, 2.275708], [-0.153213,  0.308526], lw=2, ls='--', c='k', label='GT1')
ax.plot([1.5, 2.295673], [-0.071579,  0.324483], lw=2, ls='-',  c='k', label='GT2')
ax.plot([1.5, 2.271826], [ 0.0975929, 0.361589], lw=2, ls='-.', c='k', label='GT3')

ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")

ax.set_ylim(-1.2, 1.2)
ax.set_xlim(1.0, 2.35)
ax.set_aspect('equal')

plt.legend()
savepath_png = PROJECT_ROOT / "output" / "sightlines_S16.png"
savepath = PROJECT_ROOT / "output" / "sightlines_S16.eps"
plt.savefig(savepath_png, format='png', dpi=300, bbox_inches='tight')
plt.savefig(savepath, format='eps', bbox_inches='tight')
plt.show()




