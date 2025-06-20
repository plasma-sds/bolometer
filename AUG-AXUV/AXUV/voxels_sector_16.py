# This script is meant to be used for emission calculation in voxels, 
# and determining the radiation spectra reaching individual AXUV diodes
# by multiplying the values in the voxels with the sensitivity matrix calculated for the diodes.
# This method is fundamentally deterministic and relatively fast.
# In this example one timestep of a JOREK simulation of a D-Ne SPI is used to determine emissions.  

import os
import sys
import csv
import h5py
import math
import pickle
import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cherab.core.atomic.elements as elements

from matplotlib import cm
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.constants import electron_mass, atomic_mass
from mpl_toolkits.axes_grid1 import make_axes_locatable

from raysect.core import Point2D, Point3D, Vector3D, rotate_basis, translate
from raysect.core.math.function.float import Interpolator2DArray
from raysect.optical import World, Spectrum
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Subtract
from raysect.optical.observer import SpectralPowerPipeline0D

from cherab.openadas import OpenADAS
from cherab.core import Species, Maxwellian, Plasma, Line
from cherab.core.math import sample3d, AxisymmetricMapper
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine, Bremsstrahlung
from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil
from cherab.tools.primitives import axisymmetric_mesh_from_polygon
from cherab.tools.inversions import ToroidalVoxelGrid


plt.rcParams.update({'font.size': 16, "figure.dpi" : 150,
                     'figure.constrained_layout.use': True})

DATADIR = str(os.environ.get("datadir")) + "/"
CURRENTDIR = DATADIR + str(os.environ.get("currentdir")) + "/"
SAVEDIR = CURRENTDIR + "output/"
INPUT_FILENAME = str(sys.argv[1])
print(INPUT_FILENAME)

# for dir in [CURRENTDIR, SAVEDIR, CURRENTDIR + "stdout/"]:
#     if not os.path.exists(dir):
#         os.mkdir(dir)

VOXEL_PATH = DATADIR + "voxel_grid_S16_new.pickle"
USE_CAD_MESH = False
WALL_MATERIAL = AbsorbingSurface()

# Set the pipeline to be used by the diodes
PIPELINES = [SpectralPowerPipeline0D(accumulate=False)]

# The emitted spectra will be calculated in 3xSPECTRAL_BINS number of total spectral bins.
# The spectrum is divided into 3 parts defined by the MIN_WAVELENGTHS and MAX_WAVELENGTHS lists.
# In each part the spectrum is divided linearly in wavelength to SPECTRAL_BINS number of bins.
# The current setup is optimal for AXUV diodes in case there are high-Z impurities.
SPECTRAL_BINS = 100
                                      # Approx photon energies in eV
MIN_WAVELENGTHS = [0.25, 12.4, 124]   # 5000, 100, 10
MAX_WAVELENGTHS = [12.4, 124, 1240]   # 100, 10, 1

# Loading diode geometry data into pandas dataframe for easier filtering
AXUV_DATAFILE = DATADIR + "AXUV_LOS_geom.txt"
try:
    AXUV_DF = pd.read_csv(AXUV_DATAFILE, sep=r"\s+", engine='python').drop(columns=["act", "con", "F", "Foil_ID", 
                                                                                    "R_Kabel", "U_Gen.", "Faktor"])
except:
    print("Could not load AXUV geometry datafile")
    
# Setting up geometry limits in the poloidal cross section
POLOIDAL_RMIN = 1
POLOIDAL_RMAX = 2.2
POLOIDAL_ZMIN = -1.2
POLOIDAL_ZMAX = 1

# Universal geometry data for the AXUV diodes and their boxes and pinholes (aka slits)
BOX_DEPTH = 0.05  # sensor to slit distances are around 0.04-0.05
BOX_WIDTH_X = 0.1  # should not matter as long as it can accomodate all the diodes
BOX_HEIGHT_Y = 0.008
SLIT_WIDTH_X = 0.0008  # poloidal
SLIT_HEIGHT_Y_HORIZ = 0.003  # toroidal for horizontal cameras DHT, DHC
SLIT_HEIGHT_Y_VERT = 0.002   # toroidal for vertical cameras DVC, D16, D13, D15, D01
SENSOR_X_SIZE = 0.002  # poloidal
SENSOR_Y_SIZE = 0.005  # toroidal

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)

# 3D to 2D routine
def _point3d_to_rz(point):
    return Point2D(math.hypot(point.x, point.y), point.z)

# Interpolation
def interpolate_param(param, points, grid_R, grid_z, method="linear"):
    """Interpolates parameters from the JOREK grid to the rectangular grid"""
    return np.nan_to_num(griddata(points, param, (grid_R, grid_z), method=method)[:, :, 0]).T

def interpolate_parameters(points, resolution_R, resolution_z, neonlist, eTemp, eDens, method="linear"):
    # The new grid for interpolation
    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)
    grid_R, grid_z = np.meshgrid(linspace_R, linspace_z)

    print("Interpolating densities and temperatures...")
    # Interpolating the Neon charge state densities
    interpolated_neon = np.zeros([11, resolution_R, resolution_z])
    for i, neon in enumerate(neonlist):
        interpolated_neon[i, :, :] = interpolate_param(neon, points, grid_R, grid_z, method)

    # Interpolating the remaining parameters
    interpolated_eTemp = interpolate_param(eTemp, points, grid_R, grid_z, method)
    interpolated_eDens = interpolate_param(eDens, points, grid_R, grid_z, method)
    return interpolated_neon, interpolated_eTemp, interpolated_eDens

# Plotting
def plot_interpolated(interpolated, title=None, cbarlabel=None, gc_d_lines=None, show=True):
    """
    Plots interpolated values along with the contours of plasma facing components.
    Takes the values, the figure title and colorbar label as parameters.
    """
    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=[6, 6.5])
    ax1.set_aspect(1)
    ax1.set_xlim(1, 2.2)
    ax1.set_ylim(-1.2, 1)
    ims = ax1.imshow(interpolated.T, extent=(1, 2.2, -1.2, 1), origin="lower")
    ax1.set_xlabel("R [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_title(title)
    
    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax1.plot(line[0], line[1], lw=.5, c="white")
    cbar = plt.colorbar(ims)
    cbar.set_label(cbarlabel)
    if show:
        plt.show()
    else:
        return ax1, ims

def get_value_at(im, x, y):
    # Function to get the data value at a given axis (x, y) coordinate from an imshow() plot
    extent = im.get_extent()
    x_min, x_max, y_min, y_max = extent
    data = im.get_array()
    
    rows, cols = data.shape

    # Convert axis coordinates (x, y) to array indices
    col = int(round((x - x_min) / (x_max - x_min) * (cols - 1)))
    row = int(round((y - y_min) / (y_max - y_min) * (rows - 1)))

    # Ensure indices are within bounds
    if 0 <= row < rows and 0 <= col < cols:
        return data[row, col]
    else:
        return None  # Out of bounds

def compare_plot_interpolated(parameter, interpolated, title=None):
    """
    Visually compares the JOREK input data, which is given on a poloidal grid
    to the interpolated data given on a rectangular grid.
    NOTE that the JOREK data is shown on a scatter plot which means datapoints overlap!
    This comparison is not needed, the code has been tested, it is only here for peace of mind.

    :param numpy.ndarray parameter: The JOREK values
    :param numpy.ndarray interpolated: The lineraly interpolated values (onto a 2D rectangular grid)
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[5, 4])
    ax1.set_aspect(1)
    ax2.set_aspect(1)
    ax2.set_xlim(1, 2.2)
    ax2.set_ylim(-1.2, 1)
    ax1.imshow(interpolated.T, extent=(1, 2.2, -1.2, 1), origin="lower")
    ax1.set_xlabel("R")
    ax1.set_ylabel("z")
    ax2.set_xlabel("R")
    # ax2.set_ylabel("z")
    
    scatter = ax2.scatter(points[:, 0], points[:, 1], c=parameter, s=20)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0)
    cax.axis('off')
    if title is not None:
        fig.suptitle(title)
    plt.colorbar(scatter, cax=cax2)
    plt.show()

def show_camera_lines_of_sight(cameralist, gc_d_lines=None):
    """
    Plots diode lines of sight with plasma facing components. 
    Takes a list of BolometerCamera objects
    """
    _, ax = plt.subplots(figsize=[4,5])
    for camera in cameralist:
        for foil in camera.foil_detectors:
            # print(foil.name)
            slit_centre = foil.slit.centre_point
            slit_centre_rz = _point3d_to_rz(slit_centre)
            ax.plot(slit_centre_rz[0], slit_centre_rz[1], 'ko')
            origin, hit, _ = foil.trace_sightline()
            centre_rz = _point3d_to_rz(foil.centre_point)
            ax.plot(centre_rz[0], centre_rz[1], 'kx')
            origin_rz = _point3d_to_rz(origin)
            hit_rz = _point3d_to_rz(hit)
            ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], 'r', lw=.5)

    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax.plot(line[0], line[1], lw=.5, c="k")
    ax.set_xlabel("R")
    ax.set_ylabel("z")
    ax.set_title("Diode lines of sight")
    ax.axis('equal')

    ax.set_xlim(0.95, 2.4)
    ax.set_ylim(-1.2, 1.2)
    plt.show()

def show_LOS_doublets(cameralist, axuv_df, gc_d_lines=None):
    """
    Visually compare the lines of sight of the created diodes 
    to the lines of sight defined in the AXUV geometry data file.
    Takes a list of cameras, uses the AXUV dataframe.
    For testing purposes.
    """
    _, ax = plt.subplots(figsize=[4,5])
    for camera in cameralist:
        for foil in camera.foil_detectors:
            # print(foil.name)
            slit_centre = foil.slit.centre_point
            slit_centre_rz = _point3d_to_rz(slit_centre)
            ax.plot(slit_centre_rz[0], slit_centre_rz[1], 'ko')
            origin, hit, _ = foil.trace_sightline()
            centre_rz = _point3d_to_rz(foil.centre_point)
            ax.plot(centre_rz[0], centre_rz[1], 'kx')
            origin_rz = _point3d_to_rz(origin)
            hit_rz = _point3d_to_rz(hit)
            ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], 'r', lw=1)

    df_0 = axuv_df[axuv_df['Cam'].str.contains("DHT", na=False)]
    df_1 = axuv_df[axuv_df['Cam'].str.contains("D16", na=False)]

    R1 = df_0['R_start']
    R2 = df_0['R_end']
    z1 = df_0['z_start']
    z2 = df_0['z_end']
    ax.plot([R1, R2], [z1, z2], ls="--", c="k", lw=1)

    R1 = df_1['R_start']
    R2 = df_1['R_end']
    z1 = df_1['z_start']
    z2 = df_1['z_end']
    ax.plot([R1, R2], [z1, z2], ls="--", c="k", lw=1)

    if gc_d_lines is not None:
        for line in gc_d_lines:
            ax.plot(line[0], line[1], lw=.5, c="k")
    ax.set_xlabel("R")
    ax.set_ylabel("z")
    ax.set_title("Diode lines of sight")
    ax.axis('equal')

    ax.set_xlim(0.95, 2.4)
    ax.set_ylim(-1.2, 1.2)
    plt.show()

def plot_world(interpolated_eDens, interpolated_eTemp):
    """
    Plots interpolated electron temperature and density, shows the diode lines of sight 
    and plots of the created plasma
    """
    plot_interpolated(interpolated_eTemp, title="Electron temperature", cbarlabel=r"T$_e$ [eV]")
    plot_interpolated(interpolated_eDens, title="Electron density", cbarlabel=r"n$_e$ [m$^{-3}$]")
    world, _ = create_observable_world(cad_mesh=USE_CAD_MESH, show_plots=True)
    create_plasma(world, show_plots=True)

# World creation
# Based on Cherab voxel based bolometry example restructured for AXUV didodes on AUG
def get_sensor_data(sensor, axuv_df, channelIDX=None):
    """
    Returns data needed for the setup of diodes in a camera from the AXUV dataframe
    Also returns camera normal vector, and camera origin

    If one passes channelIDX, this can be used in a for loop
    for cameras which have 16 sensors for one slit, for example D16
    to extract data for channel numbers in range(channelIDX + 1, channelIDX + 17)
    
    If channelIDX is None, this can be used for cameras, in which all the sensors 
    are in one camera with one slit for example DHT
    """

    df_0 = axuv_df[axuv_df['Cam'].str.contains(sensor, na=False)]
    if channelIDX is not None:
        df = df_0[df_0['chan'].isin(range(channelIDX + 1, channelIDX + 17))]
    else:
        df = df_0
    angles = df['alpha'].to_numpy()
    distances = df['d(Folie-Blende)'].to_numpy()
    signalnames = df['RAW'].to_list()

    df_mid = df.iloc[0]
    R1 = df_mid['R_start']
    R2 = df_mid['R_end']
    z1 = df_mid['z_start']
    z2 = df_mid['z_end']

    alpha = angles[0]

    v_ext = np.array([R1-R2, z1-z2])
    camera_origin = np.array([R1, z1])

    beta = np.rad2deg(np.arctan(v_ext[1]/v_ext[0]))

    # Needed due to np.arctan() returning values in ]-pi/2, pi/2[
    if beta < 0:
        beta = 180 + beta

    gamma = 180 - alpha + beta
    v_n_camera = np.array([np.cos(np.deg2rad(gamma)), np.sin(np.deg2rad(gamma))])

    return angles, distances, signalnames, v_n_camera, camera_origin

def make_axuv_camera(sensor_angles, sensor_distances, signalnames, slit_id, detector_id_start=0, spectrum_part=0):
    """
    Creates and returns an AXUV camera (box, slit, diodes) as a BolometerCamera object

    :param sensor_angles: angles from AXUV dataframe (alpha in the AXUV LOS geometry file)
    :param sensor_distances: distances from AXUV dataframe (d(Folie-Blende))
    :param signalnames: raw signal names e.g. S1L0A01 (RAW)
    :param slit_id: In some cases one camera has actually three different boxes and slits
      and in that case three different BolometerCamera objects need to be created
    :param detector_id_start: Internal identifier of the diode number, when there is only
      one box in a camera, the diodes are indexed 0-47, and to keep with this notation 
      when there are three boxes in a camera, the indexing goes 0-15, 16-31, 32-47
    :param spectrum_part: Which part of the spectrum needs to be simulated 
      (see details above SPECTRAL_BINS declaration)
    """
    # First let us define the box of the diodes. In the AXUV datafile "AXUV_LOS_geom.txt"
    # "alpha" is the angle of the specific diodes from the main Z-axis of the camera
    # "d(Folie-Blende)" is the distance of the diodes from the origin in thew camera coordinates
    # A camera consists of a box with a rectangular slit and 48 or 16 sensors (diodes).
    # In its local coordinate system, the camera's slit is located at the
    # origin and the sensors below the X-Y plane (z=0), looking up towards the slit.
    #
    #               Z-axis
    #                |
    #   -------------|-------------
    #   | $ $ $ $ $ $|$ $ $ $ $ $ |    $ signs note example diode locations for AXUV cameras   
    #   |            |            |    
    #   |            |            |    Y-axis points outwards from the screen to the viewer
    #   |            |            |       
    #   -----------  O  ----------- -> X-axis
    #               _|_ The pinhole and the ORIGIN are located at O
    #               \ /
    #                ˇ
    # In this application, the diodes are located in the X-Z plane (y=0)

    # Create a box
    camera_box = Box(lower=Point3D(-BOX_WIDTH_X / 2, -BOX_HEIGHT_Y / 2, -BOX_DEPTH),
                     upper=Point3D(BOX_WIDTH_X / 2, BOX_HEIGHT_Y / 2, 0))
    
    # Hollow out the box
    inside_box = Box(lower=camera_box.lower + Vector3D(1e-5, 1e-5, 1e-5),
                     upper=camera_box.upper - Vector3D(1e-5, 1e-5, 1e-5))
    camera_box = Subtract(camera_box, inside_box)

    # The slit is a hole in the box
    if slit_id in ["DHT", "DHC"]:
        slit_height_y = SLIT_HEIGHT_Y_HORIZ
    else:
        slit_height_y = SLIT_HEIGHT_Y_VERT

    aperture = Box(lower=Point3D(-SLIT_WIDTH_X / 2, -slit_height_y / 2, -1e-4),
                   upper=Point3D(SLIT_WIDTH_X / 2, slit_height_y / 2, 1e-4))
    camera_box = Subtract(camera_box, aperture)

    camera_box.material = AbsorbingSurface()
    # Create the camera object
    diode_camera = BolometerCamera(camera_geometry=camera_box)

    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(slit_id=slit_id, centre_point=ORIGIN,
                         basis_x=XAXIS, dx=SLIT_WIDTH_X, basis_y=YAXIS, dy=slit_height_y,
                         parent=diode_camera)
    
    for j, angle in enumerate(sensor_angles):
        # A number of diodes, spaced based on their distance from the slit 
        # and the angle measured from the Z-axis
        # The diode positions and orientations are given in the local coordinate system of the camera
        distance_from_slit = sensor_distances[j]
        angle_rad = np.deg2rad(angle)
        diode_x = distance_from_slit * np.sin(angle_rad)
        diode_z = -(distance_from_slit * np.cos(angle_rad))

        # rotate_basis(): -forward: Z-axis of object
        #                 -up: Y-axis
        #                 -X defined by Z and Y so that a right-handed orthogonal coordinate system is created
        diode_transform = translate(diode_x, 0, diode_z) * rotate_basis(forward=Vector3D(-diode_x, 0, -diode_z), up=YAXIS)
        diode = BolometerFoil(detector_id="{} #{} {}".format(slit_id, detector_id_start + j + 1, signalnames[j]),
                             centre_point=ORIGIN.transform(diode_transform), units="Power",
                             basis_x=XAXIS.transform(diode_transform), dx=SENSOR_X_SIZE,
                             basis_y=YAXIS.transform(diode_transform), dy=SENSOR_Y_SIZE,
                             slit=slit, parent=diode_camera,accumulate=False, curvature_radius=0)

        # spectral settings
        diode.spectral_bins = SPECTRAL_BINS
        diode.min_wavelength = MIN_WAVELENGTHS[spectrum_part]
        diode.max_wavelength = MAX_WAVELENGTHS[spectrum_part]
        diode.pipelines = PIPELINES
        
        # Adding the specific diode to the camera
        diode_camera.add_foil_detector(diode)

    return diode_camera

def create_observable_world(axuv_df, spectrum_part=0, cad_mesh=USE_CAD_MESH, show_plots=False):
    """
    Creates world with cameras in sector 16 (DHT, D16)

    Can be changed to work for other sectors by adding the other camera names
    However, one would have to also apply toroidal rotation to those cameras
    """
    # Set up scenegraph
    print("Creating world with cameras...")
    world = World()

    # Set up horizontal (DHT) and vertical (D16) AXUV cameras in sector 16 (SPI sector)
    cameras = []
    for camera_name in ["DHT"]:
        angles, distances, signalnames, v_n_camera, camera_origin = get_sensor_data(camera_name, axuv_df)
        camera = make_axuv_camera(angles, distances, signalnames, camera_name, spectrum_part=spectrum_part)

        # Rotate and move camera into position
        forward_v = Vector3D(v_n_camera[0], 0, v_n_camera[1])
        transform_camera = translate(camera_origin[0], 0, camera_origin[1]) * rotate_basis(forward=forward_v, up=YAXIS)
        camera.transform = transform_camera
        camera.parent = world
        camera.name = camera_name
        cameras.append(camera)

    for camera_name in ["D16"]:
        # cameras need to be moved a bit toroidally so that camera boxes don't intersect with each other
        y_offset = [-0.05, 0, 0.05]
        for i in range(3):
            angles, distances, signalnames, v_n_camera, camera_origin = get_sensor_data(camera_name, axuv_df, channelIDX=i*16)
            c_name = camera_name + "_" + str(i+1)
            camera = make_axuv_camera(angles, distances, signalnames, c_name, detector_id_start=i*16, spectrum_part=spectrum_part)

            # Rotate and move camera into position
            forward_v = Vector3D(v_n_camera[0], 0, v_n_camera[1])
            transform_camera = translate(camera_origin[0], y_offset[i], camera_origin[1]) * rotate_basis(forward=forward_v, up=YAXIS)
            camera.transform = transform_camera
            camera.parent = world
            camera.name = c_name
            cameras.append(camera)

    if cad_mesh is True:
        print("Using CAD mesh not implemented yet")
    else:
        # NOTE Set up primitive rectangle bounding box as first wall
        # If we do not care about reflections and therefore use AbsorbingSurface(),
        # then it also does not matter which shape the toroidal bounding box is
        # The only important thing is to cover the radiation coming from the other side of the torus
        wall_polygon = [
            [1, -1.2],    # R1, z1
            [2.5, -1.2],  # R2, z2
            [2.5, 1.2],   # R3, z3
            [1, 1.2]      # R4, z4
        ]

        # rotate the bounding rectangle toroidally
        wall_mesh = axisymmetric_mesh_from_polygon(wall_polygon)
        wall_mesh.parent = world
        wall_mesh.material = AbsorbingSurface()  # fully absorbing surface, no reflections

    # # Check camera lines of sight visually
    if show_plots:
        show_camera_lines_of_sight(cameras)

    return world, cameras

def create_plasma(world, interpolated_eDens, interpolated_eTemp, interpolated_neon, 
                  resolution_R, resolution_z, show_plots=False):
    """
    Creates toroidal plasma in the simulated world with deuterium and neon
    """
    print("Creating plasma...")
    plasma = Plasma(parent=world)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    plasma_mesh = axisymmetric_mesh_from_polygon(convex_hull)
    plasma.geometry = plasma_mesh

    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)

    # No net velocity for any species
    zero_velocity = Vector3D(0, 0, 0)

    deuterium_mass = elements.deuterium.atomic_weight * atomic_mass
    neon_mass = elements.neon.atomic_weight * atomic_mass

    extrap_x = 3
    extrap_y = 1

    # Calculate D1 density from quasi-neutrality
    calculated_d1 = (interpolated_eDens - interpolated_neon[1, :, :] - 2 * interpolated_neon[2, :, :] 
                    - 3 * interpolated_neon[3, :, :] - 4 * interpolated_neon[4, :, :] - 5 * interpolated_neon[5, :, :]
                    - 6 * interpolated_neon[6, :, :] - 7 * interpolated_neon[7, :, :] - 8 * interpolated_neon[8, :, :]
                    - 9 * interpolated_neon[9, :, :] - 10 * interpolated_neon[10, :, :])

    # create 2D interpolators for the densities and temperature
    e_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_eDens, 
                                           interpolation_type="linear", extrapolation_type="nearest", 
                                           extrapolation_range_x=extrap_x, extrapolation_range_y=extrap_y)
    e_temperature_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_eTemp, "linear", "nearest", extrap_x, extrap_y)

    ne0_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[0, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne1_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[1, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne2_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[2, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne3_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[3, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne4_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[4, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne5_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[5, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne6_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[6, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne7_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[7, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne8_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[8, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne9_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[9, :, :], "linear", "nearest", extrap_x, extrap_y)
    ne10_density_interp = Interpolator2DArray(linspace_R, linspace_z, interpolated_neon[10, :, :], "linear", "nearest", extrap_x, extrap_y)

    de1_density_interp = Interpolator2DArray(linspace_R, linspace_z, calculated_d1, "linear", "nearest", extrap_x, extrap_y)

    # map the 2D interpolators into 3D functions using the axisymmetry operator
    e_density = AxisymmetricMapper(e_density_interp)
    e_temperature = AxisymmetricMapper(e_temperature_interp)

    ne0_density = AxisymmetricMapper(ne0_density_interp)
    ne1_density = AxisymmetricMapper(ne1_density_interp)
    ne2_density = AxisymmetricMapper(ne2_density_interp)
    ne3_density = AxisymmetricMapper(ne3_density_interp)
    ne4_density = AxisymmetricMapper(ne4_density_interp)
    ne5_density = AxisymmetricMapper(ne5_density_interp)
    ne6_density = AxisymmetricMapper(ne6_density_interp)
    ne7_density = AxisymmetricMapper(ne7_density_interp)
    ne8_density = AxisymmetricMapper(ne8_density_interp)
    ne9_density = AxisymmetricMapper(ne9_density_interp)
    ne10_density = AxisymmetricMapper(ne10_density_interp)

    de1_density = AxisymmetricMapper(de1_density_interp)

    # Set up the distributions to be Maxwellians
    e_distribution = Maxwellian(e_density, e_temperature, zero_velocity, electron_mass)

    ne0_distribution = Maxwellian(ne0_density, e_temperature, zero_velocity, neon_mass)
    ne1_distribution = Maxwellian(ne1_density, e_temperature, zero_velocity, neon_mass)
    ne2_distribution = Maxwellian(ne2_density, e_temperature, zero_velocity, neon_mass)
    ne3_distribution = Maxwellian(ne3_density, e_temperature, zero_velocity, neon_mass)
    ne4_distribution = Maxwellian(ne4_density, e_temperature, zero_velocity, neon_mass)
    ne5_distribution = Maxwellian(ne5_density, e_temperature, zero_velocity, neon_mass)
    ne6_distribution = Maxwellian(ne6_density, e_temperature, zero_velocity, neon_mass)
    ne7_distribution = Maxwellian(ne7_density, e_temperature, zero_velocity, neon_mass)
    ne8_distribution = Maxwellian(ne8_density, e_temperature, zero_velocity, neon_mass)
    ne9_distribution = Maxwellian(ne9_density, e_temperature, zero_velocity, neon_mass)
    ne10_distribution = Maxwellian(ne10_density, e_temperature, zero_velocity, neon_mass)

    de1_distribution = Maxwellian(de1_density, e_temperature, zero_velocity, deuterium_mass)

    # Define the different plasma species
    ne0_species = Species(elements.neon, 0, ne0_distribution)
    ne1_species = Species(elements.neon, 1, ne1_distribution)
    ne2_species = Species(elements.neon, 2, ne2_distribution)
    ne3_species = Species(elements.neon, 3, ne3_distribution)
    ne4_species = Species(elements.neon, 4, ne4_distribution)
    ne5_species = Species(elements.neon, 5, ne5_distribution)
    ne6_species = Species(elements.neon, 6, ne6_distribution)
    ne7_species = Species(elements.neon, 7, ne7_distribution)
    ne8_species = Species(elements.neon, 8, ne8_distribution)
    ne9_species = Species(elements.neon, 9, ne9_distribution)
    ne10_species = Species(elements.neon, 10, ne10_distribution)

    de1_species = Species(elements.deuterium, 1, de1_distribution)

    ##############################################################
    # Get Neon lines from Photon Emissivity Coefficients datafiles
    ##############################################################
    neon_lines = []
    with open(DATADIR + "ne.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            i, part1, part2 = row

            neon_lines.append(ExcitationLine(Line(elements.neon, int(i), (part1, part2)), lineshape=GaussianLine))
            neon_lines.append(RecombinationLine(Line(elements.neon, int(i), (part1, part2)), lineshape=GaussianLine))

    # add all neon lines to the plasma + Bremsstrahlung
    # NOTE hydrogen lines only relevant if data has neutral deuterium species
    plasma.models = [
        # ExcitationLine(hydrogen_I_410, lineshape=GaussianLine),
        # RecombinationLine(hydrogen_I_410, lineshape=GaussianLine),
        *neon_lines,
        Bremsstrahlung()
    ]

    # define species, field and composition
    plasma.b_field = Vector3D(0, 0, 0)
    plasma.electron_distribution = e_distribution
    plasma.composition = [ne0_species, ne1_species, ne2_species, ne3_species, ne4_species,
                          ne5_species, ne6_species, ne7_species, ne8_species, ne9_species,
                          ne10_species, de1_species]

    # Show top-down temperature and poloidal cross section temperature
    if show_plots:
        _, ax = plt.subplots()
        coord = 2.5
        _, _, _, t_samples = sample3d(e_temperature, (-coord, coord, 400), (-coord, coord, 400), (0, 0, 1))
        plt.imshow(np.squeeze(t_samples), extent=[-coord, coord, -coord, coord], origin="lower")
        m = cm.ScalarMappable(cmap="viridis")
        m.set_array(t_samples)
        cbar = plt.colorbar(m, ax=ax)
        cbar.set_label("Electron temperature [eV]")
        plt.axis('equal')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.title("Electron temperature in the x-y plane")
        plt.show()

        _, ax = plt.subplots(figsize=[8,3])
        _, _, _, t_samples = sample3d(e_temperature, (-2.2, 2.2, 2200), (0, 0, 1), (-1.2, 1.0, 400))
        plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[-2.2, 2.2, -1.2, 1.0], origin='lower')
        m = cm.ScalarMappable(cmap="viridis")
        m.set_array(t_samples)
        cbar = plt.colorbar(m, ax=ax)
        cbar.set_label("Electron temperature [eV]")
        plt.axis('equal')
        plt.xlabel('x axis')
        plt.ylabel('z axis')
        plt.title("Electron temperature in the x-z plane")
        plt.show()

    return plasma

# Voxel creation, loading and sampling
def make_voxels(world, cameras, nx, ny, axuv_df, cad_mesh=USE_CAD_MESH, show_plots=False):
    """
    Set up the voxels and generate the sensitivity matrix
    NOTE It is recommended to use the same poloidal resolution as the interpolated 
    JOREK data (default)

    We'll use a grid of rectangular voxels here, all of which are the same
    size. Neither the shape nor the uniform size are required for using the
    voxels, but it makes this a bit simpler.

    :param world:       The simulated world object that the voxels will be located in
    :param cameras:     list of BolometerCamera objects that observe the world
    :params nx, ny:     x and y resolution of the voxel grid (R and z resolution)
      It is best to use the same resolution that the interpolated data has, 
      since the sampling of the plasma parameters will be done in only 
      one location per voxel to speed up calculations
    :param material: wall material if the CAD mesh is loaded
    :param cad_mesh: to load the CAD mesh of AUG or not
    :param savedir: where to save the pickle object created from the voxel data
    :param show_plots:  Show the sensitivity matrix for all the diodes 
      in the poloidal cross section

    :return voxel_grid:          The created ToroidalVoxelGrid with parent=world
    :return sensitivity_matrix:  A matrix connecting the specific diode lines of sight
      with the voxel cells. Shows how sensitive the diodes are for radiation coming 
      from the specific cells.
    """
    world, cameras = create_observable_world(axuv_df=axuv_df, cad_mesh=cad_mesh)
    print("Producing the voxel grid...")

    # Define the centres of each voxel, as an (nx, ny, 2) array
    dr = (POLOIDAL_RMAX - POLOIDAL_RMIN) / nx
    dz = (POLOIDAL_ZMAX - POLOIDAL_ZMIN) / ny
    cell_r = np.linspace(POLOIDAL_RMIN + dr/2, POLOIDAL_RMAX - dr/2, nx)
    print(cell_r)
    cell_z = np.linspace(POLOIDAL_ZMIN + dz/2, POLOIDAL_ZMAX - dz/2, ny)
    cell_centres = np.swapaxes(np.asarray(np.meshgrid(cell_r, cell_z)), 0, 2)

    cell_dx = cell_centres[1, 0] - cell_centres[0, 0]
    cell_dy = cell_centres[0, 1] - cell_centres[0, 0]
    print(cell_dx, "\n", cell_dy)
    # Define the positions of the vertices of the voxels, as an (nx, ny, 4, 2) array
    cell_vertex_displacements = np.asarray([-0.5 * cell_dx - 0.5 * cell_dy,
                                            -0.5 * cell_dx + 0.5 * cell_dy,
                                            0.5 * cell_dx + 0.5 * cell_dy,
                                            0.5 * cell_dx - 0.5 * cell_dy])

    all_cell_vertices = np.swapaxes(cell_centres[..., None], -2, -1) + cell_vertex_displacements

    # Produce a (ncells, nvertices, 2) array of coordinates to initialise the
    # ToroidalVoxelCollection. Here, ncells = number of cells inside mask,
    # nvertices = 4. The ToroidalVoxelGrid expects a flat list of (nvertices, 2)
    # arrays to define voxels, since there is no implicit assumption that the voxels
    # lie on a grid.

    enclosed_cells = []
    grid_mask = np.empty((nx, ny), dtype=bool)
    grid_index_2D_to_1D_map = {}
    grid_index_1D_to_2D_map = {}

    # Identify the cells that are enclosed by the polygon,
    # simultaneously write out grid mask and grid map.
    unwrapped_cell_index = 0
    for ix in range(nx):
        for iy in range(ny):
            vertices = all_cell_vertices[ix, iy]
            p = cell_centres[ix, iy, :]

            # if any points are inside the polygon, retain this cell
            point = shapely.geometry.Point([p[0], p[1]])
            if polygon.contains(point):
                grid_mask[ix, iy] = True
                # We'll need these maps for generating the regularisation operator
                grid_index_2D_to_1D_map[(ix, iy)] = unwrapped_cell_index
                grid_index_1D_to_2D_map[unwrapped_cell_index] = (ix, iy)
                enclosed_cells.append(vertices)
                unwrapped_cell_index += 1
            else:
                grid_mask[ix, iy] = False


    num_cells = len(enclosed_cells)
    print("Number of cells in the toroidal voxel grid: " + str(num_cells))

    voxel_data = np.empty((num_cells, 4, 2))  # (number of cells, 4 coordinates, x and y values)
    for i, row in enumerate(enclosed_cells):
        p1, p2, p3, p4 = row
        voxel_data[i, 0, :] = p1
        voxel_data[i, 1, :] = p2
        voxel_data[i, 2, :] = p3
        voxel_data[i, 3, :] = p4

    voxel_grid = ToroidalVoxelGrid(voxel_data)

    ###############################################################################
    # Produce a regularisation operator for inversions - from the bolometry example
    ###############################################################################
    # We'll use simple isotropic smoothing here, in which case an ND second
    # derivative operator (the laplacian operator) is appropriate
    grid_laplacian = np.zeros((num_cells, num_cells))

    for ith_cell in range(num_cells):
        # get the 2D mesh coordinates of this cell
        ix, iy = grid_index_1D_to_2D_map[ith_cell]

        neighbours = 0

        try:
            n1 = grid_index_2D_to_1D_map[ix - 1, iy]  # neighbour 1
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n1] = -1
            neighbours += 1

        try:
            n2 = grid_index_2D_to_1D_map[ix - 1, iy + 1]  # neighbour 2
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n2] = -1
            neighbours += 1

        try:
            n3 = grid_index_2D_to_1D_map[ix, iy + 1]  # neighbour 3
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n3] = -1
            neighbours += 1

        try:
            n4 = grid_index_2D_to_1D_map[ix + 1, iy + 1]  # neighbour 4
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n4] = -1
            neighbours += 1

        try:
            n5 = grid_index_2D_to_1D_map[ix + 1, iy]  # neighbour 5
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n5] = -1
            neighbours += 1

        try:
            n6 = grid_index_2D_to_1D_map[ix + 1, iy - 1]  # neighbour 6
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n6] = -1
            neighbours += 1

        try:
            n7 = grid_index_2D_to_1D_map[ix, iy - 1]  # neighbour 7
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n7] = -1
            neighbours += 1

        try:
            n8 = grid_index_2D_to_1D_map[ix - 1, iy - 1]  # neighbour 8
        except KeyError:
            pass
        else:
            grid_laplacian[ith_cell, n8] = -1
            neighbours += 1

        grid_laplacian[ith_cell, ith_cell] = neighbours


    ############################################
    # Calculate the geometry matrix for the grid
    ############################################
    print("Calculating the geometry matrix...")
    # The voxel grid must be in the same world as the bolometers
    voxel_grid.parent = world

    sensitivity_matrix = []
    for camera in cameras:
        for foil in camera:
            print("Calculating sensitivity for {}...".format(foil.name))
            sensitivity_matrix.append(foil.calculate_sensitivity(voxel_grid))
    sensitivity_matrix = np.asarray(sensitivity_matrix)

    # Save the voxel grid information and the geometry matrix for use in other demos
    voxel_grid_data = {'voxel_data': voxel_data, 'laplacian': grid_laplacian,
                       'grid_index_1D_to_2D_map': grid_index_1D_to_2D_map,
                       'grid_index_2D_to_1D_map': grid_index_2D_to_1D_map,
                       'sensitivity_matrix': sensitivity_matrix}

    with open(VOXEL_PATH, "wb") as f:
        pickle.dump(voxel_grid_data, f)
        print("Voxel data saved")

    # Plot the sensitivity matrix, summed over all foils
    if show_plots:
        _, ax = plt.subplots()
        voxel_grid.plot(ax=ax, title="Total sensitivity [m³sr]",
                        voxel_values=sensitivity_matrix.sum(axis=0))
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        plt.show()

    return voxel_grid, sensitivity_matrix

def load_voxels(world, fpath=VOXEL_PATH):
    """
    Loads voxel data from pickle at fpath, returns voxel grid and sensitivity matrix.
    """

    try:
        with open(fpath, "rb") as f:
            grid_data = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            "Geometry data not found: please create the voxels first with make_voxels(world, cameras, *args)."
        )
    voxel_grid = ToroidalVoxelGrid(grid_data['voxel_data'], parent=world)
    sensitivity_matrix = grid_data['sensitivity_matrix']
    print("Voxel data loaded.")
    return voxel_grid, sensitivity_matrix

def sample_voxel_emissions(voxel_grid, i_eDens, i_eTemp, i_neon, resolution_R, resolution_z, axuv_df,
                           use_cad_mesh=USE_CAD_MESH, material=WALL_MATERIAL, savedir=SAVEDIR, 
                           fname_ending="", grid_samples=1):
    """
    Creates the world and the plasma as many times as the number of spectrum parts,
    makes the emission calculations and saves all data to one file.
    Prints progress updates to the terminal.

    :param ToroidalVoxelGrid voxel_grid: the Cherab voxel grid 
      with parent properly set as the simulation world
    :params i_eDens, i_eTemp, i_neon: The interpolated electron density,
      electron temperature and the list of interpolated neon charge state densities
      returned by the interpolate_parameters() function
    :params int resolution_R, resolution_z: R and z resolution of the plasma parameters
    :param bool use_cad_mesh: use CAD mesh of AUG vessel and PFCs - only relevant if
      one wants to consider reflection effects
    :param material: wall material if using CAD mesh, do not use AbsorbingSurface, 
      since then it is pointless to load the CAD mesh, suggestion: RoughTungsten(0.29)
    :param str savedir: where to save the resulting emission HDF5 datafile
    :param fname_ending: filename ending, will be converted to str
    """
    # The sensitivity matrix has units of [m3*sr] when the units used to calculate
    # the sensitivity in BolometerFoil.calculate_sensitivity are "Power". The
    # radiation function we get from the plasma models defines emissivity with units of [W/(m3*sr*nm)]
    # Multiplying the sensitivity matrix with the emission matrix we get [W/nm] on the diodes.

    num_voxels = int(voxel_grid.count)
    # make a <total spectral bins> by <number of voxels> sized numpy array for the results
    total_spectral_bins = int(len(MIN_WAVELENGTHS)*SPECTRAL_BINS)
    voxel_emissions = np.zeros([total_spectral_bins, num_voxels])
    emission_wavelengths = np.zeros(total_spectral_bins)
    dlambdas = np.zeros(len(MIN_WAVELENGTHS))  # wavelength bin widths

    got_diode_names = False
    diodename_list = []
    
    for spectrum_part in range(len(MIN_WAVELENGTHS)):
        world, cameras = create_observable_world(axuv_df, spectrum_part, material, use_cad_mesh)
        voxel_grid, sensitivity_matrix = load_voxels(world)
        # the list of diode names only needs to be populated once
        # this prepares the script for use with cameras other than in the example
        if not got_diode_names:
            for camera in cameras:
                for diode in camera.foil_detectors:
                    diodename_list.append(diode.name)

            got_diode_names = True

        plasma = create_plasma(world, i_eDens, i_eTemp, i_neon, resolution_R, resolution_z)
        print("\nObservation {}/{}".format(spectrum_part+1, len(MIN_WAVELENGTHS)))

        spectrum = Spectrum(MIN_WAVELENGTHS[spectrum_part], MAX_WAVELENGTHS[spectrum_part], SPECTRAL_BINS)
        emission_wavelengths[int(spectrum_part*SPECTRAL_BINS):int((spectrum_part+1)*SPECTRAL_BINS)] = spectrum.wavelengths
        dlambdas[spectrum_part] = spectrum.wavelengths[1] - spectrum.wavelengths[0]
        # This direction would be for the ray-tracer, it is arbitrary, but has to be supplied to the emission model
        direction = Vector3D(0, 0, 1)

        for i in range(SPECTRAL_BINS):
            # calculates the emitted power of all voxel grid cells for the specific spectral bin
            print("Spectral bin {}/{}".format(i+1, SPECTRAL_BINS), end="\r")

            def emission_function_3d(x, y, z):
                """
                A new function has to be defined for each spectral bin, since there are a lot of plasma models 
                (~880 with the OpenADAS neon lines), and from those models we only need the emissions 
                in a specific spectral range

                Provides emission in units of W m^-3 sr^-1 nm^-1
                """
                emission = 0

                for model in plasma.models:
                    point = Point3D(x, y, z)
                    emission += model.emission(point, direction, spectrum.new_spectrum()).samples[i]
            
                return emission

            firstindex = int(spectrum_part * SPECTRAL_BINS + i)
            # Determine the emissivities based on the emission function, which contains the needed parts
            # of the plasma model emissions
            # NOTE the grid_samples=1 means that in each voxel grid cell the emission function will be sampled
            # at only one point. This speeds up the calculation compared to the default 10, but therefore is only
            # reliable if the voxel grid has the same resolution as the interpolated plasma parameters
            voxel_emissions[firstindex, :] = voxel_grid.emissivities_from_function(emission_function_3d, grid_samples=grid_samples)

        print("\n")

    emission_energies = 1239.8 / emission_wavelengths

    NUM_OF_DIODES = sensitivity_matrix.shape[0]
    measured_spectra = np.zeros([NUM_OF_DIODES, total_spectral_bins])
    for i in range(NUM_OF_DIODES):
        for j in range(total_spectral_bins):
            measured_spectra[i, j] = np.sum(sensitivity_matrix[i, :] * voxel_emissions[j, :])

    # Saving the emission data as HDF5
    with h5py.File(savedir + "voxel_emissions_" + str(fname_ending) + ".h5", "w") as file:
        file.create_dataset("emissions", data=voxel_emissions)
        file.create_dataset("wavelengths", data=emission_wavelengths)
        file.create_dataset("energies", data=emission_energies)
        file.create_dataset("wavelength_bin_widths", data=dlambdas)
        file.create_dataset("diode_names", data=diodename_list)
        file.create_dataset("diode_measurements", data=measured_spectra)

    print("\nSaved emission data.")

if __name__ == "__main__":
    # Loading plasma facing components for visualization
    # with open(os.path.join(DATADIR, "gc_d_lines.obj"), "rb") as fp:
    #     # in a list of 2-element (R, z) lists of arrays 
    #     gc_d_lines = pickle.load(fp)
    if AXUV_DF is None:
        raise RuntimeError("Please provide the AXUV geometry datafile")
    
    # Loading data from the custom file generated from JOREK output files
    with h5py.File(INPUT_FILENAME, "r") as f:
        majorR = f["R"][()][:, np.newaxis]      # major radii of points where data is defined
        zaxis = f["Z"][()][:, np.newaxis]       # Z coordinates of points where data is defined

        eTemp = f["Te"][()][:, np.newaxis]      # electron temperature
        eDens = f["ne"][()][:, np.newaxis]      # electron density

        neon0 = f["Ne0"][()][:, np.newaxis]     # Ne-0 density
        neon1 = f["Ne1"][()][:, np.newaxis]     # Ne-1 density
        neon2 = f["Ne2"][()][:, np.newaxis]     # Ne-2 density
        neon3 = f["Ne3"][()][:, np.newaxis]     # Ne-3 density
        neon4 = f["Ne4"][()][:, np.newaxis]     # Ne-4 density
        neon5 = f["Ne5"][()][:, np.newaxis]     # Ne-5 density
        neon6 = f["Ne6"][()][:, np.newaxis]     # Ne-6 density
        neon7 = f["Ne7"][()][:, np.newaxis]     # Ne-7 density
        neon8 = f["Ne8"][()][:, np.newaxis]     # Ne-8 density
        neon9 = f["Ne9"][()][:, np.newaxis]     # Ne-9 density
        neon10 = f["Ne10"][()][:, np.newaxis]   # Ne-10 density

        SI_time = f["time"][()][0]              # time in seconds
        print(SI_time)
        
    mask_negative = True
    resolution_multiplier = 1  # for plasma parameters only, the voxels should stay lower resolution
    interp_method = "linear"
    grid_samples = 1

    print("Interpolation method: " + interp_method)
    print("Resolution multiplier: " + str(resolution_multiplier))
    print("Masking negative values: " + str(mask_negative))
    print("Sampling each voxel " + str(grid_samples) + " times")

    if mask_negative:
        eTemp = np.maximum(eTemp, 1.)
        eDens = np.maximum(eDens, 0)
        neon0 = np.maximum(neon0, 0)
        neon1 = np.maximum(neon1, 0)
        neon2 = np.maximum(neon2, 0)
        neon3 = np.maximum(neon3, 0)
        neon4 = np.maximum(neon4, 0)
        neon5 = np.maximum(neon5, 0)
        neon6 = np.maximum(neon6, 0)
        neon7 = np.maximum(neon7, 0)
        neon8 = np.maximum(neon8, 0)
        neon9 = np.maximum(neon9, 0)
        neon10 = np.maximum(neon10, 0)

    neonlist = [neon0, neon1, neon2, neon3, neon4, neon5, neon6, neon7, neon8, neon9, neon10]

    # Setting up interpolation of JOREK data
    # In this case the vertical and horizontal distances between the gridpoints will be the same
    # Later the voxel grid will have the same dimensions, but will be masked where there is no plasma
    # this results in ~4 GB memory allocation for the creation of the ~4000 element voxel grid 
    # NOTE Doubling the total number of grid points results in a 2^2=4 times increase in the memory needed!
    # NOTE Doubling the resolution in both directions results in a (2*2)^2=16 times increase!
    resolution_R = 60
    resolution_z = 110

    plasma_res_R=int(resolution_R*resolution_multiplier)
    plasma_res_z=int(resolution_z*resolution_multiplier)

    # The points at which the JOREK data is defined
    points = np.hstack([majorR, zaxis])

    i_neon, i_eTemp, i_eDens = interpolate_parameters(points, plasma_res_R, plasma_res_z, neonlist,
                                                      eTemp, eDens, method=interp_method)

    # A convex hull is created around the JOREK datapoints to be used as boundary for the voxel grid later
    hull = ConvexHull(points)
    convex_hull = points[hull.vertices]
    polygon_minimum = shapely.geometry.Polygon(convex_hull)
    polygon = polygon_minimum.buffer(0.02, join_style=2)  # make the polygon a bit bigger (by 2%)

    # Dummy world for the voxels
    world, cameras = create_observable_world(axuv_df=AXUV_DF)
    try:
        voxel_grid, sensitivity_matrix = load_voxels(world)
    except RuntimeError:
        voxel_grid, sensitivity_matrix = make_voxels(world, cameras, axuv_df=AXUV_DF, nx=resolution_R, ny=resolution_z)


    sample_voxel_emissions(voxel_grid, i_eDens, i_eTemp, i_neon, plasma_res_R, plasma_res_z,
                           axuv_df=AXUV_DF, fname_ending="{:.6f}".format(SI_time), grid_samples=grid_samples)

    