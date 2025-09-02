# This script is used for calculating sensitivity matrices for all AXUV didoes in DHT and D16

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
from mpl_toolkits.mplot3d import Axes3D

from raysect.core import Point2D, Point3D, Vector3D, rotate_basis, translate
from raysect.core.math.function.float import Interpolator2DArray
from raysect.optical import World, Spectrum
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Subtract, Mesh
from raysect.optical.observer import SpectralPowerPipeline0D

from cherab.openadas import OpenADAS
from cherab.core import Species, Maxwellian, Plasma, Line
from cherab.core.math import sample3d, AxisymmetricMapper
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine, Bremsstrahlung
from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil
from cherab.tools.primitives import axisymmetric_mesh_from_polygon
from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D

from cad_files import *


plt.rcParams.update({'font.size': 14, "figure.dpi" : 150,
                     'figure.constrained_layout.use': True})
#%matplotlib widget
plt.close('all')

DATADIR = str(os.environ.get("datadir")) + "/"
CURRENTDIR = DATADIR + str(os.environ.get("currentdir")) + "/"
SAVEDIR = CURRENTDIR + "output/"
INPUT_FILENAME = str(sys.argv[1])

RAYTRANSFER_PATH = DATADIR + "raytransfer_S16_reflections.h5"
USE_CAD_MESH = True

# Set the pipeline to be used by the diodes
PIPELINES = [RayTransferPipeline0D()]

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
BOX_DEPTH = 0.06  # sensor to slit distances are around 0.04-0.05
BOX_WIDTH_X = 0.2  # for horizontal cameras
FRUSTUM_WIDTH_X = 0.08  # for vertical cameras
BOX_HEIGHT_Y = 0.006  # toroidal
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

# loading plasma facing components
with open("data/gc_d_lines.obj", "rb") as fp:
    # in a list of 2-element (R, z) lists of arrays 
    gc_d_lines = pickle.load(fp)

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
    """Function to get the data value at a given axis (x, y) coordinate from an imshow() plot"""
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

def show_camera_lines_of_sight(cameralist, gc_d_lines=gc_d_lines):
    """
    Plots diode lines of sight with plasma facing components. 
    Takes a list of BolometerCamera objects.
    Also shows slit center locations.
    """
    _, ax = plt.subplots(figsize=[4,5])
    for camera in cameralist:
        for foil in camera.foil_detectors:
            # print(foil.slit.centre_point)
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

def show_camera_lines_of_sight_3D(cameralist):
    """
    Plots diode lines of sight with plasma facing components. 
    Takes a list of BolometerCamera objects.
    Also shows slit center locations.
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for camera in cameralist:
        for foil in camera.foil_detectors:
            origin, hit, _ = foil.trace_sightline()
            # LOS
            ax.plot([origin.x, hit.x], [origin.y, hit.y], [origin.z, hit.z], 'r', lw=.5)
            ax.plot(foil.slit.centre_point.x, foil.slit.centre_point.y, foil.slit.centre_point.z, 'ko')
            ax.plot(foil.centre_point.x, foil.centre_point.y, foil.centre_point.z, 'kx')

    ax.axis('equal')
    ax.view_init(elev=30, azim=120)
    plt.show()

# World creation
def toroidal_to_cartesian(R, phi, z):
    X = R * np.cos(np.deg2rad(phi))
    Y = R * np.sin(np.deg2rad(phi))
    Z = z

    cartesian_coords = np.array([X, Y, Z])  
    return cartesian_coords

def get_sensor_data(sensor, channelIDX=None):
    """
    Returns data needed for the setup of diodes in a camera
    Also returns camera normal vector, and camera origin in 3D cartesian coordinates

    If one passes channelIDX, this can be used in a for loop
    for cameras which have 16 sensors for one slit, for example D16
    to extract data for channel numbers in range(channelIDX + 1, channelIDX + 17)
    
    If channelIDX is None, this can be used
    for cameras, in which all the sensors are in one camera with one slit
    for example DHT
    """

    df_0 = AXUV_DF[AXUV_DF['Cam'].str.contains(sensor, na=False)]
    if channelIDX is not None:
        df = df_0[df_0['chan'].isin(range(channelIDX + 1, channelIDX + 17))]
    else:
        df = df_0
    angles = df['alpha'].to_numpy()
    distances = df['d(Folie-Blende)'].to_numpy()
    signalnames = df['RAW'].to_list()

    n = len(df)
    i1, i2 = n // 2 - 1, n // 2

    df_mid1 = df.iloc[i1]
    df_mid2 = df.iloc[i2]

    R1a, phi1a, z1a = df_mid1['R_start'], df_mid1['Phi_start'], df_mid1['z_start']
    R2a, phi2a, z2a = df_mid1['R_end'], df_mid1['Phi_end'], df_mid1['z_end']

    R1b, phi1b, z1b = df_mid2['R_start'], df_mid2['Phi_start'], df_mid2['z_start']
    R2b, phi2b, z2b = df_mid2['R_end'], df_mid2['Phi_end'], df_mid2['z_end']


    origin_cartesian = toroidal_to_cartesian(R1a, phi1a, z1a)  # New - toroidal_to_cartesian is a custom function
    camera_origin = Vector3D(*origin_cartesian)  # New - Vector3D is from raysect

    p1a = np.array(toroidal_to_cartesian(R1a, phi1a, z1a))
    p2a = np.array(toroidal_to_cartesian(R2a, phi2a, z2a))

    p1b = np.array(toroidal_to_cartesian(R1b, phi1b, z1b))
    p2b = np.array(toroidal_to_cartesian(R2b, phi2b, z2b))

    v1 = p2a - p1a
    v2 = p2b - p1b

   # Normalize both (important for true bisector)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Compute bisector
    bisector = v1 + v2
    bisector /= np.linalg.norm(bisector)

    forward_v = bisector

    up_v = np.array([-np.sin(np.deg2rad(phi1a)), np.cos(np.deg2rad(phi1a)), 0])  # toroidal tangential
    return angles, distances, signalnames, Vector3D(*forward_v), camera_origin, Vector3D(*up_v)

def make_axuv_camera(sensor_angles, sensor_distances, signalnames, slit_id, detector_id_start=0):
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
    # In this application, the diodes are located in the X-Z plane (y=0)

    # The slit is a hole in the box
    if slit_id in ["DHT", "DHC"]:
        slit_y = SLIT_HEIGHT_Y_HORIZ
    else:
        slit_y = SLIT_HEIGHT_Y_VERT

    slit_x = SLIT_WIDTH_X
    top_x = slit_x  # just to have a small edge around the slit
    top_y = slit_y 

    inner_back_x = FRUSTUM_WIDTH_X / 2
    inner_back_y = BOX_HEIGHT_Y / 2
    inner_depth = BOX_DEPTH

    material_width = 1e-6
    outer_back_x = inner_back_x + material_width
    outer_back_y = inner_back_y + material_width
    outer_depth = inner_depth + material_width

    top_x_inner = top_x - material_width
    top_y_inner = top_y - material_width

    bottom_vertices = [
        Point3D(-outer_back_x, -outer_back_y, -outer_depth),
        Point3D( outer_back_x, -outer_back_y, -outer_depth),
        Point3D( outer_back_x,  outer_back_y, -outer_depth),
        Point3D(-outer_back_x,  outer_back_y, -outer_depth)
    ]

    top_vertices = [
        Point3D(-top_x, -top_y, material_width / 2),
        Point3D( top_x, -top_y, material_width / 2),
        Point3D( top_x,  top_y, material_width / 2),
        Point3D(-top_x,  top_y, material_width / 2)
    ]

    # Combine into one list
    vertices = [[v.x, v.y, v.z] for v in (bottom_vertices + top_vertices)]


    # Indexing:
    # bottom: 0-3, top: 4-7
    faces = [
        # Bottom face (two triangles)
        (0, 1, 2), (0, 2, 3),
        
        # Top face (two triangles)
        (4, 6, 5), (4, 7, 6),

        # Sides (4 sides × 2 triangles each)
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),

        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7)
    ]

    # Create the mesh
    camera_box_outer = Mesh(
        vertices=vertices,
        triangles=faces,
        closed=True,
        smoothing=False
    )

    # To subratct a slightly smaller chucnk from inside
    bottom_vertices_inner = [
        Point3D(-inner_back_x, -inner_back_y, -inner_depth),
        Point3D( inner_back_x, -inner_back_y, -inner_depth),
        Point3D( inner_back_x,  inner_back_y, -inner_depth),
        Point3D(-inner_back_x,  inner_back_y, -inner_depth)
    ]

    top_vertices_inner = [
        Point3D(-top_x_inner, -top_y_inner, -material_width / 2),
        Point3D( top_x_inner, -top_y_inner, -material_width / 2),
        Point3D( top_x_inner,  top_y_inner, -material_width / 2),
        Point3D(-top_x_inner,  top_y_inner, -material_width / 2)
    ]

    # Combine into one list
    vertices_inner = [[v.x, v.y, v.z] for v in (bottom_vertices_inner + top_vertices_inner)]

    # Indexing:
    # bottom: 0-3, top: 4-7
    faces_inner = [
        # Bottom face (two triangles)
        (0, 1, 2), (0, 2, 3),
        
        # Top face (two triangles)
        (4, 6, 5), (4, 7, 6),

        # Sides (4 sides × 2 triangles each)
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),

        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7)
    ]

    # Create the mesh
    camera_box_inner = Mesh(
        vertices=vertices_inner,
        triangles=faces_inner,
        closed=True,
        smoothing=False
    )

    # Hollow out the box, in this case frustum
    camera_box = Subtract(camera_box_outer, camera_box_inner)

    # The slit is a hole in the box
    if slit_id in ["DHT", "DHC"]:
        slit_height_y = SLIT_HEIGHT_Y_HORIZ
    else:
        slit_height_y = SLIT_HEIGHT_Y_VERT

    aperture = Box(lower=Point3D(-SLIT_WIDTH_X / 2, -slit_height_y / 2, -1e-5),
                   upper=Point3D(SLIT_WIDTH_X / 2, slit_height_y / 2, 1e-5))
    camera_box = Subtract(camera_box, aperture)

    camera_box.material = AbsorbingSurface()
    # Create the camera object
    diode_camera = BolometerCamera(camera_geometry=camera_box)

    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(slit_id=slit_id, centre_point=ORIGIN,
                         basis_x=XAXIS, dx=SLIT_WIDTH_X, basis_y=YAXIS, dy=slit_y,
                         parent=diode_camera)
    
    for j, angle in enumerate(sensor_angles):
        # A number of diodes, spaced based on their distance from the slit 
        # and the angle measured from the Z-axis
        # The diode positions and orientations are given in the local coordinate system of the camera
        distance_from_slit = sensor_distances[j]
        angle_rad = np.deg2rad(angle)
        diode_x = distance_from_slit * np.sin(angle_rad)
        diode_z = -(distance_from_slit * np.cos(angle_rad))

        # rotate_basis(): forward: Z-axis of object
        #                 up: Y-axis
        #                 X: defined by Z and Y so that a right-handed orthogonal coordinate system is created
        diode_transform = translate(diode_x, 0, diode_z) * rotate_basis(forward=Vector3D(-diode_x, 0, -diode_z), up=YAXIS)
        diode = BolometerFoil(detector_id="{} #{} {}".format(slit_id, detector_id_start + j + 1, signalnames[j]),
                             centre_point=ORIGIN.transform(diode_transform), units="Power",
                             basis_x=XAXIS.transform(diode_transform), dx=SENSOR_X_SIZE,
                             basis_y=YAXIS.transform(diode_transform), dy=SENSOR_Y_SIZE,
                             slit=slit, parent=diode_camera,accumulate=False, curvature_radius=0)

        # spectral settings
        diode.pipelines = PIPELINES
        
        # Adding the specific diode to the camera
        diode_camera.add_foil_detector(diode)

    return diode_camera

def make_axuv_camera_box(sensor_angles, sensor_distances, signalnames, slit_id, detector_id_start=0):
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

        # rotate_basis(): forward: Z-axis of object
        #                 up: Y-axis
        #                 X: defined by Z and Y so that a right-handed orthogonal coordinate system is created
        diode_transform = translate(diode_x, 0, diode_z) * rotate_basis(forward=Vector3D(-diode_x, 0, -diode_z), up=YAXIS)
        diode = BolometerFoil(detector_id="{} #{} {}".format(slit_id, detector_id_start + j + 1, signalnames[j]),
                             centre_point=ORIGIN.transform(diode_transform), units="Power",
                             basis_x=XAXIS.transform(diode_transform), dx=SENSOR_X_SIZE,
                             basis_y=YAXIS.transform(diode_transform), dy=SENSOR_Y_SIZE,
                             slit=slit, parent=diode_camera,accumulate=False, curvature_radius=0)

        # spectral settings
        diode.pipelines = PIPELINES
        
        # Adding the specific diode to the camera
        diode_camera.add_foil_detector(diode)

    return diode_camera

def create_observable_world(cad_mesh=USE_CAD_MESH, show_plots=False):
    """
    Creates world with cameras in sector 16 (DHT, D16)

    Can be changed to work for other sectors by adding the other camera names
    However, one would have to also apply toroidal rotation to those cameras
    """
    #################################
    # Set up scenegraph
    print("Creating world with cameras...")
    world = World()

    # Set up horizontal (DHT) and vertical (D16) AXUV cameras in sector 16 (SPI sector)
    cameras = []
    for camera_name in ["DHT"]:
        angles, distances, signalnames, forward_v, camera_origin, up_v = get_sensor_data(camera_name)
        camera = make_axuv_camera_box(angles, distances, signalnames, camera_name)
        
        # Rotate and move camera into position
        transform_camera = translate(*camera_origin) * rotate_basis(forward=forward_v, up=up_v)
        camera.transform = transform_camera
        camera.parent = world
        camera.name = camera_name
        cameras.append(camera)

    for camera_name in ["D16"]:
        for i in range(3):
            angles, distances, signalnames, forward_v, camera_origin, up_v = get_sensor_data(camera_name, channelIDX=i*16)
            c_name = camera_name + "_" + str(i+1)
            camera = make_axuv_camera(angles, distances, signalnames, c_name)

            # Rotate and move camera into position
            transform_camera = translate(*camera_origin) * rotate_basis(forward=forward_v, up=up_v)
            camera.transform = transform_camera
            camera.parent = world
            camera.name = c_name
            cameras.append(camera)

    if cad_mesh is True:
        import_aug_mesh(world=world)
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

if __name__ == "__main__":
    with h5py.File(INPUT_FILENAME, "r") as f:
        majorR = f["R"][()][:, np.newaxis]
        zaxis = f["Z"][()][:, np.newaxis]

        neon0 = f["Ne0"][()][:, np.newaxis]
        neon1 = f["Ne1"][()][:, np.newaxis]
        neon2 = f["Ne2"][()][:, np.newaxis]
        neon3 = f["Ne3"][()][:, np.newaxis]
        neon4 = f["Ne4"][()][:, np.newaxis]
        neon5 = f["Ne5"][()][:, np.newaxis]
        neon6 = f["Ne6"][()][:, np.newaxis]
        neon7 = f["Ne7"][()][:, np.newaxis]
        neon8 = f["Ne8"][()][:, np.newaxis]
        neon9 = f["Ne9"][()][:, np.newaxis]
        neon10 = f["Ne10"][()][:, np.newaxis]
        
        eTemp = f["Te"][()][:, np.newaxis]
        eDens = f["ne"][()][:, np.newaxis]
        try:
            SI_time = f["time"][()][0]
        except:
            SI_time = INPUT_FILENAME.strip(CURRENTDIR+'/input/output_step').strip('_out.h5')

    # Remove negative values
    neon0[neon0 < 0] = 0
    neon1[neon1 < 0] = 0
    neon2[neon2 < 0] = 0
    neon3[neon3 < 0] = 0
    neon4[neon4 < 0] = 0
    neon5[neon5 < 0] = 0
    neon6[neon6 < 0] = 0
    neon7[neon7 < 0] = 0
    neon8[neon8 < 0] = 0
    neon9[neon9 < 0] = 0
    neon10[neon10 < 0] = 0
    eTemp[eTemp < 0] = 0
    eDens[eDens < 0] = 0 

    neonlist = [neon0, neon1, neon2, neon3, neon4, neon5, neon6, neon7, neon8, neon9, neon10]

    # Setting up interpolation of JOREK data
    # In this case the vertical and horizontal distances between the gridpoints will be the same
    # Later the voxel grid will have the same dimensions, but will be masked where there is no plasma
    # this results in ~4 GB memory allocation for the creation of the ~4000 element voxel grid 
    # NOTE Doubling the total number of grid points results in a 2^2=4 times increase in the memory needed!
    # NOTE Doubling the resolution in both directions results in a (2*2)^2=16 times increase!
    resolution_R = 120
    resolution_z = 220

    plasma_res_R=int(resolution_R)
    plasma_res_z=int(resolution_z)

    # The points at which the JOREK data is defined
    points = np.hstack([majorR, zaxis])

    i_neon, i_eTemp, i_eDens = interpolate_parameters(points, plasma_res_R, plasma_res_z, neonlist,
                                                    eTemp, eDens, method="linear")

    try:
        with h5py.File(RAYTRANSFER_PATH, 'r') as h5f:
            sensitivity_matrix = h5f["sensitivity_matrix"][()]
            grid_centres = h5f["grid_centres"][()]
            voxel_map = h5f["voxel_map"][()]
            inverse_voxel_map = h5f["inverse_voxel_map"][()]
            laplacian = h5f["laplacian"][()]
            mask = h5f["mask"][()]
            completed_bins = h5f["completed_bins"][()]
        print("Sensitivity matrix loaded")
    except:
        print("Sensitivity matrix file not found!")

    # A convex hull is created around the JOREK datapoints to be used as boundary for the voxel grid later
    hull = ConvexHull(points)
    convex_hull = points[hull.vertices]

    world, cameras = create_observable_world(cad_mesh=False, show_plots=False)    

    print("Creating plasma...")
    plasma = Plasma(parent=world)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    plasma_mesh = axisymmetric_mesh_from_polygon(convex_hull)
    plasma.geometry = plasma_mesh

    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)

    interpolated_eDens = i_eDens
    interpolated_eTemp = i_eTemp
    interpolated_neon = i_neon

    # No net velocity for any species
    zero_velocity = Vector3D(0, 0, 0)

    deuterium_mass = elements.deuterium.atomic_weight * atomic_mass
    neon_mass = elements.neon.atomic_weight * atomic_mass

    extrap_x = 0.1
    extrap_y = 0.1

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
    plasma.models = [
        *neon_lines,
        Bremsstrahlung()
    ]

    # define species, field and composition
    plasma.b_field = Vector3D(0, 0, 0)
    plasma.electron_distribution = e_distribution
    plasma.composition = [ne0_species, ne1_species, ne2_species, ne3_species, ne4_species,
                        ne5_species, ne6_species, ne7_species, ne8_species, ne9_species,
                        ne10_species, de1_species]

    # Define spectral measurements array - has to be size: num of diodes by spectral bins
    NUM_OF_DIODES = sensitivity_matrix.shape[0]
    SPECTRAL_BINS = 100
                                          # Approx photon energies in eV
    MIN_WAVELENGTHS = [0.25, 12.4, 124]   # 5000, 100, 10
    MAX_WAVELENGTHS = [12.4, 124, 1240]   # 100, 10, 1

    def get_spectrum_part(part):
        return np.linspace(MIN_WAVELENGTHS[part], MAX_WAVELENGTHS[part], SPECTRAL_BINS)

    wavelengths = np.unique(np.array([*get_spectrum_part(0),*get_spectrum_part(1),*get_spectrum_part(2)]))
    energies_eV = 1239.8 / wavelengths
    total_wavelength_bins = len(wavelengths) - 1

    def emission_function_3d_v3(x, y, z, part):
        """
        Provides emission in units of W m^-3 sr^-1 nm^-1
        """
        spectrum = Spectrum(MIN_WAVELENGTHS[part], MAX_WAVELENGTHS[part], SPECTRAL_BINS-1)
        direction = Vector3D(0, 0, 1)
        emission = np.zeros(SPECTRAL_BINS-1)

        for model in plasma.models:
            point = Point3D(x, y, z)
            emission += model.emission(point, direction, spectrum.new_spectrum()).samples

        return emission
        
    emissions = np.zeros([inverse_voxel_map.shape[0], total_wavelength_bins])

    for i in range(inverse_voxel_map.shape[0]):
        # Get the indices of the i-th voxel in the grid_centres array
        aa = inverse_voxel_map[i, 0, 0]
        cc = inverse_voxel_map[i, 2, 0]

        # Get the real world coordinates corresponding to the i-th voxel
        xi = grid_centres[aa, cc, 0]  # To get R coordinate
        yi = 0  # Assume y coordinate is 0
        zi = grid_centres[aa, cc, 1]  # To get z coordinate

        emission_in_point = np.zeros(total_wavelength_bins)
        for part in range(3):
            emission_in_point[part*99:(part+1)*99] = emission_function_3d_v3(xi, yi, zi, part)

        emissions[i, :] = emission_in_point
        print(str(i)+"/"+str(inverse_voxel_map.shape[0]), end="\r")

    # measured_spectra = np.zeros([NUM_OF_DIODES, total_wavelength_bins])
    # for i in range(NUM_OF_DIODES):
    #     for j in range(total_wavelength_bins):
    #         measured_spectra[i, j] = np.sum(sensitivity_matrix[i, :, j] * emissions[:, j])

    # Saving the emission data as HDF5
    with h5py.File(SAVEDIR + "raytransfer_emissions_" + "{:.6f}".format(SI_time) + ".h5", "w") as file:
        file.create_dataset("emissions", data=emissions)
        file.create_dataset("wavelengths", data=wavelengths)
        file.create_dataset("energies", data=energies_eV)
        # file.create_dataset("diode_measurements", data=measured_spectra)

    print("\nSaved emission data.")
