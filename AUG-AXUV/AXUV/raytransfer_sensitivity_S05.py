# This script is used for calculating sensitivity matrices for all AXUV didoes in DHC and DVC
# Sector 5 of AUG

from pandas.core.series import Series
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from matplotlib.axes._axes import Axes


from matplotlib.image import AxesImage


from numpy._typing._array_like import NDArray
import os
from typing import Any
import h5py
import math
import pickle
import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

from raysect.core import Point2D, Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Subtract, Mesh

from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil
from cherab.tools.primitives import axisymmetric_mesh_from_polygon
from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D

from cad_files import *


plt.rcParams.update({'font.size': 14, "figure.dpi" : 150,
                     'figure.constrained_layout.use': True})
#%matplotlib widget
plt.close('all')

# data is only needed for masking the sensitivity matrix
DATADIR = "data/"

USE_CAD_MESH = True
WALL_MATERIAL = AbsorbingSurface()

# Set the pipeline to be used by the diodes
PIPELINES: list = [RayTransferPipeline0D()]

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
def plot_interpolated(interpolated, title: str="", cbarlabel: str="", gc_d_lines=None, show=True):
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
    material_width = 1e-6
    top_x = slit_x / 2 + material_width  # let's not have any edge around the slit
    top_y = slit_y / 2 + material_width

    inner_back_x = FRUSTUM_WIDTH_X / 2
    inner_back_y = BOX_HEIGHT_Y / 2
    inner_depth = BOX_DEPTH

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

    aperture = Box(lower=Point3D(-SLIT_WIDTH_X / 2, -slit_height_y / 2, -material_width),
                   upper=Point3D(SLIT_WIDTH_X / 2, slit_height_y / 2, material_width))
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

def create_observable_world_S5(cad_mesh=USE_CAD_MESH, show_plots=False):
    """
    Creates world with cameras in sector 5 (DHC, DVC)

    Can be changed to work for other sectors by adding the other camera names
    However, one would have to also apply toroidal rotation to those cameras
    """
    #################################
    # Set up scenegraph
    print("Creating world with cameras...")
    world = World()

    # Set up horizontal (DHC) and vertical (DVC) AXUV cameras in sector 5
    cameras = []
    for camera_name in ["DHC"]:
        angles, distances, signalnames, forward_v, camera_origin, up_v = get_sensor_data(camera_name)
        camera = make_axuv_camera_box(angles, distances, signalnames, camera_name)
        
        # Rotate and move camera into position
        transform_camera = translate(*camera_origin) * rotate_basis(forward=forward_v, up=up_v)
        camera.transform = transform_camera
        camera.parent = world
        camera.name = camera_name
        cameras.append(camera)

    for camera_name in ["DVC"]:
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

timestep = "02410"
# loading JOREK output data
print(timestep)
fname = "step" + timestep + "_out.h5"
filepath = "data/" + fname
with h5py.File(filepath, "r") as f:
    majorR = f["R"][()][108:, np.newaxis]
    zaxis = f["Z"][()][108:, np.newaxis]

    neon0 = f["Ne0"][()][108:, np.newaxis]
    neon1 = f["Ne1"][()][108:, np.newaxis]
    neon2 = f["Ne2"][()][108:, np.newaxis]
    neon3 = f["Ne3"][()][108:, np.newaxis]
    neon4 = f["Ne4"][()][108:, np.newaxis]
    neon5 = f["Ne5"][()][108:, np.newaxis]
    neon6 = f["Ne6"][()][108:, np.newaxis]
    neon7 = f["Ne7"][()][108:, np.newaxis]
    neon8 = f["Ne8"][()][108:, np.newaxis]
    neon9 = f["Ne9"][()][108:, np.newaxis]
    neon10 = f["Ne10"][()][108:, np.newaxis]
    
    eTemp = f["Te"][()][108:, np.newaxis]
    eDens = f["ne"][()][108:, np.newaxis]

neonlist = [neon0, neon1, neon2, neon3, neon4, neon5, neon6, neon7, neon8, neon9, neon10]

# Setting up interpolation of JOREK data
# In this case the vertical and horizontal distances between the gridpoints will be the same
# Later the voxel grid will have the same dimensions, but will be masked where there is no plasma
# this results in ~4 GB memory allocation for the creation of the ~4000 element voxel grid 
# NOTE Doubling the total number of grid points results in a 2^2=4 times increase in the memory needed!
# NOTE Doubling the resolution in both directions results in a (2*2)^2=16 times increase!
resolution_R = 60
resolution_z = 110

plasma_res_R: int=int(resolution_R)
plasma_res_z: int=int(resolution_z)

# The points at which the JOREK data is defined
points: NDArray[Any] = np.hstack([majorR, zaxis])

i_neon, i_eTemp, i_eDens = interpolate_parameters(points, plasma_res_R, plasma_res_z, neonlist,
                                                    eTemp, eDens, method="linear")

# A convex hull is created around the JOREK datapoints to be used as boundary for the voxel grid later
hull = ConvexHull(points)
convex_hull = points[hull.vertices]
polygon_minimum = shapely.geometry.Polygon(convex_hull)
polygon = polygon_minimum.buffer(0.1, join_style=2)  # make the polygon a bit bigger (by 2%)

# Dummy world for the voxels
world, cameras = create_observable_world_S5(cad_mesh=USE_CAD_MESH, show_plots=False)


########################################################################
# Produce a voxel grid
########################################################################
print("Producing the voxel grid...")
# Define the centres of each voxel, as an (nx, ny, 2) array
nx = resolution_R
ny = resolution_z
cell_r, cell_dx = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nx, retstep=True)
cell_z, cell_dz = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, ny, retstep=True)
cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2) array

# Define the positions of the vertices of the voxels
cell_vertices_r = np.linspace(cell_r[0] - 0.5 * cell_dx, cell_r[-1] + 0.5 * cell_dx, nx + 1)
cell_vertices_z = np.linspace(cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, ny + 1)

# Build a mask, only including cells within the wall
# The inversions will be performed on the emission profile used in the
# radiation_function.py demo, so we'll trim the voxel grid down to the
# emitting region using the shapely polygon

grid_mask = np.empty(shape=(nx, ny), dtype=bool)
for ix in range(nx):
    for iy in range(ny):
        point1 = shapely.geometry.Point([cell_vertices_r[ix], cell_vertices_z[iy]])
        point2 = shapely.geometry.Point([cell_vertices_r[ix+1], cell_vertices_z[iy]])
        point3 = shapely.geometry.Point([cell_vertices_r[ix], cell_vertices_z[iy+1]])
        point4 = shapely.geometry.Point([cell_vertices_r[ix+1], cell_vertices_z[iy+1]])
        if polygon.contains(point1) or polygon.contains(point2) or polygon.contains(point3) or polygon.contains(point4):
            grid_mask[ix, iy] = True
        else:
            grid_mask[ix, iy] = False

# The RayTransferCylinder object is fully 3D, but for simplicity we're only
# working in 2D as this case is axisymmetric. It is easy enough to pass 3D
# views of our 2D data into the RayTransferCylinder object: we just ues a
# numpy.newaxis (or equivalently, None) for the toroidal dimension.
grid_mask = grid_mask[:, None, :]

num_cells = grid_mask.sum()

ray_transfer_grid = RayTransferCylinder(
    radius_outer=cell_vertices_r[-1],
    radius_inner=cell_vertices_r[0],
    height=cell_vertices_z[-1] - cell_vertices_z[0],
    n_radius=nx, n_height=ny, mask=grid_mask, n_polar=1,
    transform=translate(0, 0, cell_vertices_z[0])
)

########################################################################
# Produce a regularisation operator for inversions
########################################################################
# We'll use simple isotropic smoothing here, in which case an ND second
# derivative operator (the laplacian operator) is appropriate. This can be
# produced in the same way as in the geometry matrix with voxels demo, but we
# show a faster vectorised method here.

# Pad the voxel map with a 1-cell-wide border.
voxel_map_with_borders = - np.ones((nx + 2, ny + 2), dtype=int)
voxel_map_with_borders[1:-1, 1:-1] = ray_transfer_grid.voxel_map[:, 0, :]
inverted_voxel_map = ray_transfer_grid.invert_voxel_map()
grid_laplacian = np.zeros((num_cells, num_cells))


for ith_cell in range(num_cells):
    # get the 2D mesh coordinates of this cell
    ix, _, iy = inverted_voxel_map[ith_cell]
    # we didn't map multiple cells into the same light source,
    # so ix and iy are single-element arrays
    ix = ix[0]
    iy = iy[0]

    neighbours_2d = ([ix, ix, ix, ix + 1, ix + 1, ix + 2, ix + 2, ix + 2],
                     [iy, iy + 1, iy + 2, iy, iy + 2, iy, iy + 1, iy + 2])

    neighbours_1d = voxel_map_with_borders[neighbours_2d]
    neighbours_1d = neighbours_1d[neighbours_1d > -1]

    grid_laplacian[ith_cell, neighbours_1d] = -1
    grid_laplacian[ith_cell, ith_cell] = neighbours_1d.size


########################################################################
# Calculate the geometry matrix for the grid
########################################################################
print("Calculating the geometry matrix...")
# The ray transfer object must be in the same world as the bolometers
ray_transfer_grid.parent = world

NUM_OF_DIODES = 96

def get_spectrum_part(part):
    SPECTRAL_BINS = 100
                                        # Approx photon energies in eV
    MIN_WAVELENGTHS = [0.25, 12.4, 124]   # 5000, 100, 10
    MAX_WAVELENGTHS = [12.4, 124, 1240]   # 100, 10, 1

    return np.linspace(MIN_WAVELENGTHS[part], MAX_WAVELENGTHS[part], SPECTRAL_BINS)

wavelengths = np.unique(np.array([*get_spectrum_part(0),*get_spectrum_part(1),*get_spectrum_part(2)]))
energies_eV = 1239.8 / wavelengths
total_wavelength_bins = len(wavelengths) - 1

HDF5_PATH = "raytransfer_S05_reflections_lowres.h5"

# === One-time file setup ===
if not os.path.exists(HDF5_PATH):
    with h5py.File(HDF5_PATH, 'w') as h5f:
        h5f.create_dataset("sensitivity_matrix", shape=(NUM_OF_DIODES, num_cells, total_wavelength_bins), dtype='f8')
        h5f.create_dataset("grid_centres", data=cell_centres)
        h5f.create_dataset("voxel_map", data=ray_transfer_grid.voxel_map)
        h5f.create_dataset("inverse_voxel_map", data=ray_transfer_grid.invert_voxel_map())
        h5f.create_dataset("laplacian", data=grid_laplacian)
        h5f.create_dataset("mask", data=ray_transfer_grid.mask)
        h5f.create_dataset("completed_bins", shape=(total_wavelength_bins,), dtype='i1')  # 0 = not done, 1 = done
        h5f.create_dataset("wavelength_bin_edges", data=wavelengths)
        h5f.create_dataset("energy_bin_edges_eV", data=energies_eV)

# === Main computation loop ===
sensitivity_matrix = np.zeros([NUM_OF_DIODES, num_cells, total_wavelength_bins])

for j in range(total_wavelength_bins):
    # Check if bin j is already completed
    with h5py.File(HDF5_PATH, 'r') as h5f:
        if h5f["completed_bins"][j]:
            print(f"Skipping wavelength bin {j+1} (already completed)")
            continue

    print(f"Calculating for wavelength bin {j+1}/{total_wavelength_bins}")
    i = 0
    for camera in cameras:
        for foil in camera:
            print(f"Calculating sensitivity for {foil.name}...", end="\r")
            foil.pipelines = [RayTransferPipeline0D(kind=foil.units)]
            foil.min_wavelength = wavelengths[j]
            foil.max_wavelength = wavelengths[j+1]
            foil.spectral_bins = ray_transfer_grid.bins
            foil.spectral_rays = 1
            foil.pixel_samples = 1e6
            foil.ray_max_depth = 50
            foil.observe()
            sensitivity_matrix[i, :, j] = foil.pipelines[0].matrix
            i += 1
            foil.max_wavelength = wavelengths[j+1] + 10

    # Open file just to save results for this bin
    with h5py.File(HDF5_PATH, 'r+') as h5f:
        h5f["sensitivity_matrix"][:, :, j] = sensitivity_matrix[:, :, j]
        h5f["completed_bins"][j] = 1
        h5f.flush()


