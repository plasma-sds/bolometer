import numpy as np
from cherab.tools.observers import BolometerCamera, BolometerFoil, BolometerSlit
from cherab.tools.primitives import axisymmetric_mesh_from_polygon
from cherab.tools.raytransfer import RayTransferPipeline0D
from raysect.core import Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Box, Mesh, Subtract

from axuv.geometry import ORIGIN, XAXIS, YAXIS, toroidal_to_cartesian

# ── Camera hardware constants ────────────────────────────────────────────────
BOX_DEPTH = 0.06  # sensor-to-slit depth [m]
BOX_WIDTH_X = 0.20  # for horizontal (box-shaped) cameras
FRUSTUM_WIDTH_X = 0.08  # for vertical (frustum-shaped) cameras
BOX_HEIGHT_Y = 0.006  # toroidal extent of the box
SLIT_WIDTH_X = 0.0008  # poloidal slit width
SLIT_HEIGHT_Y_HORIZ = 0.003  # toroidal slit height — horizontal cameras (DHT, DHC)
SLIT_HEIGHT_Y_VERT = 0.002  # toroidal slit height — vertical cameras (D16, DVC, D13, …)
SENSOR_X_SIZE = 0.002  # poloidal sensor size
SENSOR_Y_SIZE = 0.005  # toroidal sensor size

# Cameras whose slit uses the horizontal (wider) height
HORIZONTAL_CAMERA_IDS = {"DHT", "DHC"}

# ── Sector → camera mapping ──────────────────────────────────────────────────
# Each entry lists the camera names for that sector.
# "box" cameras (horizontal) are built with make_axuv_camera_box.
# "frustum" cameras (vertical) are built with make_axuv_camera and are always
# physically split into 3 sub-units of 16 channels each.
SECTOR_CAMERAS = {
    "S5":  {"box": ["DHC"], "frustum": ["DVC"]},
    "S16": {"box": ["DHT"], "frustum": ["D16"]},
    "S1":  {"box": [],      "frustum": ["D01"]},
    "S13": {"box": [],      "frustum": ["D13"]},
    "S15": {"box": [],      "frustum": ["D15"]},
}

# Set the pipeline to be used by the diodes
PIPELINES = [RayTransferPipeline0D()]

# ── Spectral configuration ───────────────────────────────────────────────────
SPECTRAL_BINS = 100  # number of spectral bins in each spectrum part
MIN_WAVELENGTHS = [1, 12.4, 124.0]  # nm  (photon energies: 1240, 100, 10 eV)
MAX_WAVELENGTHS = [12.4, 124.0, 1240.0]  # nm  (photon energies:  100,  10,  1 eV)


def get_sensor_data(sensor, axuv_df, channelIDX=None):
    """
    Returns angles, distances, signal names, and the camera's forward/up
    vectors and origin in Cartesian coordinates.

    :param sensor:     camera name string, e.g. "DHT", "D16", "DVC"
    :param axuv_df:    the loaded AXUV geometry DataFrame
    :param channelIDX: if not None, selects channels in range(channelIDX+1, channelIDX+17)
                       (used for split cameras like D16, DVC)
    """
    df_0 = axuv_df.loc[axuv_df["Cam"].str.contains(sensor, na=False)]
    df = (
        df_0.loc[df_0["chan"].isin(range(channelIDX + 1, channelIDX + 17))]
        if channelIDX is not None
        else df_0
    )

    angles      = df['alpha'].to_numpy()
    distances   = df['d(Folie-Blende)'].to_numpy()
    signalnames = df['RAW'].to_list()

    n = len(df)
    df_mid1 = df.iloc[n // 2 - 1]
    df_mid2 = df.iloc[n // 2]

    p1a = toroidal_to_cartesian(
        df_mid1["R_start"], df_mid1["Phi_start"], df_mid1["z_start"]
    )
    p2a = toroidal_to_cartesian(df_mid1["R_end"], df_mid1["Phi_end"], df_mid1["z_end"])
    p1b = toroidal_to_cartesian(
        df_mid2["R_start"], df_mid2["Phi_start"], df_mid2["z_start"]
    )
    p2b = toroidal_to_cartesian(df_mid2["R_end"], df_mid2["Phi_end"], df_mid2["z_end"])

    camera_origin = Vector3D(*p1a)

    v1 = (p2a - p1a) / np.linalg.norm(p2a - p1a)
    v2 = (p2b - p1b) / np.linalg.norm(p2b - p1b)
    bisector = (v1 + v2) / np.linalg.norm(v1 + v2)

    phi1a = df_mid1["Phi_start"]
    up_v = np.array([-np.sin(np.deg2rad(phi1a)), np.cos(np.deg2rad(phi1a)), 0])
    return (
        angles,
        distances,
        signalnames,
        Vector3D(*bisector),
        camera_origin,
        Vector3D(*up_v),
    )


def make_axuv_camera(
    sensor_angles, sensor_distances, signalnames, slit_id, detector_id_start=0
):
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
        Point3D(outer_back_x, -outer_back_y, -outer_depth),
        Point3D(outer_back_x, outer_back_y, -outer_depth),
        Point3D(-outer_back_x, outer_back_y, -outer_depth),
    ]

    top_vertices = [
        Point3D(-top_x, -top_y, material_width / 2),
        Point3D(top_x, -top_y, material_width / 2),
        Point3D(top_x, top_y, material_width / 2),
        Point3D(-top_x, top_y, material_width / 2),
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
        vertices=vertices, triangles=faces, closed=True, smoothing=False
    )

    # To subratct a slightly smaller chucnk from inside
    bottom_vertices_inner = [
        Point3D(-inner_back_x, -inner_back_y, -inner_depth),
        Point3D(inner_back_x, -inner_back_y, -inner_depth),
        Point3D(inner_back_x, inner_back_y, -inner_depth),
        Point3D(-inner_back_x, inner_back_y, -inner_depth),
    ]

    top_vertices_inner = [
        Point3D(-top_x_inner, -top_y_inner, -material_width / 2),
        Point3D(top_x_inner, -top_y_inner, -material_width / 2),
        Point3D(top_x_inner, top_y_inner, -material_width / 2),
        Point3D(-top_x_inner, top_y_inner, -material_width / 2),
    ]

    # Combine into one list
    vertices_inner = [
        [v.x, v.y, v.z] for v in (bottom_vertices_inner + top_vertices_inner)
    ]

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
        vertices=vertices_inner, triangles=faces_inner, closed=True, smoothing=False
    )

    # Hollow out the box, in this case frustum
    camera_box = Subtract(camera_box_outer, camera_box_inner)

    aperture = Box(
        lower=Point3D(-SLIT_WIDTH_X / 2, -slit_y / 2, -1e-5),
        upper=Point3D(SLIT_WIDTH_X / 2, slit_y / 2, 1e-5),
    )
    camera_box = Subtract(camera_box, aperture)

    camera_box.material = AbsorbingSurface()
    # Create the camera object
    diode_camera = BolometerCamera(camera_geometry=camera_box)

    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(
        slit_id=slit_id,
        centre_point=ORIGIN,
        basis_x=XAXIS,
        dx=SLIT_WIDTH_X,
        basis_y=YAXIS,
        dy=slit_y,
        parent=diode_camera,
    )

    print(
        f"  slit created for {diode_camera.name}: {slit_id}, centre_point: {ORIGIN}, basis_x: {XAXIS}, dx: {SLIT_WIDTH_X}, basis_y: {YAXIS}, dy: {slit_y}"
    )

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
        diode_transform = translate(diode_x, 0, diode_z) * rotate_basis(
            forward=Vector3D(-diode_x, 0, -diode_z), up=YAXIS
        )

        diode_id = "{} #{} {}".format(
            slit_id, detector_id_start + j + 1, signalnames[j]
        )

        print(f"{j + 1}/{len(sensor_angles)}    diode: {diode_id}")
        diode = BolometerFoil(
            detector_id=diode_id,
            centre_point=ORIGIN.transform(diode_transform),
            units="Power",
            basis_x=XAXIS.transform(diode_transform),
            dx=SENSOR_X_SIZE,
            basis_y=YAXIS.transform(diode_transform),
            dy=SENSOR_Y_SIZE,
            slit=slit,
            parent=diode_camera,
            accumulate=False,
            curvature_radius=0,
        )

        # spectral settings
        diode.pipelines = PIPELINES

        # Adding the specific diode to the camera
        diode_camera.add_foil_detector(diode)

    return diode_camera


def make_axuv_camera_box(
    sensor_angles, sensor_distances, signalnames, slit_id, detector_id_start=0
):
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
    camera_box = Box(
        lower=Point3D(-BOX_WIDTH_X / 2, -BOX_HEIGHT_Y / 2, -BOX_DEPTH),
        upper=Point3D(BOX_WIDTH_X / 2, BOX_HEIGHT_Y / 2, 0),
    )

    # Hollow out the box
    inside_box = Box(
        lower=camera_box.lower + Vector3D(1e-5, 1e-5, 1e-5),
        upper=camera_box.upper - Vector3D(1e-5, 1e-5, 1e-5),
    )
    camera_box = Subtract(camera_box, inside_box)

    # The slit is a hole in the box
    if slit_id in ["DHT", "DHC"]:
        slit_height_y = SLIT_HEIGHT_Y_HORIZ
    else:
        slit_height_y = SLIT_HEIGHT_Y_VERT

    aperture = Box(
        lower=Point3D(-SLIT_WIDTH_X / 2, -slit_height_y / 2, -1e-4),
        upper=Point3D(SLIT_WIDTH_X / 2, slit_height_y / 2, 1e-4),
    )
    camera_box = Subtract(camera_box, aperture)

    camera_box.material = AbsorbingSurface()
    # Create the camera object
    diode_camera = BolometerCamera(camera_geometry=camera_box)

    # The bolometer slit in this instance just contains targeting information
    # for the ray tracing, since we have already given our camera a geometry
    # The slit is defined in the local coordinate system of the camera
    slit = BolometerSlit(
        slit_id=slit_id,
        centre_point=ORIGIN,
        basis_x=XAXIS,
        dx=SLIT_WIDTH_X,
        basis_y=YAXIS,
        dy=slit_height_y,
        parent=diode_camera,
    )
    print(
        f"  slit: {slit_id}, centre_point: {ORIGIN}, basis_x: {XAXIS}, dx: {SLIT_WIDTH_X}, basis_y: {YAXIS}, dy: {slit_height_y} created for {diode_camera.name}"
    )

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
        diode_transform = translate(diode_x, 0, diode_z) * rotate_basis(
            forward=Vector3D(-diode_x, 0, -diode_z), up=YAXIS
        )
        diode_id = "{} #{} {}".format(
            slit_id, detector_id_start + j + 1, signalnames[j]
        )
        print(f"{j + 1}/{len(sensor_angles)}    diode: {diode_id}")
        diode = BolometerFoil(
            detector_id=diode_id,
            centre_point=ORIGIN.transform(diode_transform),
            units="Power",
            basis_x=XAXIS.transform(diode_transform),
            dx=SENSOR_X_SIZE,
            basis_y=YAXIS.transform(diode_transform),
            dy=SENSOR_Y_SIZE,
            slit=slit,
            parent=diode_camera,
            accumulate=False,
            curvature_radius=0,
        )

        # spectral settings
        diode.pipelines = PIPELINES

        # Adding the specific diode to the camera
        diode_camera.add_foil_detector(diode)

    return diode_camera


def create_observable_world(sectors, axuv_df, cad_mesh=False, show_plots=False, etendue_mode=False):
    """
    Build a Raysect world containing AXUV cameras for the requested sectors.

    :param sectors:   list of sector strings, e.g. ["S5"], ["S16"], or ["S5", "S16"]
    :param axuv_df:   the loaded AXUV geometry DataFrame (from io.load_axuv_geometry)
    :param pipelines: list of Raysect pipelines to attach to every diode
    :param cad_mesh:  if True, import the full AUG CAD mesh; otherwise use a
                      simple absorbing bounding box
    :param show_plots: if True, plot lines of sight after construction
    """
    print(f"Creating world for sectors: {sectors}")
    world = World()
    cameras = []

    for sector in sectors:
        config = SECTOR_CAMERAS[sector]
        print(f"  sector: {sector}, box: {config['box']}, frustum: {config['frustum']}")

        for cam_name in config["box"]:
            angles, distances, names, fwd, origin, up = get_sensor_data(
                cam_name, axuv_df
            )
            cam = make_axuv_camera_box(angles, distances, names, cam_name)
            cam.transform = translate(*origin) * rotate_basis(forward=fwd, up=up)
            cam.parent = world
            cam.name = cam_name
            cameras.append(cam)
            print(f"    box camera: {cam_name} created")

        for cam_name in config["frustum"]:
            for i in range(3):
                angles, distances, names, fwd, origin, up = get_sensor_data(
                    cam_name, axuv_df, channelIDX=i * 16
                )
                c_name = f"{cam_name}_{i + 1}"
                cam = make_axuv_camera(
                    angles, distances, names, c_name, detector_id_start=i * 16
                )
                cam.transform = translate(*origin) * rotate_basis(forward=fwd, up=up)
                cam.parent = world
                cam.name = c_name
                cameras.append(cam)
                print(f"    frustum camera: {c_name} created")

    if cad_mesh:
        from axuv.cad_files import import_aug_mesh

        import_aug_mesh(world=world)
    elif not etendue_mode:
        wall_polygon = [[1, -1.2], [2.5, -1.2], [2.5, 1.2], [1, 1.2]]
        wall_mesh = axisymmetric_mesh_from_polygon(wall_polygon)
        wall_mesh.parent = world
        wall_mesh.material = AbsorbingSurface()
    elif etendue_mode:
        pass

    if show_plots:
        from axuv.plotting import show_camera_lines_of_sight

        show_camera_lines_of_sight(cameras)

    return world, cameras
