import os
from pathlib import Path

from raysect.optical.library.metal import RoughTungsten
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Mesh, import_stl

CADMESH_PATH = Path(__file__).parent.parent


VESSEL = [
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s02-03.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s04-05.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s06-07.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s08-09.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s10-11.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s12-13.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s14-15.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/vessel_s16-01.rsm"),
]

INNER_HEAT_SHIELD = [
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s01.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s02.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s03.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s04.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s05.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s06.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s07.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s08.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s09.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s10.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s11.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s12.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s13.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s14.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s15.rsm"),
    os.path.join(CADMESH_PATH, "aug/vessel/inner_heat_shield_s16.rsm"),
]

# PSL = [os.path.join(CADMESH_PATH, 'aug/vessel/PSL_s01-16_2010_10_21.rsm')]

ICRH = [
    os.path.join(CADMESH_PATH, "aug/icrh/icrh_s02.rsm"),
    os.path.join(CADMESH_PATH, "aug/icrh/icrh_s04.rsm"),
    os.path.join(CADMESH_PATH, "aug/icrh/icrh_s10.rsm"),
    os.path.join(CADMESH_PATH, "aug/icrh/icrh_s12.rsm"),
]

DIVERTOR = [
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s02-03.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s04-05.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s06-07.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s08-09.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s10-11.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s12-13.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s14-15.rsm"),
    os.path.join(CADMESH_PATH, "aug/divertor/divertor_s16-01.rsm"),
]

A_B_COILS = [
    os.path.join(CADMESH_PATH, "aug/a_b_coils/A-coils_s01-16_2010_10_21.rsm"),
    os.path.join(CADMESH_PATH, "aug/a_b_coils/cover_Bl-coils_s01-s16.rsm"),
    os.path.join(
        CADMESH_PATH, "aug/a_b_coils/cover_Bu-coils_s01-s16_modified.rsm"
    ),  # one small section removed
]  # that would cover some D16 LOS


AUG_FULL_MESH = VESSEL + INNER_HEAT_SHIELD + ICRH + DIVERTOR + A_B_COILS


def import_aug_mesh(world, material=RoughTungsten(0.29)):
    """Imports the full AUG first wall mesh with default materials.

    :param str material: flag for material setting to us. 'ABSORBING' indicates
      the mesh should be loaded with a perfectly absorbing surface.
    """

    material = material or AbsorbingSurface()

    for mesh_path in AUG_FULL_MESH:
        print("importing {}  ...".format(os.path.split(mesh_path)[1]))
        directory, filename = os.path.split(mesh_path)
        mesh_name, ext = filename.split(".")
        if ext == "rsm":
            Mesh.from_file(mesh_path, parent=world, material=material, name=mesh_name)
        else:
            import_stl(
                mesh_path,
                parent=world,
                material=material,
                name=mesh_name,
                scaling=0.001,
            )
