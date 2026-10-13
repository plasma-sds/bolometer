import json
import os
from pathlib import Path

import numpy as np
import raysect
from raysect.optical import InterpolatedSF
from raysect.optical.material import AbsorbingSurface, RoughConductor
from raysect.primitive import Mesh, import_stl

from axuv.datafiles import data_path


class RoughTungstenExtended(RoughConductor):
    """
    RoughTungsten with n/k data extended to 1 nm (n=1, k=0 anchor point),
    since the sensitive range of the AXUV diodes is ~1-1000 nm.
    """

    def __init__(self, roughness):
        data_path = (
            Path(raysect.__file__).parent / "optical/library/metal/data/tungsten.json"
        )
        with open(data_path) as f:
            data = json.load(f)
        wl = np.concatenate([[1.0], data["wavelength"]])
        n = np.concatenate([[1.0], data["index"]])
        k = np.concatenate([[0.0], data["extinction"]])
        super().__init__(InterpolatedSF(wl, n), InterpolatedSF(wl, k), roughness)


# AUG first-wall CAD meshes, as data keys (see axuv.datafiles)
VESSEL = [
    "cad/vessel/vessel_s02-03.rsm",
    "cad/vessel/vessel_s04-05.rsm",
    "cad/vessel/vessel_s06-07.rsm",
    "cad/vessel/vessel_s08-09.rsm",
    "cad/vessel/vessel_s10-11.rsm",
    "cad/vessel/vessel_s12-13.rsm",
    "cad/vessel/vessel_s14-15.rsm",
    "cad/vessel/vessel_s16-01.rsm",
]

INNER_HEAT_SHIELD = [
    "cad/vessel/inner_heat_shield_s01.rsm",
    "cad/vessel/inner_heat_shield_s02.rsm",
    "cad/vessel/inner_heat_shield_s03.rsm",
    "cad/vessel/inner_heat_shield_s04.rsm",
    "cad/vessel/inner_heat_shield_s05.rsm",
    "cad/vessel/inner_heat_shield_s06.rsm",
    "cad/vessel/inner_heat_shield_s07.rsm",
    "cad/vessel/inner_heat_shield_s08.rsm",
    "cad/vessel/inner_heat_shield_s09.rsm",
    "cad/vessel/inner_heat_shield_s10.rsm",
    "cad/vessel/inner_heat_shield_s11.rsm",
    "cad/vessel/inner_heat_shield_s12.rsm",
    "cad/vessel/inner_heat_shield_s13.rsm",
    "cad/vessel/inner_heat_shield_s14.rsm",
    "cad/vessel/inner_heat_shield_s15.rsm",
    "cad/vessel/inner_heat_shield_s16.rsm",
]

# PSL = ["cad/vessel/PSL_s01-16_2010_10_21.rsm"]

ICRH = [
    "cad/icrh/icrh_s02.rsm",
    "cad/icrh/icrh_s04.rsm",
    "cad/icrh/icrh_s10.rsm",
    "cad/icrh/icrh_s12.rsm",
]

DIVERTOR = [
    "cad/divertor/divertor_s02-03.rsm",
    "cad/divertor/divertor_s04-05.rsm",
    "cad/divertor/divertor_s06-07.rsm",
    "cad/divertor/divertor_s08-09.rsm",
    "cad/divertor/divertor_s10-11.rsm",
    "cad/divertor/divertor_s12-13.rsm",
    "cad/divertor/divertor_s14-15.rsm",
    "cad/divertor/divertor_s16-01.rsm",
]

A_B_COILS = [
    "cad/a_b_coils/A-coils_s01-16_2010_10_21.rsm",
    "cad/a_b_coils/cover_Bl-coils_s01-s16.rsm",
    "cad/a_b_coils/cover_Bu-coils_s01-s16_modified.rsm",  # one small section removed
]  # that would cover some D16 LOS


AUG_FULL_MESH = VESSEL + INNER_HEAT_SHIELD + ICRH + DIVERTOR + A_B_COILS


def import_aug_mesh(world, material=RoughTungstenExtended(0.29)):
    """Imports the full AUG first wall mesh with default materials.

    :param str material: flag for material setting to us. 'ABSORBING' indicates
      the mesh should be loaded with a perfectly absorbing surface.
    """

    material = material or AbsorbingSurface()

    for mesh_key in AUG_FULL_MESH:
        mesh_path = str(data_path(mesh_key))
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
