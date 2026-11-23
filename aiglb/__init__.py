"""aiglb — the tiny bridge between LLMs and binary glTF (.glb) files.

Write a short Python script describing parts, materials and transforms;
``scene.save("model.glb")`` emits a textured, PBR .glb that renders in any
glTF viewer (three.js, Blender, model viewers, game engines).

    import aiglb

    scene = aiglb.Scene("robot")
    scene.add("body", aiglb.box(0.7, 0.9, 0.4), "steel", position=(0, 1.02, 0))
    scene.add("head", aiglb.sphere(0.3), "plastic_white", position=(0, 1.65, 0))
    scene.add("eye",  aiglb.sphere(0.05), "neon_cyan", position=(0, 1.68, 0.26))
    scene.save("robot.glb")

See README.md — it is written to be read by an AI agent from top to bottom.
"""

from .primitives import (
    box, plane, cylinder, cone, tube, sphere, capsule, torus,
    lathe, extrude, wedge,
)
from .materials import (
    color_material, texture_material, library_material, preset_material,
    list_presets, PRESET_NAMES,
)
from .textures import list_textures, TEXTURE_NAMES
from .library import (
    list_library, library_categories, library_info,
)
from .ops import (
    transformed, concat, mirror, linear_array, radial_array,
    box_project, transform_matrix,
)
from .humanize import (
    rounded_box, superellipsoid, displace, humanize, subdivide,
)
from .fit import (
    touch, place_on, fit_diameter, fit_box, bounds_gap, connection_report,
)
from .scene import Scene, Part
from .validate import inspect_glb

__version__ = "0.2.0"

__all__ = [
    # primitives
    "box", "plane", "cylinder", "cone", "tube", "sphere", "capsule", "torus",
    "lathe", "extrude", "wedge",
    "rounded_box", "superellipsoid",
    # materials
    "color_material", "texture_material", "library_material", "preset_material",
    "list_presets", "PRESET_NAMES", "list_textures", "TEXTURE_NAMES",
    # bundled CC0 texture library
    "list_library", "library_categories", "library_info",
    # humanizer
    "displace", "humanize", "subdivide",
    # part mating
    "touch", "place_on", "fit_diameter", "fit_box", "bounds_gap",
    "connection_report",
    # ops
    "transformed", "concat", "mirror", "linear_array", "radial_array",
    "box_project", "transform_matrix",
    # scene & validation
    "Scene", "Part", "inspect_glb",
    "__version__",
]
