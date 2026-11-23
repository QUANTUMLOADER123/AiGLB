"""Stylized AK — the "top model" showcase for aiglb.

Every part is mated by measurement, not by eye: wooden parts overlap the
receiver, tape wraps hug the handguards, the stock slides *into* the receiver
by 10 mm. Real CC0 library materials with procedural aging do the rest —
chipped red-orange paint on wood, scratched blued receiver, dusty tape.

Run:  python examples/rifle.py   →  rifle.glb + hero preview on stdout
"""
import math

import numpy as np

import aiglb

scene = aiglb.Scene("rifle")

# ---------------------------------------------------------------- materials
RECEIVER = aiglb.library_material(
    "texturecan/Metal0056", tint=(0.60, 0.61, 0.68), contrast=0.85,
    scratches=0.55, roughness=0.40, uv_scale=2.6, seed=7, name="receiver_blued")
DUST_COVER = aiglb.library_material(
    "texturecan/Metal0056", tint=(0.47, 0.48, 0.55), contrast=0.85,
    scratches=0.4, roughness=0.46, uv_scale=2.6, seed=19, name="dust_cover")
FURNITURE = aiglb.library_material(
    "ambientcg/PaintedWood006B", wear=0.45, uv_scale=1.7, seed=11,
    name="furniture_worn_red")
MAGAZINE = aiglb.library_material(
    "texturecan/Metal0059", tint=(0.36, 0.36, 0.40), scratches=0.45,
    wear=0.2, uv_scale=2.0, seed=23, name="magazine")
GUNMETAL = "gunmetal"
RUBBER = "rubber_black"
TAPE = aiglb.library_material(
    "ambientcg/PaintedMetal007", tint=(0.16, 0.24, 0.78), roughness=0.62,
    uv_scale=3.0, fingerprints=0.35, seed=31, name="blue_tape")
STAND = aiglb.color_material("#1a1c1f", roughness=0.88, name="stand")
STAND_BASE = aiglb.color_material("#101114", roughness=0.9, name="stand_base")

# rifle sits on a museum-style stand; barrel axis at this height
AXIS_Y = 0.165

# ------------------------------------------------------------- the rifle
# receiver & top cover
scene.add("receiver", aiglb.rounded_box(0.400, 0.050, 0.033, sharpness=7.0),
          RECEIVER, position=(0.0, AXIS_Y, 0.0))
scene.add("dust_cover", aiglb.rounded_box(0.384, 0.013, 0.0295, sharpness=6.0),
          DUST_COVER, position=(0.0, AXIS_Y + 0.0305, 0.0))

# barrel, gas tube, gas block, muzzle
scene.add("barrel", aiglb.cylinder(0.010, 0.185, 24), GUNMETAL,
          position=(0.290, AXIS_Y + 0.006, 0.0), rotation=(0, 0, 90))
scene.add("gas_tube", aiglb.cylinder(0.0108, 0.175, 24), RECEIVER,
          position=(0.183, AXIS_Y + 0.0355, 0.0), rotation=(0, 0, 90))
scene.add("gas_block", aiglb.rounded_box(0.034, 0.026, 0.029, sharpness=6.0),
          GUNMETAL, position=(0.268, AXIS_Y + 0.020, 0.0))
scene.add("muzzle", aiglb.cylinder(0.0124, 0.048, 24), GUNMETAL,
          position=(0.385, AXIS_Y + 0.006, 0.0), rotation=(0, 0, 90))
scene.add("muzzle_ring", aiglb.cylinder(0.0138, 0.014, 24), "black_metal",
          position=(0.405, AXIS_Y + 0.006, 0.0), rotation=(0, 0, 90))

# front sight block + post + protective wings
scene.add("front_sight_block", aiglb.rounded_box(0.026, 0.040, 0.029, sharpness=6.0),
          RECEIVER, position=(0.300, AXIS_Y + 0.048, 0.0))
scene.add("front_sight_post", aiglb.cylinder(0.0032, 0.024, 12), RUBBER,
          position=(0.300, AXIS_Y + 0.0775, 0.0))
for k, side in enumerate((1.0, -1.0)):
    scene.add(f"sight_wing_{k}", aiglb.rounded_box(0.018, 0.030, 0.004, sharpness=4.0),
              RECEIVER, position=(0.300, AXIS_Y + 0.074, side * 0.0125))

# rear sight block + tangent leaf
scene.add("rear_sight_block", aiglb.rounded_box(0.030, 0.016, 0.024, sharpness=6.0),
          GUNMETAL, position=(-0.030, AXIS_Y + 0.0395, 0.0))
scene.add("rear_sight_leaf", aiglb.rounded_box(0.024, 0.006, 0.012, sharpness=4.0),
          GUNMETAL, position=(-0.030, AXIS_Y + 0.0495, 0.0), rotation=(0, 0, 5))

# buttstock — slides 10 mm *into* the receiver, slight droop
stock_mesh = aiglb.rounded_box(0.225, 0.046, 0.027, sharpness=6.0)
scene.add("stock", stock_mesh, FURNITURE,
          position=(-0.318, AXIS_Y + 0.011, 0.0), rotation=(0, 0, -3.2))
scene.add("buttplate", aiglb.rounded_box(0.013, 0.052, 0.029, sharpness=5.0),
          RUBBER, position=(-0.4295, AXIS_Y + 0.0045, 0.0), rotation=(0, 0, -3.2))
for k, side in enumerate((1.0, -1.0)):
    scene.add(f"buttplate_screw_{k}", aiglb.cylinder(0.0026, 0.004, 10), GUNMETAL,
              position=(-0.437, AXIS_Y + 0.0045 + (0.014 if k else -0.014), 0.0),
              rotation=(0, 0, 90))

# pistol grip — tilted back like the real thing
scene.add("grip", aiglb.rounded_box(0.034, 0.102, 0.030, sharpness=5.5),
          FURNITURE, position=(-0.141, AXIS_Y - 0.064, 0.0), rotation=(0, 0, 16.5))

# trigger guard: rear strut + bottom bar + front strut
scene.add("guard_front", aiglb.rounded_box(0.008, 0.020, 0.013, sharpness=4.0),
          GUNMETAL, position=(-0.0145, AXIS_Y - 0.0435, 0.0))
scene.add("guard_bar", aiglb.rounded_box(0.077, 0.0085, 0.012, sharpness=4.0),
          GUNMETAL, position=(-0.057, AXIS_Y - 0.0525, 0.0))
scene.add("trigger", aiglb.rounded_box(0.005, 0.021, 0.010, sharpness=4.0),
          GUNMETAL, position=(-0.062, AXIS_Y - 0.040, 0.0), rotation=(0, 0, 10))

# magazine — 5 plates on a progressive arc (lean grows downward), like a 30-rd curve
mag_start = np.array([0.035, AXIS_Y - 0.030])
seg_h = 0.036
lean = math.radians(7.0)
step = 0.0315
pos = mag_start.copy()
plate_names = []
for i in range(5):
    plate = aiglb.rounded_box(0.030, seg_h + 0.008, 0.0175, sharpness=4.5)
    cx, cy = pos + np.array([math.sin(lean), -math.cos(lean)]) * (seg_h / 2)
    scene.add(f"mag_plate_{i}", plate, MAGAZINE,
              position=(cx, cy, 0.0), rotation=(0, 0, -math.degrees(lean)))
    plate_names.append(f"mag_plate_{i}")
    pos = pos + np.array([math.sin(lean), -math.cos(lean)]) * step
    lean += math.radians(8.4)
# baseplate at the end of the arc
scene.add("mag_base", aiglb.rounded_box(0.033, 0.013, 0.021, sharpness=4.0),
          GUNMETAL,
          position=(pos[0] + math.sin(lean) * 0.004, pos[1] - math.cos(lean) * 0.004, 0.0),
          rotation=(0, 0, -math.degrees(lean)))

# selector lever + pivot on the right side, charging handle on the left-rear
scene.add("selector", aiglb.rounded_box(0.030, 0.006, 0.006, sharpness=3.0),
          GUNMETAL, position=(-0.118, AXIS_Y + 0.002, 0.0185), rotation=(0, 0, -10))
scene.add("selector_pivot", aiglb.cylinder(0.0042, 0.004, 12), GUNMETAL,
          position=(-0.131, AXIS_Y + 0.0065, 0.0180), rotation=(90, 0, 0))
scene.add("charging_handle", aiglb.cylinder(0.0048, 0.016, 12), GUNMETAL,
          position=(-0.168, AXIS_Y + 0.018, -0.020), rotation=(90, 0, 0))

# cleaning rod under the barrel
scene.add("cleaning_rod", aiglb.cylinder(0.0022, 0.30, 10), GUNMETAL,
          position=(0.230, AXIS_Y - 0.0165, 0.0), rotation=(0, 0, 90))

# wooden handguards flanking the gas tube / barrel
scene.add("handguard_upper", aiglb.rounded_box(0.168, 0.024, 0.052, sharpness=6.0),
          FURNITURE, position=(0.152, AXIS_Y + 0.0360, 0.0))
scene.add("handguard_lower", aiglb.rounded_box(0.163, 0.026, 0.050, sharpness=6.0),
          FURNITURE, position=(0.152, AXIS_Y - 0.0285, 0.0))

# blue electrical tape wraps (2 on each handguard)
for i, tx in enumerate((0.183, 0.203)):
    scene.add(f"tape_upper_{i}", aiglb.rounded_box(0.0135, 0.0295, 0.0585, sharpness=4.0),
              TAPE, position=(tx, AXIS_Y + 0.0360, 0.0))
    scene.add(f"tape_lower_{i}", aiglb.rounded_box(0.0135, 0.0315, 0.0565, sharpness=4.0),
              TAPE, position=(tx, AXIS_Y - 0.0285, 0.0))

# receiver + stock rivets
for i, rx in enumerate((-0.148, -0.082, 0.020, 0.122)):
    for k, side in enumerate((1.0, -1.0)):
        scene.add(f"rivet_{i}_{k}", aiglb.cylinder(0.0026, 0.0050, 10),
                  GUNMETAL, position=(rx, AXIS_Y + 0.006, side * 0.0163),
                  rotation=(90, 0, 0))

# bayonet lug under the muzzle
scene.add("bayonet_lug", aiglb.rounded_box(0.016, 0.010, 0.010, sharpness=3.0),
          GUNMETAL, position=(0.352, AXIS_Y - 0.0115, 0.0))

# ------------------------------------------------------------- smiley sticker
from PIL import Image, ImageDraw


def _smiley() -> Image.Image:
    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([18, 18, s - 18, s - 18], fill=(247, 214, 63, 255),
              outline=(24, 24, 24, 255), width=13)
    d.rounded_rectangle([84, 88, 106, 132], radius=6, fill=(22, 22, 22, 255))
    d.rounded_rectangle([150, 88, 172, 132], radius=6, fill=(22, 22, 22, 255))
    d.arc([70, 92, 186, 208], start=12, end=168,
          fill=(22, 22, 22, 255), width=15)
    return img


sticker_tex = _smiley()
from trimesh.visual.material import PBRMaterial
sticker_mat = PBRMaterial(name="smiley_sticker",
                          baseColorTexture=sticker_tex,
                          baseColorFactor=[255, 255, 255, 255],
                          metallicFactor=0.0, roughnessFactor=0.45,
                          alphaMode="MASK", alphaCutoff=0.5)
for k, side in enumerate((1.0, -1.0)):
    scene.add(f"sticker_receiver_{k}",
              aiglb.box(0.036, 0.036, 0.0009),
              sticker_mat,
              position=(0.115, AXIS_Y - 0.002, side * 0.0171))
for k, side in enumerate((1.0, -1.0)):
    scene.add(f"sticker_guard_{k}",
              aiglb.box(0.030, 0.030, 0.0009),
              sticker_mat,
              position=(0.148, AXIS_Y + 0.0360, side * 0.0269))

# ------------------------------------------------------------- display stand
scene.add("stand_base", aiglb.rounded_box(0.95, 0.024, 0.30, sharpness=5.0),
          STAND_BASE, position=(-0.01, 0.012, 0.0))
for i, px in enumerate((-0.335, 0.235)):
    scene.add(f"stand_post_{i}", aiglb.cylinder(0.016, 0.115, 20),
              STAND, position=(px, 0.078, 0.0))
    for k, tilt in enumerate((28.0, -28.0)):
        scene.add(f"stand_cradle_{i}_{k}",
                  aiglb.rounded_box(0.010, 0.055, 0.075, sharpness=3.0),
                  STAND,
                  position=(px, 0.1335, 0.020 * (1 if tilt > 0 else -1)),
                  rotation=(tilt, 0, 0))

scene.save("rifle.glb")
