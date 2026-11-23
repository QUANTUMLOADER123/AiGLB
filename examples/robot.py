"""cute_robot.glb — a chunky cartoon robot demonstrating scene.add + transforms.

Run:  python3 examples/robot.py      (or: aiglb build examples/robot.py -o robot.glb)
"""
import math

import aiglb

scene = aiglb.Scene("cute_robot")

GUN = "gunmetal"
STEEL = "steel"
DARK = "black_metal"

# ---- legs ---------------------------------------------------------------
for side in (-1, 1):  # -1 = left, +1 = right
    x = 0.32 * side
    scene.add(f"foot_{side}", aiglb.box(0.42, 0.16, 0.58), DARK, position=(x, 0.08, 0.04))
    scene.add(f"leg_{side}", aiglb.cylinder(0.13, 0.62, 20), GUN, position=(x, 0.47, 0))
    scene.add(f"hip_{side}", aiglb.sphere(0.17, 20, 12), DARK, position=(x, 0.8, 0))

# ---- torso ----------------------------------------------------------------
scene.add("pelvis", aiglb.box(0.62, 0.22, 0.36), DARK, position=(0, 0.92, 0))
scene.add("torso", aiglb.box(0.78, 0.72, 0.44), STEEL, position=(0, 1.4, 0))
scene.add("belly_plate", aiglb.cylinder(0.2, 0.05, 28), GUN,
          position=(0, 1.36, 0.23), rotation=(90, 0, 0))
scene.add("core", aiglb.cylinder(0.1, 0.055, 24), "neon_cyan",
          position=(0, 1.36, 0.245), rotation=(90, 0, 0))
# chest bolts
bolt = aiglb.cylinder(0.035, 0.03, 10)
for i, (bx, by) in enumerate([(-0.31, 1.66), (0.31, 1.66), (-0.31, 1.14), (0.31, 1.14)]):
    scene.add(f"bolt_{i}", bolt, "chrome", position=(bx, by, 0.225), rotation=(90, 0, 0))

# ---- backpack + tanks -------------------------------------------------------
scene.add("backpack", aiglb.box(0.5, 0.5, 0.18), GUN, position=(0, 1.45, -0.31))
for side in (-1, 1):
    scene.add(f"tank_{side}", aiglb.capsule(0.09, 0.3, 16), DARK,
              position=(0.17 * side, 1.48, -0.44))

# ---- arms -------------------------------------------------------------------
for side in (-1, 1):
    x = 0.53 * side
    scene.add(f"shoulder_{side}", aiglb.sphere(0.16, 20, 12), GUN, position=(x, 1.66, 0))
    scene.add(f"upper_arm_{side}", aiglb.cylinder(0.1, 0.4, 16), STEEL,
              position=(x, 1.36, 0), rotation=(0, 0, 12 * side))
    scene.add(f"elbow_{side}", aiglb.sphere(0.11, 16, 10), DARK, position=(x + 0.09 * side, 1.17, 0))
    scene.add(f"forearm_{side}", aiglb.cylinder(0.09, 0.36, 16), GUN,
              position=(x + 0.13 * side, 0.94, 0), rotation=(0, 0, 6 * side))
    # claw: two fingers
    for finger in (-1, 1):
        scene.add(f"claw_{side}_{finger}", aiglb.box(0.05, 0.18, 0.05), DARK,
                  position=(x + 0.13 * side + 0.055 * finger, 0.68, 0.03 * finger),
                  rotation=(0, 0, -14 * finger))

# ---- head ---------------------------------------------------------------------
scene.add("neck", aiglb.cylinder(0.09, 0.1, 12), DARK, position=(0, 1.8, 0))
scene.add("head", aiglb.box(0.52, 0.42, 0.46), STEEL, position=(0, 2.06, 0))
scene.add("face", aiglb.box(0.4, 0.24, 0.03), DARK, position=(0, 2.06, 0.235))
for side in (-1, 1):
    scene.add(f"eye_{side}", aiglb.sphere(0.06, 16, 10), "neon_cyan",
              position=(0.11 * side, 2.06, 0.25))
scene.add("antenna", aiglb.cylinder(0.018, 0.28, 8), GUN, position=(0.16, 2.42, 0))
scene.add("beacon", aiglb.sphere(0.045, 14, 8), "neon_pink", position=(0.16, 2.58, 0))
for side in (-1, 1):  # side ear disks
    scene.add(f"ear_{side}", aiglb.cylinder(0.09, 0.07, 16), GUN,
              position=(0.30 * side, 2.08, 0), rotation=(0, 0, 90))

scene.save()
