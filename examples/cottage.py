"""cottage.glb — brick cottage with a plank gable roof, warm windows, trees.

Run:  python3 examples/cottage.py   (or: aiglb build examples/cottage.py -o cottage.glb)
"""
import aiglb

scene = aiglb.Scene("cottage")

# ---- ground & walkway ------------------------------------------------------
scene.add("ground", aiglb.plane(15, 15, uv_scale=0.3), "grass", position=(0, -0.01, 0))
scene.add("walkway", aiglb.plane(1.4, 2.6, uv_scale=0.8), "concrete", position=(0, 0.0, 3.4))

# ---- walls: rectangular footprint, brick -----------------------------------
footprint = [(-3.2, -2.3), (3.2, -2.3), (3.2, 2.3), (-3.2, 2.3)]
scene.add("walls", aiglb.extrude(footprint, 2.35, uv_scale=0.55), "brick")

# ---- gable roof: two wedges mirrored across the ridge (z = 0) --------------
orth = 2.45          # horizontal reach of each slope (overhang beyond walls)
roof_h = 1.5
half = aiglb.wedge(7.6, roof_h, orth)   # vertical face at z = -orth/2 ... see docs
scene.add("roof_front", half, "dark_wood",
          position=(0, 2.35, orth / 2))                       # slope faces +Z
scene.add("roof_back", half, "dark_wood",
          position=(0, 2.35, -orth / 2), rotation=(0, 180, 0))  # slope faces -Z
# ridge beam
scene.add("ridge", aiglb.box(7.7, 0.09, 0.22), "walnut", position=(0, 2.35 + roof_h - 0.02, 0))

# ---- chimney (pokes through the back slope) ---------------------------------
scene.add("chimney", aiglb.box(0.62, 1.6, 0.62), "brick", position=(1.9, 3.1, -1.05))
scene.add("chimney_cap", aiglb.box(0.78, 0.12, 0.78), "concrete", position=(1.9, 3.94, -1.05))

# ---- door + step -------------------------------------------------------------
scene.add("door", aiglb.box(1.0, 2.0, 0.1), "oak", position=(0, 1.0, 2.33))
scene.add("door_knob", aiglb.sphere(0.045, 12, 8), "gold", position=(0.36, 1.0, 2.39))
scene.add("step", aiglb.box(1.5, 0.18, 0.5), "concrete", position=(0, 0.09, 2.55))

# ---- windows (warm emissive panes + dark frames) ------------------------------
def add_window(name, x, y, z, rot_y=0.0):
    scene.add(f"{name}_frame", aiglb.box(0.92, 1.0, 0.06), "black_metal",
              position=(x, y, z), rotation=(0, rot_y, 0))
    scene.add(f"{name}_pane", aiglb.box(0.74, 0.82, 0.075), "lamp_warm",
              position=(x, y, z), rotation=(0, rot_y, 0))
    # mullions
    scene.add(f"{name}_bar_v", aiglb.box(0.05, 0.82, 0.085), "black_metal",
              position=(x, y, z), rotation=(0, rot_y, 0))
    scene.add(f"{name}_bar_h", aiglb.box(0.74, 0.05, 0.085), "black_metal",
              position=(x, y, z), rotation=(0, rot_y, 0))

add_window("win_fl", -1.75, 1.45, 2.31)
add_window("win_fr", 1.75, 1.45, 2.31)
add_window("win_bl", -1.75, 1.45, -2.31)
add_window("win_br", 1.75, 1.45, -2.31)
# side windows on ±X walls
for sx, side in ((1, "r"), (-1, "l")):
    scene.add(f"win_{side}_frame", aiglb.box(0.06, 1.0, 0.92), "black_metal", position=(3.21 * sx, 1.45, 0.2))
    scene.add(f"win_{side}_pane", aiglb.box(0.075, 0.82, 0.74), "lamp_warm", position=(3.21 * sx, 1.45, 0.2))

# ---- trees flanking the house --------------------------------------------------
for i, (tx, tz, s) in enumerate([(-5.2, 1.6, 1.0), (5.4, -0.6, 1.25), (-4.6, -3.4, 0.8)]):
    scene.add(f"trunk_{i}", aiglb.cylinder(0.16 * s, 1.1 * s, 10), "bark", position=(tx, 0.55 * s, tz))
    scene.add(f"leaves_low_{i}", aiglb.cone(1.05 * s, 1.9 * s, 14), "leaf_green", position=(tx, 2.0 * s, tz))
    scene.add(f"leaves_top_{i}", aiglb.cone(0.7 * s, 1.3 * s, 14), "leaf_green", position=(tx, 3.1 * s, tz))

scene.save()
