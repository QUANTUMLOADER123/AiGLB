"""lighthouse.glb — striped lighthouse on granite rocks in the sea.

Run:  python3 examples/lighthouse.py   (or: aiglb build examples/lighthouse.py -o lighthouse.glb)
"""
import aiglb

scene = aiglb.Scene("lighthouse")

# sea + rock base
scene.add("sea", aiglb.plane(9, 9, uv_scale=0.08), "water", position=(0, 0.0, 0))
scene.add("rock", aiglb.sphere(2.2, 24, 14), "granite", position=(0, -1.12, 0), scale=(1.25, 0.62, 1.0))

# tower: tapered lathe profile, red/white stripes
profile = [(0.0, 0.0), (1.05, 0.02), (0.98, 0.35), (0.72, 3.4), (0.78, 3.6), (0.92, 3.68), (0.92, 3.78), (0.0, 3.78)]
scene.add("tower", aiglb.lathe(profile, segments=44, smooth=True, uv_scale=0.35),
          aiglb.texture_material("stripes", c1=(0.92, 0.18, 0.16), c2=(0.95, 0.94, 0.92),
                                 roughness=0.6, name="lighthouse_stripes"),
          position=(0, 0.35, 0))

# lamp room: dark gallery + glowing lamp + black roof
scene.add("gallery", aiglb.cylinder(1.02, 0.16, 32), "black_metal", position=(0, 4.2, 0))
scene.add("lamp_room", aiglb.cylinder(0.72, 0.95, 32, capped=False), "glass", position=(0, 4.75, 0))
scene.add("lamp", aiglb.cylinder(0.4, 0.62, 24), "lamp_warm", position=(0, 4.75, 0))
scene.add("roof", aiglb.cone(0.95, 0.9, 32), "black_metal", position=(0, 5.66, 0))
scene.add("beacon", aiglb.sphere(0.09, 16, 10), "neon_pink", position=(0, 6.18, 0))

# door + little window
scene.add("door", aiglb.box(0.72, 1.15, 0.1), "dark_wood", position=(0, 0.95, 1.03))
scene.add("window", aiglb.box(0.34, 0.34, 0.08), "lamp_warm", position=(0, 2.5, 0.78))

scene.save()
