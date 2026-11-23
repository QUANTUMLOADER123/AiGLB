"""mug.glb — lathed ceramic mug + saucer, demonstrating the lathe primitive.

Run:  python3 examples/mug.py   (or: aiglb build examples/mug.py -o mug.glb)
"""
import aiglb

scene = aiglb.Scene("mug")

# closed profile: outer bottom -> outer wall -> rim -> inner wall -> inner floor
mug_profile = [
    (0.00, 0.00), (0.30, 0.00), (0.42, 0.015), (0.455, 0.06),
    (0.465, 0.30), (0.45, 0.52), (0.475, 0.635), (0.455, 0.665),   # outer lip
    (0.385, 0.665), (0.40, 0.60), (0.385, 0.30), (0.375, 0.115),   # inner lip -> inner floor
    (0.00, 0.115),
]
teal = aiglb.color_material("#2e6f6a", roughness=0.22, name="mug_teal")
scene.add("mug", aiglb.lathe(mug_profile, segments=56, smooth=True), teal)

# handle: torus embedded in the side
scene.add("handle", aiglb.torus(0.225, 0.052, 40, 18), teal,
          position=(0.47, 0.35, 0), rotation=(90, 0, 0))

# coffee surface: dark disk slightly below the rim
scene.add("coffee", aiglb.cylinder(0.375, 0.012, 40), aiglb.color_material("#17100a", roughness=0.12, name="coffee"), position=(0, 0.52, 0))

# saucer: shallow lathe dish
saucer_profile = [
    (0.00, 0.00), (0.42, 0.00), (0.62, 0.02), (0.78, 0.075), (0.82, 0.105),
    (0.78, 0.115), (0.55, 0.055), (0.00, 0.035),
]
scene.add("saucer", aiglb.lathe(saucer_profile, segments=56, smooth=True),
          aiglb.color_material("#e8e4da", roughness=0.18, name="saucer_white"),
          position=(0, -0.004, 0))

scene.save()
